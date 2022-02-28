import ctypes
import math
import multiprocessing as mp
import os
from pathlib import Path

import torch
from transformers import SwinConfig, AutoModelForMaskedImageModeling, FEATURE_EXTRACTOR_MAPPING

import hivemind
import torch.distributed
import transformers
from torch.optim.lr_scheduler import LambdaLR
from tasks.base import ParamGroups, LRSchedulerBase

from arguments import BasePeerArguments, CollaborativeArguments, HFTrainerArguments
from lib.training.lamb_8bit import CPULAMB8Bit
from tasks.base import TrainingTaskBase, register_task

from .data import SimMIMDataset

hivemind.use_hivemind_log_handler("in_root_logger")
logger = hivemind.get_logger()


FEATURE_EXTRACTOR_TYPES = {
    conf.model_type: feature_extractor_class
    for conf, feature_extractor_class in FEATURE_EXTRACTOR_MAPPING.items()
}


@register_task("mim")
class MaskedImageModelingTask(TrainingTaskBase):
    """A container that defines the training config, model, tokenizer, optimizer and other local training utilities"""

    _dht = _optimizer = _training_dataset = _authorizer = None

    def __init__(
        self, peer_args: BasePeerArguments, trainer_args: HFTrainerArguments, collab_args: CollaborativeArguments
    ):
        transformers.set_seed(trainer_args.seed)  # seed used for initialization
        self.dataset_path = peer_args.dataset_path
        self.config = SwinConfig(
            image_size=192,
            patch_size=4,
            embed_dim=512,
            depths=[2, 2, 26, 2],
            num_heads=[16, 32, 64, 128],
            window_size=6,
        )
        self.feature_extractor = FEATURE_EXTRACTOR_TYPES[self.config.model_type]()

        output_dir = Path(trainer_args.output_dir)
        logger.info(f'Checkpoint dir {output_dir}, contents {list(output_dir.glob("checkpoint*"))}')
        latest_checkpoint_dir = max(output_dir.glob("checkpoint*"), default=None, key=os.path.getctime)

        if latest_checkpoint_dir is None:
            logger.info(f"Creating model")
            model = AutoModelForMaskedImageModeling.from_config(self.config)
        else:
            logger.info(f"Loading model from {latest_checkpoint_dir}")
            model = AutoModelForMaskedImageModeling.from_pretrained(latest_checkpoint_dir)
        if trainer_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        super().__init__(model, peer_args, trainer_args, collab_args)
        self.current_sequence_length = mp.Value(ctypes.c_int64, self.trainer_args.max_sequence_length)

    def _make_param_groups(self) -> ParamGroups:
        no_decay = ["bias", "norm.weight"]
        return [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(n.endswith(nd) for nd in no_decay)],
                "weight_decay": self.trainer_args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(n.endswith(nd) for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

    def _make_base_optimizer(self, param_groups: ParamGroups) -> torch.optim.Optimizer:
        return CPULAMB8Bit(
            param_groups,
            lr=self.trainer_args.learning_rate,
            betas=(self.trainer_args.adam_beta1, self.trainer_args.adam_beta2),
            min_8bit_size=self.trainer_args.min_8bit_size,
            max_grad_norm=self.trainer_args.max_grad_norm,
            clamp_value=self.trainer_args.clamp_value,
            eps=self.trainer_args.adam_epsilon,
            weight_decay=self.trainer_args.weight_decay,
            reuse_grad_buffers=True,
            bias_correction=True,
        )

    def _make_scheduler(self, optimizer: torch.optim.Optimizer) -> LRSchedulerBase:
        num_warmup_steps = self.trainer_args.warmup_steps
        num_training_steps = self.trainer_args.total_steps
        min_learning_rate = self.trainer_args.min_learning_rate

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            decay_ratio = min(1.0, (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps))
            return max(min_learning_rate, 0.5 * (math.cos(math.pi * decay_ratio) + 1.0))

        return LambdaLR(optimizer, lr_lambda)

    @property
    def training_dataset(self):
        if self._training_dataset is None:
            self._training_dataset = SimMIMDataset(
                self.dataset_path,
                image_size=self.config.image_size,
                model_patch_size=self.config.patch_size,
                mask_patch_size=32, mask_ratio=0.5,
                feature_extractor=self.feature_extractor
            )
        return self._training_dataset

    def on_step_end(self):
        pass

    @property
    def data_collator(self):
        return collate_fn


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    mask = torch.stack([example["mask"] for example in examples])
    return {"pixel_values": pixel_values, "bool_masked_pos": mask}
