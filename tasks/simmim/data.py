import torch
import numpy as np
from torchvision.datasets import ImageFolder
from transformers.models.vit.feature_extraction_vit import ViTFeatureExtractor
from torchvision.transforms import Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor


class SimMIMDataset(ImageFolder):
    def __init__(self, root: str, image_size=192,
                 model_patch_size: int = 4, mask_patch_size: int = 32, mask_ratio: float = 0.5,
                 feature_extractor=ViTFeatureExtractor()):
        assert isinstance(image_size, int)
        super().__init__(root)
        self.feature_extractor = feature_extractor
        self.mask_generator = MaskGenerator(
            input_size=image_size,
            mask_patch_size=mask_patch_size,
            model_patch_size=model_patch_size,
            mask_ratio=mask_ratio,
        )

        self.transform_image = Compose(
            [
                Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                RandomResizedCrop(image_size, scale=(0.67, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
            ]
        )

    def __getitem__(self, index: int):
        image, _ = super().__getitem__(index)
        return dict(pixel_values=self.transform_image(image), mask=self.mask_generator())


class MaskGenerator:
    """
    A class to generate boolean masks for the pretraining task.
    A mask is a 1D tensor of shape (model_patch_size**2,) where the value is either 0 or 1,
    where 1 indicates "masked".
    """

    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        if self.input_size % self.mask_patch_size != 0:
            raise ValueError("Input size must be divisible by mask patch size")
        if self.mask_patch_size % self.model_patch_size != 0:
            raise ValueError("Mask patch size must be divisible by model patch size")

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return torch.tensor(mask.flatten())


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    mask = torch.stack([example["mask"] for example in examples])
    return {"pixel_values": pixel_values, "bool_masked_pos": mask}
