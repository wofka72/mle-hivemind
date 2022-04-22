import io
import itertools
import logging
import random
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import requests
import torch
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
from torchvision.transforms import Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor
from transformers.models.vit.feature_extraction_vit import ViTFeatureExtractor


# Hide urllib warnings
logging.getLogger('urllib3.connection').setLevel(logging.ERROR)
logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)

# Hide PIL warnings
for message in [
    "Palette images with Transparency expressed in bytes should be converted to RGBA images",
    "image file could not be identified because WEBP support not installed",
]:
    warnings.filterwarnings("ignore", category=UserWarning, message=message)


executor = ThreadPoolExecutor(32)


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


def download_image(url, transform):
    try:
        r = requests.get(url, timeout=1)
        img = Image.open(io.BytesIO(r.content))
        return transform(img)
    except Exception as e:
        logging.debug('Failed to download `{url}`', exception=True)
        return None


def preprocess_batch(batch, mask_generator, transform):
    filtered = [
        (
            nsfw == 'UNLIKELY' and
            orig_width > 0 and orig_height > 0
        ) for nsfw, orig_width, orig_height in
        zip(batch['NSFW'], batch['WIDTH'], batch['HEIGHT'])
    ]
    logging.debug(f'{np.mean(filtered) * 100:.1f}% of examples left after filtering')

    pixel_values = [item
                    for item in executor.map(lambda url: download_image(url, transform),
                                             itertools.compress(batch['URL'], filtered))
                    if item is not None]
    masks = [mask_generator() for item in pixel_values]
    return dict(pixel_values=pixel_values, mask=masks)


def make_dataset(
    *,
    image_size = 192,
    model_patch_size: int = 4,
    mask_patch_size: int = 32,
    mask_ratio: float = 0.5,
    feature_extractor=ViTFeatureExtractor(),

    shuffle_buffer_size: int = 4096,
    shuffle_seed: Optional[int] = None,
    preprocessing_batch_size: int = 4096,
):
    mask_generator = MaskGenerator(
        input_size=image_size,
        mask_patch_size=mask_patch_size,
        model_patch_size=model_patch_size,
        mask_ratio=mask_ratio,
    )
    transform = Compose([
        Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        RandomResizedCrop(image_size, scale=(0.67, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
    ])

    ds = load_dataset('laion/laion400m', split='train', streaming=True)
    ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=shuffle_seed)
    ds = ds.map(lambda batch: preprocess_batch(batch, mask_generator, transform),
                batch_size=preprocessing_batch_size,
                batched=True,
                remove_columns=['SAMPLE_ID', 'URL', 'TEXT', 'HEIGHT', 'WIDTH', 'LICENSE', 'NSFW', 'similarity'])
    ds = ds.with_format('torch')
    return ds
