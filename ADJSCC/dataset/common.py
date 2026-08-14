"""Shared dataset wrappers for channel-condition generation."""

from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


def image_to_tensor(image: Any) -> Tensor:
    """Convert PIL/NumPy/tensor images to float CHW tensors in ``[0, 1]``."""
    if torch.is_tensor(image):
        tensor = image
        if tensor.ndim != 3:
            raise ValueError("an image tensor must be 3-dimensional")
        if tensor.shape[0] not in (1, 3, 4) and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        tensor = tensor.float()
        if tensor.numel() and tensor.max().item() > 1.0:
            tensor = tensor / 255.0
        return tensor[:3]
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise TypeError("unsupported image type: {}".format(type(image).__name__))
    return TF.to_tensor(image.convert("RGB"))


class ChannelConditionDataset(Dataset):
    """Attach SNR, fading, or burst conditions to an image dataset."""

    def __init__(
        self,
        images: Dataset,
        snr_db: Optional[float] = None,
        snr_low: Optional[float] = None,
        snr_high: Optional[float] = None,
        fading: bool = False,
        burst_prob: Optional[float] = None,
        burst_stddev: Optional[float] = None,
    ) -> None:
        if snr_db is None and (snr_low is None or snr_high is None):
            raise ValueError("provide either snr_db or both snr_low and snr_high")
        if snr_db is not None and (snr_low is not None or snr_high is not None):
            raise ValueError("fixed and ranged SNR settings are mutually exclusive")
        if snr_low is not None and snr_high is not None and snr_high < snr_low:
            raise ValueError("snr_high must be greater than or equal to snr_low")
        if (burst_prob is None) != (burst_stddev is None):
            raise ValueError("burst_prob and burst_stddev must be provided together")
        self.images = images
        self.snr_db = snr_db
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.fading = fading
        self.burst_prob = burst_prob
        self.burst_stddev = burst_stddev

    def __len__(self) -> int:
        return len(self.images)

    def _snr(self) -> Tensor:
        if self.snr_db is not None:
            value = float(self.snr_db)
        else:
            value = float(
                torch.empty(1).uniform_(float(self.snr_low), float(self.snr_high)).item()
            )
        return torch.tensor([value], dtype=torch.float32)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        sample = self.images[index]
        image = sample[0] if isinstance(sample, (tuple, list)) else sample
        output = {"image": image_to_tensor(image), "snr_db": self._snr()}
        if self.fading:
            scale = 2.0 ** -0.5
            output["h_real"] = torch.randn(1, dtype=torch.float32) * scale
            output["h_imag"] = torch.randn(1, dtype=torch.float32) * scale
        if self.burst_prob is not None:
            output["b_prob"] = torch.tensor(
                [self.burst_prob], dtype=torch.float32
            )
            output["b_stddev"] = torch.tensor(
                [self.burst_stddev], dtype=torch.float32
            )
        return output


class TensorImageDataset(Dataset):
    """A minimal image-only dataset backed by an NCHW tensor."""

    def __init__(self, images: Tensor) -> None:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [N, 3, H, W]")
        self.images = images

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> Tensor:
        return self.images[index]
