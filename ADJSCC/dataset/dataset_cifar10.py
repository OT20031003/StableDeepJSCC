"""CIFAR-10 data access for the PyTorch ADJSCC implementation."""

from pathlib import Path
from typing import Tuple

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from .common import ChannelConditionDataset


DEFAULT_ROOT = Path(__file__).resolve().parent / "cifar10"


def _cifar10(root: str, train: bool, download: bool) -> Dataset:
    return CIFAR10(
        root=str(root), train=train, transform=ToTensor(), download=download
    )


def load_cifar10(
    root: str = str(DEFAULT_ROOT), train: bool = True, download: bool = True
) -> Dataset:
    """Return the underlying torchvision CIFAR-10 image dataset."""
    return _cifar10(root, train, download)


def _pair(
    root: str,
    download: bool,
    snr_db=None,
    snr_low=None,
    snr_high=None,
    fading: bool = False,
) -> Tuple[Tuple[Dataset, int], Tuple[Dataset, int]]:
    train_images = _cifar10(root, True, download)
    test_images = _cifar10(root, False, download)
    options = dict(
        snr_db=snr_db,
        snr_low=snr_low,
        snr_high=snr_high,
        fading=fading,
    )
    train = ChannelConditionDataset(train_images, **options)
    test = ChannelConditionDataset(test_images, **options)
    return (train, len(train)), (test, len(test))


def get_dataset_snr(
    snr_db: float, root: str = str(DEFAULT_ROOT), download: bool = True
):
    return _pair(root, download, snr_db=snr_db)


def get_dataset_snr_and_h(
    snr_db: float, root: str = str(DEFAULT_ROOT), download: bool = True
):
    return _pair(root, download, snr_db=snr_db, fading=True)


def get_dataset_snr_range(
    snr_db_low: float,
    snr_db_high: float,
    root: str = str(DEFAULT_ROOT),
    download: bool = True,
):
    return _pair(root, download, snr_low=snr_db_low, snr_high=snr_db_high)


def get_dataset_snr_range_and_h(
    snr_db_low: float,
    snr_db_high: float,
    root: str = str(DEFAULT_ROOT),
    download: bool = True,
):
    return _pair(
        root,
        download,
        snr_low=snr_db_low,
        snr_high=snr_db_high,
        fading=True,
    )


def get_test_dataset_burst(
    snr_db: float,
    b_prob: float,
    b_stddev: float,
    root: str = str(DEFAULT_ROOT),
    download: bool = True,
):
    images = _cifar10(root, False, download)
    dataset = ChannelConditionDataset(
        images,
        snr_db=snr_db,
        burst_prob=b_prob,
        burst_stddev=b_stddev,
    )
    return dataset, len(dataset)
