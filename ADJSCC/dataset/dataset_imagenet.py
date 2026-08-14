"""ImageNet patches, TFRecord compatibility, and Kodak evaluation images."""

import io
import os
import struct
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .common import ChannelConditionDataset


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGENET_ROOT = BASE_DIR / "imagenet"
DEFAULT_KODAK_ROOT = BASE_DIR / "kodak"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_varint(data: bytes, position: int) -> Tuple[int, int]:
    value = 0
    shift = 0
    while position < len(data):
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, position
        shift += 7
        if shift >= 70:
            break
    raise ValueError("invalid protobuf varint")


def _protobuf_fields(data: bytes):
    """Yield ``(field_number, wire_type, value)`` without TensorFlow/protobuf."""
    position = 0
    while position < len(data):
        key, position = _read_varint(data, position)
        field_number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, position = _read_varint(data, position)
        elif wire_type == 1:
            end = position + 8
            value = data[position:end]
            position = end
        elif wire_type == 2:
            length, position = _read_varint(data, position)
            end = position + length
            if end > len(data):
                raise ValueError("truncated protobuf field")
            value = data[position:end]
            position = end
        elif wire_type == 5:
            end = position + 4
            value = data[position:end]
            position = end
        else:
            raise ValueError("unsupported protobuf wire type {}".format(wire_type))
        yield field_number, wire_type, value


def _extract_encoded_image(example: bytes) -> bytes:
    """Extract ``image/encoded`` from a serialized ``tf.train.Example``."""
    features_message = None
    for field, wire_type, value in _protobuf_fields(example):
        if field == 1 and wire_type == 2:
            features_message = value
            break
    if features_message is None:
        raise ValueError("TFRecord Example does not contain Features")

    for field, wire_type, entry in _protobuf_fields(features_message):
        if field != 1 or wire_type != 2:
            continue
        key = None
        feature = None
        for entry_field, entry_wire, entry_value in _protobuf_fields(entry):
            if entry_field == 1 and entry_wire == 2:
                key = entry_value.decode("utf-8")
            elif entry_field == 2 and entry_wire == 2:
                feature = entry_value
        if key != "image/encoded" or feature is None:
            continue
        for feature_field, feature_wire, bytes_list in _protobuf_fields(feature):
            if feature_field != 1 or feature_wire != 2:
                continue
            for list_field, list_wire, encoded in _protobuf_fields(bytes_list):
                if list_field == 1 and list_wire == 2:
                    return encoded
    raise ValueError("TFRecord Example does not contain image/encoded")


def _scan_tfrecord(path: Path) -> List[Tuple[int, int]]:
    """Return data offsets and lengths from a standard TFRecord file."""
    records = []
    with path.open("rb") as handle:
        while True:
            length_bytes = handle.read(8)
            if not length_bytes:
                break
            if len(length_bytes) != 8:
                raise ValueError("truncated TFRecord length in {}".format(path))
            length = struct.unpack("<Q", length_bytes)[0]
            if len(handle.read(4)) != 4:
                raise ValueError("truncated TFRecord length CRC in {}".format(path))
            offset = handle.tell()
            handle.seek(length, os.SEEK_CUR)
            if len(handle.read(4)) != 4:
                raise ValueError("truncated TFRecord data CRC in {}".format(path))
            records.append((offset, length))
    return records


def get_num_samples(tfrecord_paths: Iterable[str]) -> int:
    return sum(len(_scan_tfrecord(Path(path))) for path in tfrecord_paths)


class PatchTransform:
    def __init__(self, patch_size: int = 128, augment: bool = True) -> None:
        self.patch_size = patch_size
        self.augment = augment

    def __call__(self, image: Image.Image) -> Tensor:
        image = image.convert("RGB")
        width, height = image.size
        if min(width, height) < self.patch_size:
            scale = self.patch_size / float(min(width, height))
            image = TF.resize(
                image,
                [int(round(height * scale)), int(round(width * scale))],
            )
        width, height = image.size
        if self.augment:
            top = int(torch.randint(0, height - self.patch_size + 1, (1,)).item())
            left = int(torch.randint(0, width - self.patch_size + 1, (1,)).item())
        else:
            top = (height - self.patch_size) // 2
            left = (width - self.patch_size) // 2
        image = TF.crop(image, top, left, self.patch_size, self.patch_size)
        if self.augment and torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
        return TF.to_tensor(image)


class RecursiveImageDataset(Dataset):
    """Read images recursively without requiring ImageFolder class folders."""

    def __init__(
        self, root: str, patch_size: int = 128, augment: bool = True
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError("ImageNet directory not found: {}".format(self.root))
        self.paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(
                "no supported images found below {}".format(self.root)
            )
        self.transform = PatchTransform(patch_size, augment)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        with Image.open(str(self.paths[index])) as image:
            return self.transform(image)


class TFRecordImageDataset(Dataset):
    """Read TensorFlow ``Example`` TFRecords using only Python and Pillow.

    CRC fields are skipped but not verified. The custom reader keeps existing
    ``Imagenet_patch_128.record`` files usable without installing TensorFlow.
    """

    def __init__(
        self, path: str, patch_size: int = 128, augment: bool = True
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError("TFRecord file not found: {}".format(self.path))
        self.records = _scan_tfrecord(self.path)
        if not self.records:
            raise ValueError("TFRecord file is empty: {}".format(self.path))
        self.transform = PatchTransform(patch_size, augment)
        self._handle = None

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()

    def __getitem__(self, index: int) -> Tensor:
        if self._handle is None:
            self._handle = self.path.open("rb")
        offset, length = self.records[index]
        self._handle.seek(offset)
        example = self._handle.read(length)
        if len(example) != length:
            raise ValueError("truncated TFRecord data at record {}".format(index))
        encoded = _extract_encoded_image(example)
        with Image.open(io.BytesIO(encoded)) as image:
            return self.transform(image)


def _resolve_imagenet_source(root: str) -> Path:
    path = Path(root)
    if path.is_file():
        return path
    legacy_record = path / "Imagenet_patch_128.record"
    if legacy_record.is_file():
        return legacy_record
    return path


def image_dataset(
    root: str = str(DEFAULT_IMAGENET_ROOT),
    patch_size: int = 128,
    augment: bool = True,
) -> Dataset:
    source = _resolve_imagenet_source(root)
    if source.is_file():
        return TFRecordImageDataset(str(source), patch_size, augment)
    return RecursiveImageDataset(str(source), patch_size, augment)


def get_dataset_snr(
    snr_db: float,
    root: str = str(DEFAULT_IMAGENET_ROOT),
    patch_size: int = 128,
):
    images = image_dataset(root, patch_size, augment=True)
    dataset = ChannelConditionDataset(images, snr_db=snr_db)
    return dataset, len(dataset)


def get_dataset_snr_range(
    snr_db_low: float,
    snr_db_high: float,
    root: str = str(DEFAULT_IMAGENET_ROOT),
    patch_size: int = 128,
    fading: bool = False,
):
    images = image_dataset(root, patch_size, augment=True)
    dataset = ChannelConditionDataset(
        images,
        snr_low=snr_db_low,
        snr_high=snr_db_high,
        fading=fading,
    )
    return dataset, len(dataset)


def get_dataset_snr_and_h(
    snr_db: float,
    root: str = str(DEFAULT_IMAGENET_ROOT),
    patch_size: int = 128,
):
    images = image_dataset(root, patch_size, augment=True)
    dataset = ChannelConditionDataset(images, snr_db=snr_db, fading=True)
    return dataset, len(dataset)


def get_dataset_snr_range_and_h(
    snr_db_low: float,
    snr_db_high: float,
    root: str = str(DEFAULT_IMAGENET_ROOT),
    patch_size: int = 128,
):
    return get_dataset_snr_range(
        snr_db_low, snr_db_high, root, patch_size, fading=True
    )


def load_image(path: str, transpose_portrait: bool = False) -> Tensor:
    with Image.open(path) as image:
        tensor = TF.to_tensor(image.convert("RGB"))
    if transpose_portrait and tensor.shape[-2] > tensor.shape[-1]:
        tensor = tensor.transpose(-2, -1)
    return tensor


def get_kodak(root: str = str(DEFAULT_KODAK_ROOT)) -> Tensor:
    root_path = Path(root)
    paths = [root_path / "kodim{:02d}.png".format(index) for index in range(1, 25)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Kodak dataset is incomplete; first missing image: {}".format(missing[0])
        )
    return torch.stack(
        [load_image(str(path), transpose_portrait=True) for path in paths]
    )
