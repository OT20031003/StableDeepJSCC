"""Train ADJSCC in the frozen Stable Diffusion VAE latent space on FFHQ."""

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config

from training import (
    load_checkpoint,
    parameter_count,
    resolve_device,
    save_checkpoint,
    save_image,
    save_json,
    seed_everything,
)
from util_channel import Channel
from util_module import AFModule, GFRModule


DEFAULT_DATA_DIR = REPO_ROOT.parent / "datasets" / "ffhq_train_70k"
DEFAULT_SD_CONFIG = REPO_ROOT / "configs" / "stable-diffusion" / "v1-inference.yaml"
DEFAULT_SD_CHECKPOINT = (
    REPO_ROOT / "models" / "ldm" / "stable-diffusion-v1" / "model.ckpt"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "sd_vae_ffhq"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    RESAMPLE_LANCZOS = Image.LANCZOS

try:
    FLIP_LEFT_RIGHT = Image.Transpose.FLIP_LEFT_RIGHT
except AttributeError:  # Pillow < 9.1
    FLIP_LEFT_RIGHT = Image.FLIP_LEFT_RIGHT


def discover_images(root: str) -> List[Path]:
    directory = Path(root)
    if not directory.is_dir():
        raise FileNotFoundError("dataset directory not found: {}".format(directory))
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError("no supported images found below {}".format(directory))
    return paths


def split_image_paths(
    paths: Sequence[Path], val_count: int, split_seed: int
) -> Tuple[List[Path], List[Path]]:
    if val_count <= 0 or val_count >= len(paths):
        raise ValueError(
            "val_count must be between 1 and {}, got {}".format(
                len(paths) - 1, val_count
            )
        )
    shuffled = list(paths)
    random.Random(split_seed).shuffle(shuffled)
    return shuffled[val_count:], shuffled[:val_count]


class FFHQDataset(Dataset):
    """Load FFHQ images as RGB tensors in ``[-1, 1]``."""

    def __init__(
        self, paths: Sequence[Path], image_size: int = 512, augment: bool = False
    ) -> None:
        if image_size <= 0 or image_size % 32:
            raise ValueError("image_size must be a positive multiple of 32")
        if not paths:
            raise ValueError("FFHQDataset requires at least one image")
        self.paths = list(paths)
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, object]:
        path = self.paths[index]
        with Image.open(str(path)) as source:
            image = source.convert("RGB")
            width, height = image.size
            side = min(width, height)
            left = (width - side) // 2
            top = (height - side) // 2
            image = image.crop((left, top, left + side, top + side))
            if image.size != (self.image_size, self.image_size):
                image = image.resize(
                    (self.image_size, self.image_size), RESAMPLE_LANCZOS
                )
            if self.augment and torch.rand(1).item() < 0.5:
                image = image.transpose(FLIP_LEFT_RIGHT)
            tensor = TF.to_tensor(image) * 2.0 - 1.0
        return {"image": tensor, "path": str(path)}


class FrozenFirstStageVAE(nn.Module):
    """Frozen SD first-stage model with LatentDiffusion-compatible scaling."""

    def __init__(self, first_stage_model: nn.Module, scale_factor: float) -> None:
        super().__init__()
        self.first_stage_model = first_stage_model
        self.register_buffer(
            "scale_factor", torch.tensor(float(scale_factor), dtype=torch.float32)
        )
        for parameter in self.first_stage_model.parameters():
            parameter.requires_grad_(False)
        self.eval()

    @classmethod
    def from_stable_diffusion(
        cls, config_path: str, checkpoint_path: str, device: torch.device
    ) -> "FrozenFirstStageVAE":
        config_file = Path(config_path)
        checkpoint_file = Path(checkpoint_path)
        if not config_file.is_file():
            raise FileNotFoundError("SD config not found: {}".format(config_file))
        if not checkpoint_file.is_file():
            raise FileNotFoundError(
                "Stable Diffusion checkpoint not found: {}".format(checkpoint_file)
            )

        config = OmegaConf.load(str(config_file))
        first_stage_config = config.model.params.first_stage_config
        scale_factor = float(config.model.params.scale_factor)
        first_stage_model = instantiate_from_config(first_stage_config)

        print("Loading Stable Diffusion VAE from {}".format(checkpoint_file))
        checkpoint = torch.load(str(checkpoint_file), map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        prefix = "first_stage_model."
        first_stage_state = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not first_stage_state:
            raise RuntimeError(
                "checkpoint contains no parameters with prefix {!r}".format(prefix)
            )
        missing, unexpected = first_stage_model.load_state_dict(
            first_stage_state, strict=False
        )
        if missing:
            raise RuntimeError(
                "Stable Diffusion VAE weights are incomplete; missing keys: {}".format(
                    missing
                )
            )
        if unexpected:
            print("Ignoring unexpected first-stage keys: {}".format(unexpected))
        if isinstance(checkpoint, dict) and "global_step" in checkpoint:
            print("Stable Diffusion global step: {}".format(checkpoint["global_step"]))

        del first_stage_state
        del state_dict
        del checkpoint
        gc.collect()
        return cls(first_stage_model, scale_factor).to(device)

    def train(self, mode: bool = True):
        # The VAE must remain in evaluation mode even when the ADJSCC model trains.
        super().train(False)
        return self

    @torch.no_grad()
    def encode_first_stage(self, images: Tensor):
        return self.first_stage_model.encode(images)

    @torch.no_grad()
    def get_first_stage_encoding(self, encoder_posterior) -> Tensor:
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            latent = encoder_posterior.sample()
        elif isinstance(encoder_posterior, Tensor):
            latent = encoder_posterior
        else:
            raise NotImplementedError(
                "unsupported first-stage posterior type: {}".format(
                    type(encoder_posterior).__name__
                )
            )
        return self.scale_factor * latent

    @torch.no_grad()
    def encode(self, images: Tensor) -> Tensor:
        # Keep this exact two-call form aligned with scripts/img2img.py.
        return self.get_first_stage_encoding(self.encode_first_stage(images)).detach()

    def decode_first_stage(self, latent: Tensor) -> Tensor:
        # ldm.LatentDiffusion.decode_first_stage() is decorated with no_grad.
        # Calling the frozen first-stage decoder directly preserves dL/d(latent).
        return self.first_stage_model.decode(latent / self.scale_factor)


class LatentAttentionEncoder(nn.Module):
    """ADJSCC analysis transform for four-channel SD latents."""

    def __init__(self, transmit_channels: int, feature_channels: int = 256) -> None:
        super().__init__()
        channels = feature_channels
        self.blocks = nn.ModuleList(
            [
                GFRModule(4, channels, 9, 2, activation="prelu"),
                GFRModule(channels, channels, 5, 2, activation="prelu"),
                GFRModule(channels, channels, 5, 1, activation="prelu"),
                GFRModule(channels, channels, 5, 1, activation="prelu"),
            ]
        )
        self.attention = nn.ModuleList([AFModule(channels) for _ in range(4)])
        self.final = GFRModule(channels, transmit_channels, 5, 1)

    def forward(self, latent: Tensor, snr_db: Tensor) -> Tensor:
        output = latent
        conditions = (snr_db,)
        for block, attention in zip(self.blocks, self.attention):
            output = attention(block(output), conditions)
        return self.final(output)


class LatentAttentionDecoder(nn.Module):
    """ADJSCC synthesis transform with an unconstrained four-channel output."""

    def __init__(self, transmit_channels: int, feature_channels: int = 256) -> None:
        super().__init__()
        channels = feature_channels
        self.blocks = nn.ModuleList(
            [
                GFRModule(
                    transmit_channels,
                    channels,
                    5,
                    1,
                    decoder=True,
                    activation="prelu",
                ),
                GFRModule(
                    channels, channels, 5, 1, decoder=True, activation="prelu"
                ),
                GFRModule(
                    channels, channels, 5, 1, decoder=True, activation="prelu"
                ),
                GFRModule(
                    channels, channels, 5, 2, decoder=True, activation="prelu"
                ),
            ]
        )
        self.attention = nn.ModuleList([AFModule(channels) for _ in range(4)])
        # activation=None keeps IGDN but removes the RGB model's final sigmoid.
        self.final = GFRModule(
            channels, 4, 9, 2, decoder=True, activation=None
        )

    def forward(self, received: Tensor, snr_db: Tensor) -> Tensor:
        output = received
        conditions = (snr_db,)
        for block, attention in zip(self.blocks, self.attention):
            output = attention(block(output), conditions)
        return self.final(output)


def _match_spatial_size(inputs: Tensor, height: int, width: int) -> Tensor:
    inputs = inputs[..., :height, :width]
    pad_height = max(height - inputs.shape[-2], 0)
    pad_width = max(width - inputs.shape[-1], 0)
    if pad_height or pad_width:
        inputs = F.pad(inputs, (0, pad_width, 0, pad_height))
    return inputs


class LatentADJSCC(nn.Module):
    """Four-channel latent ADJSCC with unit-power normalization and AWGN."""

    def __init__(
        self, transmit_channels: int = 16, feature_channels: int = 256
    ) -> None:
        super().__init__()
        self.transmit_channels = transmit_channels
        self.feature_channels = feature_channels
        self.encoder = LatentAttentionEncoder(
            transmit_channels, feature_channels=feature_channels
        )
        self.channel = Channel("awgn")
        self.decoder = LatentAttentionDecoder(
            transmit_channels, feature_channels=feature_channels
        )

    def forward(self, latent: Tensor, snr_db: Tensor) -> Tensor:
        if latent.ndim != 4 or latent.shape[1] != 4:
            raise ValueError(
                "latent must have shape [batch, 4, height, width], got {}".format(
                    tuple(latent.shape)
                )
            )
        height, width = latent.shape[-2:]
        channel_input = self.encoder(latent, snr_db)
        received = self.channel(channel_input, snr_db)
        reconstruction = self.decoder(received, snr_db)
        return _match_spatial_size(reconstruction, height, width)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def sample_snr(
    batch_size: int,
    low: float,
    high: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if high < low:
        raise ValueError("SNR upper bound must be >= lower bound")
    if high == low:
        return torch.full((batch_size, 1), low, device=device, dtype=dtype)
    return torch.empty(batch_size, 1, device=device, dtype=dtype).uniform_(low, high)


def _average_metrics(totals: Dict[str, float], samples: int) -> Dict[str, float]:
    if samples == 0:
        raise ValueError("data loader produced no samples")
    return {name: value / samples for name, value in totals.items()}


def _effective_batch_count(loader: DataLoader, max_batches: Optional[int]) -> int:
    total = len(loader)
    return total if max_batches is None else min(total, max_batches)


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{}:{:02d}:{:02d}".format(hours, minutes, seconds)


def _should_log_progress(
    processed_batches: int, total_batches: int, log_every: int
) -> bool:
    return log_every > 0 and (
        processed_batches % log_every == 0 or processed_batches == total_batches
    )


def _progress_timing(
    started_at: float, processed_batches: int, total_batches: int
) -> Tuple[float, float, float]:
    elapsed = time.monotonic() - started_at
    seconds_per_batch = elapsed / max(processed_batches, 1)
    remaining = seconds_per_batch * max(total_batches - processed_batches, 0)
    percentage = 100.0 * processed_batches / max(total_batches, 1)
    return elapsed, remaining, percentage


def _optimizer_step(
    model: nn.Module,
    optimizer: Adam,
    accumulated_batches: int,
    grad_clip: float,
) -> None:
    if accumulated_batches <= 0:
        return
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(float(accumulated_batches))
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def train_one_epoch(
    adjscc: LatentADJSCC,
    vae: FrozenFirstStageVAE,
    loader: DataLoader,
    optimizer: Adam,
    device: torch.device,
    snr_low: float,
    snr_high: float,
    latent_weight: float,
    image_weight: float,
    accumulation_steps: int = 1,
    grad_clip: float = 0.0,
    max_batches: Optional[int] = None,
    epoch: int = 1,
    log_every: int = 0,
    save_every_percent: float = 0.0,
    checkpoint_callback: Optional[Callable[[int, int, float], None]] = None,
) -> Dict[str, float]:
    adjscc.train()
    vae.eval()
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "latent_mse": 0.0, "image_l1": 0.0}
    samples = 0
    accumulated = 0
    interval_totals = {name: 0.0 for name in totals}
    interval_samples = 0
    total_batches = _effective_batch_count(loader, max_batches)
    started_at = time.monotonic()
    next_checkpoint_percent = (
        save_every_percent if save_every_percent > 0 else None
    )

    def maybe_save_progress(processed_batches: int) -> None:
        nonlocal next_checkpoint_percent
        if checkpoint_callback is None or next_checkpoint_percent is None:
            return
        # The normal epoch-end checkpoint covers 100% after validation. Keeping
        # intra-epoch saves below 100% avoids a redundant large write.
        if processed_batches >= total_batches:
            return
        percentage = 100.0 * processed_batches / max(total_batches, 1)
        if percentage + 1e-9 < next_checkpoint_percent:
            return
        checkpoint_callback(processed_batches, total_batches, percentage)
        while next_checkpoint_percent <= percentage + 1e-9:
            next_checkpoint_percent += save_every_percent

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(
            device, non_blocking=device.type == "cuda"
        )
        latent = vae.encode(images)
        snr_db = sample_snr(
            images.shape[0], snr_low, snr_high, device, images.dtype
        )
        reconstructed_latent = adjscc(latent, snr_db)
        latent_mse = F.mse_loss(reconstructed_latent, latent)

        if image_weight > 0:
            reconstructed_images = vae.decode_first_stage(reconstructed_latent)
            image_l1 = F.l1_loss(reconstructed_images, images)
        else:
            image_l1 = torch.zeros((), dtype=latent_mse.dtype, device=device)
        loss = latent_weight * latent_mse + image_weight * image_l1
        loss.backward()
        accumulated += 1

        batch_size = images.shape[0]
        samples += batch_size
        batch_metrics = {
            "loss": loss.item(),
            "latent_mse": latent_mse.item(),
            "image_l1": image_l1.item(),
        }
        for name, value in batch_metrics.items():
            totals[name] += value * batch_size
            interval_totals[name] += value * batch_size
        interval_samples += batch_size

        optimizer_stepped = False
        if accumulated == accumulation_steps:
            _optimizer_step(adjscc, optimizer, accumulated, grad_clip)
            accumulated = 0
            optimizer_stepped = True

        processed_batches = batch_index + 1
        if _should_log_progress(processed_batches, total_batches, log_every):
            interval_metrics = _average_metrics(
                interval_totals, interval_samples
            )
            elapsed, remaining, percentage = _progress_timing(
                started_at, processed_batches, total_batches
            )
            print(
                (
                    "Epoch {} train [{}/{} ({:.2f}%)] "
                    "loss={:.8f}, latent_mse={:.8f}, image_l1={:.8f}, "
                    "elapsed={}, ETA={}"
                ).format(
                    epoch,
                    processed_batches,
                    total_batches,
                    percentage,
                    interval_metrics["loss"],
                    interval_metrics["latent_mse"],
                    interval_metrics["image_l1"],
                    _format_duration(elapsed),
                    _format_duration(remaining),
                ),
                flush=True,
            )
            interval_totals = {name: 0.0 for name in totals}
            interval_samples = 0

        # Optimizer gradients are not part of state_dict. Save only immediately
        # after an optimizer step so every processed gradient is represented.
        if optimizer_stepped:
            maybe_save_progress(processed_batches)

    if accumulated:
        _optimizer_step(adjscc, optimizer, accumulated, grad_clip)
    return _average_metrics(totals, samples)


@torch.no_grad()
def evaluate(
    adjscc: LatentADJSCC,
    vae: FrozenFirstStageVAE,
    loader: DataLoader,
    device: torch.device,
    snr_db: float,
    latent_weight: float,
    image_weight: float,
    max_batches: Optional[int] = None,
    sample_count: int = 4,
    log_every: int = 0,
    progress_label: str = "Validation",
) -> Tuple[Dict[str, float], Optional[Tuple[Tensor, Tensor]]]:
    adjscc.eval()
    vae.eval()
    latent_squared_error = 0.0
    latent_energy = 0.0
    latent_elements = 0
    image_squared_error = 0.0
    image_absolute_error = 0.0
    image_elements = 0
    sample_pair = None
    total_batches = _effective_batch_count(loader, max_batches)
    started_at = time.monotonic()

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(
            device, non_blocking=device.type == "cuda"
        )
        latent = vae.encode(images)
        condition = torch.full(
            (images.shape[0], 1), snr_db, dtype=images.dtype, device=device
        )
        reconstructed_latent = adjscc(latent, condition)
        reconstructed_images = vae.decode_first_stage(reconstructed_latent)

        latent_squared_error += F.mse_loss(
            reconstructed_latent, latent, reduction="sum"
        ).item()
        latent_energy += latent.square().sum().item()
        latent_elements += latent.numel()

        target_01 = ((images + 1.0) / 2.0).clamp(0.0, 1.0)
        reconstruction_01 = ((reconstructed_images + 1.0) / 2.0).clamp(
            0.0, 1.0
        )
        image_squared_error += F.mse_loss(
            reconstruction_01, target_01, reduction="sum"
        ).item()
        image_absolute_error += F.l1_loss(
            reconstructed_images, images, reduction="sum"
        ).item()
        image_elements += images.numel()

        if sample_pair is None and sample_count > 0:
            count = min(sample_count, images.shape[0])
            sample_pair = (
                target_01[:count].cpu(), reconstruction_01[:count].cpu()
            )

        processed_batches = batch_index + 1
        if _should_log_progress(processed_batches, total_batches, log_every):
            elapsed, remaining, percentage = _progress_timing(
                started_at, processed_batches, total_batches
            )
            running_latent_mse = latent_squared_error / max(latent_elements, 1)
            running_image_mse = image_squared_error / max(image_elements, 1)
            running_psnr = (
                float("inf")
                if running_image_mse == 0
                else 10.0 * math.log10(1.0 / running_image_mse)
            )
            print(
                (
                    "{} [{}/{} ({:.2f}%)] latent_mse={:.8f}, "
                    "image_PSNR={:.4f} dB, elapsed={}, ETA={}"
                ).format(
                    progress_label,
                    processed_batches,
                    total_batches,
                    percentage,
                    running_latent_mse,
                    running_psnr,
                    _format_duration(elapsed),
                    _format_duration(remaining),
                ),
                flush=True,
            )

    if latent_elements == 0 or image_elements == 0:
        raise ValueError("evaluation data loader produced no samples")
    latent_mse = latent_squared_error / latent_elements
    image_mse = image_squared_error / image_elements
    image_l1 = image_absolute_error / image_elements
    metrics = {
        "loss": latent_weight * latent_mse + image_weight * image_l1,
        "latent_mse": latent_mse,
        "latent_nmse": latent_squared_error / max(latent_energy, 1e-12),
        "image_l1": image_l1,
        "image_mse": image_mse,
        "image_psnr": float("inf")
        if image_mse == 0
        else 10.0 * math.log10(1.0 / image_mse),
    }
    return metrics, sample_pair


def save_reconstruction_grid(
    path: str, sample_pair: Optional[Tuple[Tensor, Tensor]]
) -> None:
    if sample_pair is None:
        return
    originals, reconstructions = sample_pair
    grid = make_grid(
        torch.cat((originals, reconstructions), dim=0), nrow=originals.shape[0]
    )
    save_image(path, grid)


def build_datasets(args):
    all_paths = discover_images(args.data_dir)
    train_paths, val_paths = split_image_paths(
        all_paths, args.val_count, args.split_seed
    )
    if args.limit_train_samples is not None:
        train_paths = train_paths[: args.limit_train_samples]
    if args.limit_val_samples is not None:
        val_paths = val_paths[: args.limit_val_samples]
    train_dataset = FFHQDataset(
        train_paths, image_size=args.image_size, augment=not args.no_augment
    )
    val_dataset = FFHQDataset(
        val_paths, image_size=args.image_size, augment=False
    )
    return train_dataset, val_dataset, len(all_paths)


def checkpoint_dimensions(args, checkpoint_path: Optional[str]):
    transmit_channels = args.transmit_channel_num
    feature_channels = args.feature_channels
    if not checkpoint_path:
        return transmit_channels, feature_channels
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict):
        metadata = payload.get("metadata", {})
        transmit_channels = int(
            metadata.get("transmit_channel_num", transmit_channels)
        )
        feature_channels = int(metadata.get("feature_channels", feature_channels))
    del payload
    return transmit_channels, feature_channels


def load_history(path: Path) -> Dict[str, List]:
    if not path.is_file():
        return {"epochs": []}
    with path.open("r", encoding="utf-8") as handle:
        history = json.load(handle)
    if "epochs" not in history or not isinstance(history["epochs"], list):
        raise ValueError("invalid history file: {}".format(path))
    return history


def print_rate_summary(image_size: int, transmit_channels: int) -> None:
    latent_side = image_size // 8
    channel_side = latent_side // 4
    complex_symbols = transmit_channels * channel_side * channel_side / 2.0
    latent_scalars = 4 * latent_side * latent_side
    image_pixels = image_size * image_size
    print("VAE latent shape: [B, 4, {}, {}]".format(latent_side, latent_side))
    print(
        "channel representation: [B, {}, {}, {}]".format(
            transmit_channels, channel_side, channel_side
        )
    )
    print("complex channel symbols/image: {:.0f}".format(complex_symbols))
    print(
        "complex symbols per latent scalar: {:.8f}".format(
            complex_symbols / latent_scalars
        )
    )
    print(
        "complex symbols per image pixel: {:.8f}".format(
            complex_symbols / image_pixels
        )
    )


def train_command(args, device: torch.device) -> None:
    train_dataset, val_dataset, discovered = build_datasets(args)
    print(
        "FFHQ images: {} discovered, {} train, {} validation".format(
            discovered, len(train_dataset), len(val_dataset)
        )
    )
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers, device
    )
    val_loader = make_loader(
        val_dataset, args.eval_batch_size, False, args.num_workers, device
    )

    transmit_channels, feature_channels = checkpoint_dimensions(args, args.resume)
    adjscc = LatentADJSCC(transmit_channels, feature_channels).to(device)
    vae = FrozenFirstStageVAE.from_stable_diffusion(
        args.sd_config, args.sd_checkpoint, device
    )
    optimizer = Adam(adjscc.parameters(), lr=args.learning_rate)
    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        payload = load_checkpoint(
            args.resume,
            adjscc,
            device,
            None if args.reset_optimizer else optimizer,
        )
        start_epoch = int(payload.get("epoch", 0))
        if not args.reset_best:
            best_loss = float(payload.get("best_loss", best_loss))
        resume_metadata = payload.get("metadata", {})
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        if resume_metadata.get("checkpoint_kind") == "intra_epoch":
            print(
                (
                    "resumed intra-epoch ADJSCC checkpoint from epoch {} "
                    "at batch {}/{} ({:.2f}%); epoch {} will restart "
                    "from its beginning"
                ).format(
                    resume_metadata.get("epoch_in_progress", start_epoch + 1),
                    resume_metadata.get("batch_in_epoch", "?"),
                    resume_metadata.get("batches_in_epoch", "?"),
                    float(resume_metadata.get("progress_percent", 0.0)),
                    start_epoch + 1,
                )
            )
        else:
            print("resumed ADJSCC from epoch {}".format(start_epoch))
        del payload
        gc.collect()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"
    history = load_history(history_path) if args.resume else {"epochs": []}
    metadata = dict(vars(args))
    metadata["transmit_channel_num"] = transmit_channels
    metadata["feature_channels"] = feature_channels
    last_checkpoint_path = output_dir / "last.pt"

    print("device: {}".format(device))
    print("ADJSCC trainable parameters: {:,}".format(parameter_count(adjscc)))
    print("VAE trainable parameters: 0")
    print_rate_summary(args.image_size, transmit_channels)

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        def save_training_progress(
            processed_batches: int,
            total_batches: int,
            percentage: float,
        ) -> None:
            progress_metadata = dict(metadata)
            progress_metadata.update(
                {
                    "checkpoint_kind": "intra_epoch",
                    "epoch_in_progress": epoch,
                    "batch_in_epoch": processed_batches,
                    "batches_in_epoch": total_batches,
                    "progress_percent": percentage,
                }
            )
            # `epoch` stores the number of fully completed epochs. Resuming this
            # snapshot therefore repeats the current epoch instead of skipping it.
            save_checkpoint(
                str(last_checkpoint_path),
                adjscc,
                optimizer,
                epoch=epoch - 1,
                best_loss=best_loss,
                metadata=progress_metadata,
            )
            print(
                (
                    "Saved intra-epoch checkpoint: {} "
                    "(epoch {}, batch {}/{}, {:.2f}%)"
                ).format(
                    last_checkpoint_path,
                    epoch,
                    processed_batches,
                    total_batches,
                    percentage,
                ),
                flush=True,
            )

        train_metrics = train_one_epoch(
            adjscc,
            vae,
            train_loader,
            optimizer,
            device,
            args.snr_low_train,
            args.snr_up_train,
            args.latent_loss_weight,
            args.image_loss_weight,
            accumulation_steps=args.accumulation_steps,
            grad_clip=args.grad_clip,
            max_batches=args.max_train_batches,
            epoch=epoch,
            log_every=args.log_every,
            save_every_percent=args.save_every_percent,
            checkpoint_callback=save_training_progress,
        )
        record = {"epoch": epoch, "train": train_metrics}
        message = (
            "Epoch {}: train loss={:.8f}, latent_mse={:.8f}, image_l1={:.8f}"
        ).format(
            epoch,
            train_metrics["loss"],
            train_metrics["latent_mse"],
            train_metrics["image_l1"],
        )

        if epoch % args.val_every == 0:
            val_metrics, samples = evaluate(
                adjscc,
                vae,
                val_loader,
                device,
                args.snr_val,
                args.latent_loss_weight,
                args.image_loss_weight,
                max_batches=args.max_eval_batches,
                sample_count=args.sample_count,
                log_every=args.log_every,
                progress_label="Epoch {} validation".format(epoch),
            )
            record["validation"] = val_metrics
            message += (
                ", val loss={:.8f}, latent_mse={:.8f}, PSNR={:.4f} dB"
            ).format(
                val_metrics["loss"],
                val_metrics["latent_mse"],
                val_metrics["image_psnr"],
            )
            save_reconstruction_grid(
                str(output_dir / "samples" / "epoch_{:04d}.png".format(epoch)),
                samples,
            )
            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                best_metadata = dict(metadata)
                best_metadata["checkpoint_kind"] = "best"
                save_checkpoint(
                    str(output_dir / "best.pt"),
                    adjscc,
                    optimizer,
                    epoch=epoch,
                    best_loss=best_loss,
                    metadata=best_metadata,
                )
                message += " (best saved)"

        completed_metadata = dict(metadata)
        completed_metadata["checkpoint_kind"] = "epoch_complete"
        save_checkpoint(
            str(last_checkpoint_path),
            adjscc,
            optimizer,
            epoch=epoch,
            best_loss=best_loss,
            metadata=completed_metadata,
        )
        history["epochs"].append(record)
        save_json(str(history_path), history)
        print(message)


def evaluate_command(args, device: torch.device) -> None:
    checkpoint_path = args.checkpoint or str(Path(args.output_dir) / "best.pt")
    transmit_channels, feature_channels = checkpoint_dimensions(
        args, checkpoint_path
    )
    adjscc = LatentADJSCC(transmit_channels, feature_channels).to(device)
    load_checkpoint(checkpoint_path, adjscc, device)
    vae = FrozenFirstStageVAE.from_stable_diffusion(
        args.sd_config, args.sd_checkpoint, device
    )
    _, val_dataset, discovered = build_datasets(args)
    loader = make_loader(
        val_dataset, args.eval_batch_size, False, args.num_workers, device
    )
    print(
        "FFHQ images: {} discovered, {} used for evaluation".format(
            discovered, len(val_dataset)
        )
    )

    output_dir = Path(args.output_dir) / "evaluation"
    results = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "sd_checkpoint": str(Path(args.sd_checkpoint).resolve()),
        "metrics": [],
    }
    for snr_db in args.eval_snrs:
        repeated = []
        sample_pair = None
        for repeat in range(args.eval_repeats):
            metrics, samples = evaluate(
                adjscc,
                vae,
                loader,
                device,
                snr_db,
                args.latent_loss_weight,
                args.image_loss_weight,
                max_batches=args.max_eval_batches,
                sample_count=args.sample_count,
                log_every=args.log_every,
                progress_label="SNR {:g} dB repeat {}/{}".format(
                    snr_db, repeat + 1, args.eval_repeats
                ),
            )
            repeated.append(metrics)
            if sample_pair is None:
                sample_pair = samples
        averaged = {
            key: float(np.mean([metrics[key] for metrics in repeated]))
            for key in repeated[0]
        }
        result = {"snr_db": float(snr_db), **averaged}
        results["metrics"].append(result)
        save_reconstruction_grid(
            str(output_dir / "snr_{:g}dB.png".format(snr_db)), sample_pair
        )
        print(
            "SNR={:g} dB: latent_mse={:.8f}, NMSE={:.8f}, "
            "image_PSNR={:.4f} dB".format(
                snr_db,
                averaged["latent_mse"],
                averaged["latent_nmse"],
                averaged["image_psnr"],
            )
        )
    save_json(str(output_dir / "metrics.json"), results)


def validate_args(args) -> None:
    if args.image_size <= 0 or args.image_size % 32:
        raise ValueError("--image-size must be a positive multiple of 32")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.transmit_channel_num <= 0 or args.feature_channels <= 0:
        raise ValueError("model channel counts must be positive")
    if args.accumulation_steps <= 0:
        raise ValueError("--accumulation-steps must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.val_every <= 0:
        raise ValueError("--val-every must be positive")
    if args.log_every < 0:
        raise ValueError("--log-every cannot be negative")
    if (
        not math.isfinite(args.save_every_percent)
        or args.save_every_percent < 0
        or args.save_every_percent > 100
    ):
        raise ValueError("--save-every-percent must be between 0 and 100")
    if args.latent_loss_weight < 0 or args.image_loss_weight < 0:
        raise ValueError("loss weights cannot be negative")
    if args.latent_loss_weight == 0 and args.image_loss_weight == 0:
        raise ValueError("at least one loss weight must be non-zero")
    if args.snr_up_train < args.snr_low_train:
        raise ValueError("--snr-up-train must be >= --snr-low-train")
    if args.eval_repeats <= 0:
        raise ValueError("--eval-repeats must be positive")
    if args.sample_count < 0:
        raise ValueError("--sample-count cannot be negative")
    for name in (
        "limit_train_samples",
        "limit_val_samples",
        "max_train_batches",
        "max_eval_batches",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError("--{} must be positive".format(name.replace("_", "-")))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "eval"))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--sd-config", default=str(DEFAULT_SD_CONFIG))
    parser.add_argument("--sd-checkpoint", default=str(DEFAULT_SD_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--image-size", default=512, type=int)
    parser.add_argument("--val-count", default=1000, type=int)
    parser.add_argument("--split-seed", default=0, type=int)
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--limit-val-samples", type=int)
    parser.add_argument("--no-augment", action="store_true")

    parser.add_argument("--transmit-channel-num", default=16, type=int)
    parser.add_argument("--feature-channels", default=256, type=int)
    parser.add_argument("--snr-low-train", default=0.0, type=float)
    parser.add_argument("--snr-up-train", default=20.0, type=float)
    parser.add_argument("--snr-val", default=10.0, type=float)
    parser.add_argument(
        "--eval-snrs", nargs="+", default=[0, 5, 10, 15, 20], type=float
    )
    parser.add_argument("--eval-repeats", default=1, type=int)

    parser.add_argument("--latent-loss-weight", default=1.0, type=float)
    parser.add_argument("--image-loss-weight", default=0.0, type=float)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--eval-batch-size", default=1, type=int)
    parser.add_argument("--accumulation-steps", default=1, type=int)
    parser.add_argument("--grad-clip", default=0.0, type=float)
    parser.add_argument("--val-every", default=1, type=int)
    parser.add_argument(
        "--log-every",
        default=100,
        type=int,
        help="print train/evaluation progress every N batches; 0 disables it",
    )
    parser.add_argument(
        "--save-every-percent",
        default=0.0,
        type=float,
        help=(
            "overwrite last.pt every N percent of an epoch; "
            "0 keeps epoch-end-only saving"
        ),
    )
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint")
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--reset-best", action="store_true")

    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--sample-count", default=4, type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    return parser


def main(args) -> None:
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if args.command == "train":
        train_command(args, device)
    else:
        evaluate_command(args, device)


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    print("Current execution parameters:")
    for argument, value in sorted(vars(parsed_args).items()):
        print("{}: {}".format(argument, value))
    main(parsed_args)
