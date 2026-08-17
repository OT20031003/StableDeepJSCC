"""Compare clean Stable Diffusion latents with ADJSCC reconstructions.

The script streams FFHQ validation images through the frozen Stable Diffusion
VAE and a trained latent ADJSCC checkpoint.  It prints dataset-level moments
for the scaled VAE latent ``z`` and the received latent ``z_hat`` without
materializing every latent in memory.

Optionally, ``--forward-timesteps`` applies the Stable Diffusion DDPM forward
process to both tensors.  The same Gaussian noise is used for ``z_t`` and
``z_hat_t`` at each image/timestep pair so that their difference is caused by
the transmitted starting latent rather than by different noise draws.
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from omegaconf import OmegaConf
from torch import Tensor


ADJSCC_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADJSCC_DIR.parent
for import_root in (ADJSCC_DIR, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from ldm.modules.diffusionmodules.util import make_beta_schedule

from adjscc_sd_vae_ffhq import (
    DEFAULT_DATA_DIR,
    DEFAULT_SD_CHECKPOINT,
    DEFAULT_SD_CONFIG,
    FFHQDataset,
    FrozenFirstStageVAE,
    LatentADJSCC,
    discover_images,
    make_loader,
    split_image_paths,
)
from training import parameter_count, resolve_device, save_json, seed_everything


DEFAULT_OUTPUT_JSON = ADJSCC_DIR / "outputs" / "latent_distribution" / "stats.json"


class StreamingLatentStatistics:
    """Accumulate population statistics for NCHW latent tensors in float64."""

    def __init__(self, channels: int = 4) -> None:
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = channels
        self.sample_count = 0
        self.position_count = 0
        self.channel_sum = torch.zeros(channels, dtype=torch.float64)
        self.channel_cross_product = torch.zeros(
            channels, channels, dtype=torch.float64
        )
        self.minimum = float("inf")
        self.maximum = float("-inf")

    @torch.no_grad()
    def update(self, latent: Tensor) -> None:
        if latent.ndim != 4 or latent.shape[1] != self.channels:
            raise ValueError(
                "latent must have shape [batch, {}, height, width], got {}".format(
                    self.channels, tuple(latent.shape)
                )
            )
        # Each image/spatial location is one four-dimensional observation for
        # channel means and covariance. CPU float64 accumulation avoids keeping
        # the full dataset in memory and limits cancellation error.
        observations = (
            latent.detach()
            .permute(0, 2, 3, 1)
            .reshape(-1, self.channels)
            .to(device="cpu", dtype=torch.float64)
        )
        self.sample_count += int(latent.shape[0])
        self.position_count += int(observations.shape[0])
        self.channel_sum += observations.sum(dim=0)
        self.channel_cross_product += observations.transpose(0, 1).matmul(
            observations
        )
        self.minimum = min(self.minimum, float(observations.min().item()))
        self.maximum = max(self.maximum, float(observations.max().item()))

    def finalize(self) -> Dict[str, object]:
        if self.position_count == 0:
            raise ValueError("no latent observations were accumulated")

        positions = float(self.position_count)
        elements = self.position_count * self.channels
        channel_mean = self.channel_sum / positions
        channel_second_moment = self.channel_cross_product / positions
        channel_covariance = channel_second_moment - torch.outer(
            channel_mean, channel_mean
        )
        # Roundoff can make a theoretically non-negative diagonal very slightly
        # negative. Symmetrize and clamp only that diagonal before taking sqrt.
        channel_covariance = 0.5 * (
            channel_covariance + channel_covariance.transpose(0, 1)
        )
        diagonal_indices = torch.arange(self.channels)
        channel_variance = torch.diagonal(channel_covariance).clamp_min(0.0)
        channel_covariance[diagonal_indices, diagonal_indices] = channel_variance

        global_sum = float(self.channel_sum.sum().item())
        global_sum_square = float(
            torch.diagonal(self.channel_cross_product).sum().item()
        )
        global_mean = global_sum / elements
        global_variance = max(
            global_sum_square / elements - global_mean * global_mean, 0.0
        )

        return {
            "sample_count": self.sample_count,
            "spatial_position_count": self.position_count,
            "element_count": elements,
            "global_mean": global_mean,
            "global_std": math.sqrt(global_variance),
            "channel_mean": channel_mean.tolist(),
            "channel_std": torch.sqrt(channel_variance).tolist(),
            "min": self.minimum,
            "max": self.maximum,
            "rms": math.sqrt(global_sum_square / elements),
            "l2_norm": math.sqrt(max(global_sum_square, 0.0)),
            "channel_covariance": channel_covariance.tolist(),
        }


class ForwardDiffusionSchedule:
    """Minimal q(z_t | z_0) schedule matching the configured SD training."""

    def __init__(self, config_path: str) -> None:
        config = OmegaConf.load(config_path)
        params = config.model.params
        self.schedule = str(params.get("beta_schedule", "linear"))
        self.timesteps = int(params.get("timesteps", 1000))
        self.linear_start = float(params.get("linear_start", 1e-4))
        self.linear_end = float(params.get("linear_end", 2e-2))
        self.cosine_s = float(params.get("cosine_s", 8e-3))
        betas = torch.from_numpy(
            make_beta_schedule(
                self.schedule,
                self.timesteps,
                linear_start=self.linear_start,
                linear_end=self.linear_end,
                cosine_s=self.cosine_s,
            )
        ).to(dtype=torch.float64)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def validate_timesteps(self, timesteps: Sequence[int]) -> None:
        for timestep in timesteps:
            if timestep < 0 or timestep >= self.timesteps:
                raise ValueError(
                    "forward timestep must be between 0 and {}, got {}".format(
                        self.timesteps - 1, timestep
                    )
                )

    def q_sample(self, latent: Tensor, timestep: int, noise: Tensor) -> Tensor:
        if noise.shape != latent.shape:
            raise ValueError("noise and latent must have identical shapes")
        alpha = self.sqrt_alphas_cumprod[timestep].to(
            device=latent.device, dtype=latent.dtype
        )
        sigma = self.sqrt_one_minus_alphas_cumprod[timestep].to(
            device=latent.device, dtype=latent.dtype
        )
        return alpha * latent + sigma * noise

    def metadata(self) -> Dict[str, object]:
        return {
            "beta_schedule": self.schedule,
            "timesteps": self.timesteps,
            "linear_start": self.linear_start,
            "linear_end": self.linear_end,
            "cosine_s": self.cosine_s,
        }


def load_adjscc_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    fallback_transmit_channels: int,
    fallback_feature_channels: int,
) -> Tuple[LatentADJSCC, Dict[str, object]]:
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(
            "ADJSCC checkpoint not found: {}".format(checkpoint_file)
        )

    print("Loading ADJSCC from {}".format(checkpoint_file), flush=True)
    payload = torch.load(str(checkpoint_file), map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        metadata = payload.get("metadata", {})
        epoch = int(payload.get("epoch", 0))
        best_loss = float(payload.get("best_loss", float("inf")))
    else:
        state_dict = payload
        metadata = {}
        epoch = 0
        best_loss = float("inf")
    if not isinstance(state_dict, dict):
        raise ValueError("ADJSCC checkpoint does not contain a state_dict")
    if not isinstance(metadata, dict):
        metadata = {}

    transmit_channels = int(
        metadata.get("transmit_channel_num", fallback_transmit_channels)
    )
    feature_channels = int(
        metadata.get("feature_channels", fallback_feature_channels)
    )
    model = LatentADJSCC(transmit_channels, feature_channels)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    checkpoint_kind = str(metadata.get("checkpoint_kind", "legacy_or_unknown"))
    information: Dict[str, object] = {
        "path": str(checkpoint_file.resolve()),
        "epoch": epoch,
        "best_loss": best_loss if math.isfinite(best_loss) else None,
        "checkpoint_kind": checkpoint_kind,
        "transmit_channel_num": transmit_channels,
        "feature_channels": feature_channels,
    }
    for name in (
        "epoch_in_progress",
        "batch_in_epoch",
        "batches_in_epoch",
        "progress_percent",
    ):
        if name in metadata:
            information[name] = metadata[name]
    if checkpoint_kind == "intra_epoch":
        print(
            "Warning: analyzing an intra-epoch checkpoint at {:.2f}% of epoch {}".format(
                float(metadata.get("progress_percent", 0.0)),
                metadata.get("epoch_in_progress", "?"),
            )
        )

    del state_dict
    del payload
    gc.collect()
    return model, information


def _format_number(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return "{:+.8g}".format(value)


def _format_vector(values: Sequence[float]) -> str:
    return "[{}]".format(", ".join(_format_number(float(value)) for value in values))


def render_statistics_table(
    title: str, series: Sequence[Tuple[str, Dict[str, object]]]
) -> str:
    """Render requested scalar/vector statistics as a Markdown-style table."""

    rows = (
        ("global mean", "global_mean", False),
        ("global std (population)", "global_std", False),
        ("channel mean [4]", "channel_mean", True),
        ("channel std [4] (population)", "channel_std", True),
        ("min", "min", False),
        ("max", "max", False),
        ("RMS", "rms", False),
        ("L2 norm (all values)", "l2_norm", False),
    )
    lines = [title, ""]
    lines.append("| metric | {} |".format(" | ".join(name for name, _ in series)))
    lines.append("| --- | {} |".format(" | ".join("---:" for _ in series)))
    for label, key, is_vector in rows:
        formatted = []
        for _, statistics in series:
            value = statistics[key]
            if is_vector:
                formatted.append(_format_vector(value))
            else:
                formatted.append(_format_number(float(value)))
        lines.append("| {} | {} |".format(label, " | ".join(formatted)))
    return "\n".join(lines)


def render_covariance_tables(
    series: Sequence[Tuple[str, Dict[str, object]]]
) -> str:
    """Render each 4x4 channel covariance as an explicit matrix table."""

    sections: List[str] = []
    for name, statistics in series:
        covariance = statistics["channel_covariance"]
        channels = len(covariance)
        lines = ["{}: channel covariance {}x{} (population)".format(name, channels, channels), ""]
        lines.append(
            "| | {} |".format(
                " | ".join("ch{}".format(index) for index in range(channels))
            )
        )
        lines.append("| --- | {} |".format(" | ".join("---:" for _ in range(channels))))
        for row_index, row in enumerate(covariance):
            lines.append(
                "| ch{} | {} |".format(
                    row_index,
                    " | ".join(_format_number(float(value)) for value in row),
                )
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _snr_label(snr_db: float) -> str:
    return "ADJSCC z_hat ({:g} dB)".format(snr_db)


def _forward_snr_label(snr_db: float, timestep: int) -> str:
    return "ADJSCC z_hat_t ({:g} dB, t={})".format(snr_db, timestep)


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{}:{:02d}:{:02d}".format(hours, minutes, seconds)


def validate_args(args) -> None:
    if args.image_size <= 0 or args.image_size % 32:
        raise ValueError("--image-size must be a positive multiple of 32")
    if args.val_count <= 0:
        raise ValueError("--val-count must be positive")
    if args.limit_samples is not None and args.limit_samples <= 0:
        raise ValueError("--limit-samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.log_every < 0:
        raise ValueError("--log-every cannot be negative")
    if args.transmit_channel_num <= 0 or args.feature_channels <= 0:
        raise ValueError("ADJSCC channel counts must be positive")
    if not args.snr_db:
        raise ValueError("--snr-db requires at least one value")
    if any(not math.isfinite(value) for value in args.snr_db):
        raise ValueError("all --snr-db values must be finite")
    if len(set(args.snr_db)) != len(args.snr_db):
        raise ValueError("--snr-db values must be unique")
    if len(set(args.forward_timesteps)) != len(args.forward_timesteps):
        raise ValueError("--forward-timesteps values must be unique")


@torch.no_grad()
def analyze(args) -> Dict[str, object]:
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)

    all_paths = discover_images(args.data_dir)
    _, validation_paths = split_image_paths(
        all_paths, args.val_count, args.split_seed
    )
    if args.limit_samples is not None:
        validation_paths = validation_paths[: args.limit_samples]
    dataset = FFHQDataset(validation_paths, image_size=args.image_size, augment=False)
    loader = make_loader(
        dataset, args.batch_size, False, args.num_workers, device
    )

    adjscc, checkpoint_info = load_adjscc_checkpoint(
        args.adjscc_checkpoint,
        device,
        args.transmit_channel_num,
        args.feature_channels,
    )
    vae = FrozenFirstStageVAE.from_stable_diffusion(
        args.sd_config, args.sd_checkpoint, device
    )
    forward_schedule: Optional[ForwardDiffusionSchedule] = None
    if args.forward_timesteps:
        forward_schedule = ForwardDiffusionSchedule(args.sd_config)
        forward_schedule.validate_timesteps(args.forward_timesteps)

    clean_statistics = StreamingLatentStatistics(4)
    received_statistics = {
        snr_db: StreamingLatentStatistics(4) for snr_db in args.snr_db
    }
    clean_forward_statistics = {
        timestep: StreamingLatentStatistics(4)
        for timestep in args.forward_timesteps
    }
    received_forward_statistics = {
        (snr_db, timestep): StreamingLatentStatistics(4)
        for snr_db in args.snr_db
        for timestep in args.forward_timesteps
    }

    print("device: {}".format(device))
    print(
        "FFHQ images: {} discovered, {} validation images analyzed".format(
            len(all_paths), len(dataset)
        )
    )
    print("ADJSCC parameters: {:,}".format(parameter_count(adjscc)))
    print("SNR values: {} dB".format(", ".join("{:g}".format(v) for v in args.snr_db)))
    if args.forward_timesteps:
        print(
            "DDPM training timesteps: {}".format(
                ", ".join(str(value) for value in args.forward_timesteps)
            )
        )

    latent_shape: Optional[List[int]] = None
    started_at = time.monotonic()
    total_batches = len(loader)
    for batch_index, batch in enumerate(loader):
        images = batch["image"].to(
            device, non_blocking=device.type == "cuda"
        )
        clean_latent = vae.encode(images)
        current_shape = list(clean_latent.shape[1:])
        if latent_shape is None:
            latent_shape = current_shape
        elif current_shape != latent_shape:
            raise RuntimeError(
                "latent spatial shape changed from {} to {}".format(
                    latent_shape, current_shape
                )
            )
        clean_statistics.update(clean_latent)

        # One noise realization per sample/timestep is shared by the clean and
        # every SNR-specific received latent.
        diffusion_noises = {
            timestep: torch.randn_like(clean_latent)
            for timestep in args.forward_timesteps
        }
        if forward_schedule is not None:
            for timestep, noise in diffusion_noises.items():
                clean_latent_t = forward_schedule.q_sample(
                    clean_latent, timestep, noise
                )
                clean_forward_statistics[timestep].update(clean_latent_t)

        for snr_db in args.snr_db:
            condition = torch.full(
                (clean_latent.shape[0], 1),
                snr_db,
                device=device,
                dtype=clean_latent.dtype,
            )
            received_latent = adjscc(clean_latent, condition)
            received_statistics[snr_db].update(received_latent)
            if forward_schedule is not None:
                for timestep, noise in diffusion_noises.items():
                    received_latent_t = forward_schedule.q_sample(
                        received_latent, timestep, noise
                    )
                    received_forward_statistics[(snr_db, timestep)].update(
                        received_latent_t
                    )

        processed_batches = batch_index + 1
        if args.log_every > 0 and (
            processed_batches % args.log_every == 0
            or processed_batches == total_batches
        ):
            elapsed = time.monotonic() - started_at
            seconds_per_batch = elapsed / processed_batches
            remaining = seconds_per_batch * (total_batches - processed_batches)
            print(
                "Analysis [{}/{} ({:.2f}%)] elapsed={}, ETA={}".format(
                    processed_batches,
                    total_batches,
                    100.0 * processed_batches / total_batches,
                    _format_duration(elapsed),
                    _format_duration(remaining),
                ),
                flush=True,
            )

    clean_result = clean_statistics.finalize()
    received_results = {
        snr_db: statistics.finalize()
        for snr_db, statistics in received_statistics.items()
    }
    pre_diffusion_series = [("clean z", clean_result)] + [
        (_snr_label(snr_db), received_results[snr_db])
        for snr_db in args.snr_db
    ]
    print()
    print(
        render_statistics_table(
            "Scaled VAE latent distribution before forward diffusion",
            pre_diffusion_series,
        )
    )
    print()
    print(render_covariance_tables(pre_diffusion_series))

    forward_results: List[Dict[str, object]] = []
    for timestep in args.forward_timesteps:
        clean_t_result = clean_forward_statistics[timestep].finalize()
        received_t_results = {
            snr_db: received_forward_statistics[(snr_db, timestep)].finalize()
            for snr_db in args.snr_db
        }
        forward_series = [("clean z_t (t={})".format(timestep), clean_t_result)] + [
            (
                _forward_snr_label(snr_db, timestep),
                received_t_results[snr_db],
            )
            for snr_db in args.snr_db
        ]
        print()
        print(
            render_statistics_table(
                "Forward-diffused latent distribution at DDPM timestep {}".format(
                    timestep
                ),
                forward_series,
            )
        )
        print()
        print(render_covariance_tables(forward_series))
        forward_results.append(
            {
                "timestep": timestep,
                "clean_z_t": clean_t_result,
                "adjscc_z_hat_t": [
                    {
                        "snr_db": float(snr_db),
                        "statistics": received_t_results[snr_db],
                    }
                    for snr_db in args.snr_db
                ],
            }
        )

    result: Dict[str, object] = {
        "analysis": "scaled Stable Diffusion VAE latent distribution before and after ADJSCC",
        "arguments": dict(vars(args)),
        "device": str(device),
        "dataset": {
            "data_dir": str(Path(args.data_dir).resolve()),
            "discovered_images": len(all_paths),
            "split": "validation",
            "split_seed": args.split_seed,
            "configured_validation_count": args.val_count,
            "analyzed_images": len(dataset),
            "image_size": args.image_size,
        },
        "sd_config": str(Path(args.sd_config).resolve()),
        "sd_checkpoint": str(Path(args.sd_checkpoint).resolve()),
        "vae_scale_factor": float(vae.scale_factor.item()),
        "adjscc_checkpoint": checkpoint_info,
        "latent_shape_per_image": latent_shape,
        "statistics_definition": {
            "std": "population standard deviation (ddof=0)",
            "channel_observation": "one [4] vector per image and spatial latent position",
            "channel_covariance": "population covariance (ddof=0)",
            "l2_norm": "L2 norm over all analyzed latent scalar values",
        },
        "clean_z": clean_result,
        "adjscc_z_hat": [
            {
                "snr_db": float(snr_db),
                "statistics": received_results[snr_db],
            }
            for snr_db in args.snr_db
        ],
        "forward_diffusion_schedule": (
            forward_schedule.metadata() if forward_schedule is not None else None
        ),
        "forward_diffusion": forward_results,
    }
    if not args.no_json:
        save_json(args.output_json, result)
        print("\nSaved statistics to {}".format(Path(args.output_json).resolve()))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjscc-checkpoint", required=True)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--sd-config", default=str(DEFAULT_SD_CONFIG))
    parser.add_argument("--sd-checkpoint", default=str(DEFAULT_SD_CHECKPOINT))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--val-count", default=1000, type=int)
    parser.add_argument("--split-seed", default=0, type=int)
    parser.add_argument(
        "--limit-samples",
        type=int,
        help="analyze only the first N validation images (default: all)",
    )
    parser.add_argument(
        "--snr-db",
        nargs="+",
        default=[0.0],
        type=float,
        help="one or more ADJSCC channel SNR values",
    )
    parser.add_argument(
        "--forward-timesteps",
        nargs="*",
        default=[],
        type=int,
        help=(
            "optional DDPM training timesteps for paired z_t/z_hat_t statistics; "
            "for SD v1 use values from 0 through 999"
        ),
    )
    parser.add_argument("--transmit-channel-num", default=16, type=int)
    parser.add_argument("--feature-channels", default=256, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--log-every", default=50, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("Current execution parameters:")
    for name, value in sorted(vars(args).items()):
        print("{}: {}".format(name, value))
    analyze(args)


if __name__ == "__main__":
    main()
