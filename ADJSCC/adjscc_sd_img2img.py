"""Run Stable Diffusion img2img after latent ADJSCC transmission over AWGN."""

import argparse
import gc
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from omegaconf import OmegaConf
from torch import Tensor, nn
from torchvision.utils import make_grid


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

from adjscc_sd_vae_ffhq import FFHQDataset, LatentADJSCC
from training import resolve_device, save_image, save_json, seed_everything


DEFAULT_SD_CONFIG = REPO_ROOT / "configs" / "stable-diffusion" / "v1-inference.yaml"
DEFAULT_SD_CHECKPOINT = (
    REPO_ROOT / "models" / "ldm" / "stable-diffusion-v1" / "model.ckpt"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "sd_adjscc_img2img"


class DeviceAwareDDIMSampler(DDIMSampler):
    """DDIMSampler variant that follows the selected model device."""

    def register_buffer(self, name, attr):
        if isinstance(attr, Tensor):
            attr = attr.to(self.model.device)
        setattr(self, name, attr)


def load_stable_diffusion(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
    verbose: bool = False,
) -> nn.Module:
    config_file = Path(config_path)
    checkpoint_file = Path(checkpoint_path)
    if not config_file.is_file():
        raise FileNotFoundError("Stable Diffusion config not found: {}".format(config_file))
    if not checkpoint_file.is_file():
        raise FileNotFoundError(
            "Stable Diffusion checkpoint not found: {}".format(checkpoint_file)
        )

    print("Loading Stable Diffusion from {}".format(checkpoint_file), flush=True)
    config = OmegaConf.load(str(config_file))
    model = instantiate_from_config(config.model)
    checkpoint = torch.load(str(checkpoint_file), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Stable Diffusion checkpoint does not contain a state_dict")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if isinstance(checkpoint, dict) and "global_step" in checkpoint:
        print("Stable Diffusion global step: {}".format(checkpoint["global_step"]))
    print(
        "Stable Diffusion load result: {} missing, {} unexpected keys".format(
            len(missing), len(unexpected)
        )
    )
    if verbose and missing:
        print("Missing keys: {}".format(missing))
    if verbose and unexpected:
        print("Unexpected keys: {}".format(unexpected))

    del state_dict
    del checkpoint
    gc.collect()
    model = model.to(device)
    # The repository's FrozenCLIPEmbedder stores a separate string device
    # attribute that nn.Module.to() does not update.
    if hasattr(model, "cond_stage_model") and hasattr(
        model.cond_stage_model, "device"
    ):
        model.cond_stage_model.device = str(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_adjscc(
    checkpoint_path: str,
    device: torch.device,
    fallback_transmit_channels: int,
    fallback_feature_channels: int,
) -> Tuple[LatentADJSCC, Dict[str, object]]:
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError("ADJSCC checkpoint not found: {}".format(checkpoint_file))

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
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    checkpoint_kind = str(metadata.get("checkpoint_kind", "legacy_or_unknown"))
    info = {
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
            info[name] = metadata[name]
    if checkpoint_kind == "intra_epoch":
        print(
            "Warning: evaluating an intra-epoch checkpoint at {:.2f}% of epoch {}".format(
                float(metadata.get("progress_percent", 0.0)),
                metadata.get("epoch_in_progress", "?"),
            )
        )

    del state_dict
    del payload
    gc.collect()
    return model, info


def load_init_image(path: str, image_size: int) -> Tensor:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError("input image not found: {}".format(image_path))
    sample = FFHQDataset([image_path], image_size=image_size, augment=False)[0]
    return sample["image"].unsqueeze(0)


def strength_schedule(strength: float, ddim_steps: int) -> Tuple[int, Optional[int]]:
    """Return the denoising step count and valid stochastic-encode index."""

    if not math.isfinite(strength) or strength < 0.0 or strength > 1.0:
        raise ValueError("--strength must be between 0 and 1")
    if ddim_steps <= 0:
        raise ValueError("--ddim-steps must be positive")
    denoise_steps = int(strength * ddim_steps)
    if denoise_steps == 0:
        return 0, None
    # The upstream img2img.py indexes with t_enc directly, which is out of range
    # at strength=1. Using t_enc-1 gives exactly t_enc valid reverse steps.
    return denoise_steps, denoise_steps - 1


def to_display_range(images: Tensor) -> Tensor:
    return ((images.float() + 1.0) / 2.0).clamp(0.0, 1.0)


def precision_scope(device: torch.device, precision: str):
    if precision == "autocast" and device.type == "cuda":
        return torch.autocast("cuda")
    return nullcontext()


def validate_args(args) -> None:
    if args.image_size <= 0 or args.image_size % 32:
        raise ValueError("--image-size must be a positive multiple of 32")
    if not math.isfinite(args.snr_db):
        raise ValueError("--snr-db must be finite")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.guidance_scale < 0 or not math.isfinite(args.guidance_scale):
        raise ValueError("--guidance-scale must be a finite non-negative value")
    if args.ddim_eta < 0 or not math.isfinite(args.ddim_eta):
        raise ValueError("--ddim-eta must be a finite non-negative value")
    if args.transmit_channel_num <= 0 or args.feature_channels <= 0:
        raise ValueError("ADJSCC channel counts must be positive")
    strength_schedule(args.strength, args.ddim_steps)


@torch.no_grad()
def run_inference(args) -> Dict[str, object]:
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print("device: {}".format(device))

    stable_diffusion = load_stable_diffusion(
        args.sd_config,
        args.sd_checkpoint,
        device,
        verbose=args.verbose_loading,
    )
    adjscc, checkpoint_info = load_adjscc(
        args.adjscc_checkpoint,
        device,
        args.transmit_channel_num,
        args.feature_channels,
    )
    init_image = load_init_image(args.init_img, args.image_size).to(
        device, non_blocking=device.type == "cuda"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    denoise_steps, noise_index = strength_schedule(args.strength, args.ddim_steps)

    with stable_diffusion.ema_scope():
        init_latent = stable_diffusion.get_first_stage_encoding(
            stable_diffusion.encode_first_stage(init_image)
        )
        snr_condition = torch.full(
            (1, 1), args.snr_db, device=device, dtype=init_latent.dtype
        )
        received_latent = adjscc(init_latent, snr_condition)
        channel_reconstruction = stable_diffusion.decode_first_stage(received_latent)

        repeated_latent = received_latent.repeat(args.num_samples, 1, 1, 1)
        with precision_scope(device, args.precision):
            if denoise_steps > 0:
                sampler = DeviceAwareDDIMSampler(stable_diffusion)
                sampler.make_schedule(
                    ddim_num_steps=args.ddim_steps,
                    ddim_eta=args.ddim_eta,
                    verbose=False,
                )
                prompt_conditioning = stable_diffusion.get_learned_conditioning(
                    [args.prompt] * args.num_samples
                )
                unconditional_conditioning = None
                if args.guidance_scale != 1.0:
                    unconditional_conditioning = (
                        stable_diffusion.get_learned_conditioning(
                            [args.negative_prompt] * args.num_samples
                        )
                    )
                timestep = torch.full(
                    (args.num_samples,),
                    noise_index,
                    device=device,
                    dtype=torch.long,
                )
                noisy_latent = sampler.stochastic_encode(repeated_latent, timestep)
                final_latent = sampler.decode(
                    noisy_latent,
                    prompt_conditioning,
                    denoise_steps,
                    unconditional_guidance_scale=args.guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                )
            else:
                final_latent = repeated_latent
            final_images = stable_diffusion.decode_first_stage(final_latent)

    input_01 = to_display_range(init_image).cpu()
    channel_01 = to_display_range(channel_reconstruction).cpu()
    final_01 = to_display_range(final_images).cpu()
    save_image(str(output_dir / "input.png"), input_01[0])
    save_image(str(output_dir / "adjscc_received.png"), channel_01[0])
    for index, image in enumerate(final_01):
        save_image(str(output_dir / "sample_{:03d}.png".format(index)), image)

    comparison = torch.cat(
        (
            input_01.repeat(args.num_samples, 1, 1, 1),
            channel_01.repeat(args.num_samples, 1, 1, 1),
            final_01,
        ),
        dim=0,
    )
    grid = make_grid(comparison, nrow=args.num_samples)
    save_image(str(output_dir / "comparison_grid.png"), grid)

    results = {
        "pipeline": (
            "image -> SD VAE encoder -> ADJSCC encoder -> AWGN -> "
            "ADJSCC decoder -> Stable Diffusion DDIM -> SD VAE decoder"
        ),
        "arguments": dict(vars(args)),
        "device": str(device),
        "sd_checkpoint": str(Path(args.sd_checkpoint).resolve()),
        "adjscc_checkpoint": checkpoint_info,
        "input_shape": list(init_image.shape),
        "initial_latent_shape": list(init_latent.shape),
        "received_latent_shape": list(received_latent.shape),
        "denoise_steps": denoise_steps,
        "noise_timestep_index": noise_index,
        "outputs": {
            "input": "input.png",
            "adjscc_received": "adjscc_received.png",
            "samples": [
                "sample_{:03d}.png".format(index)
                for index in range(args.num_samples)
            ],
            "comparison_grid": "comparison_grid.png",
        },
    }
    save_json(str(output_dir / "metadata.json"), results)
    print("Saved inference results to {}".format(output_dir.resolve()))
    print(
        "strength={:.4f}: {} DDIM denoising steps, SNR={:g} dB".format(
            args.strength, denoise_steps, args.snr_db
        )
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-img", required=True)
    parser.add_argument("--prompt", default="a high quality portrait photograph")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--adjscc-checkpoint", required=True)
    parser.add_argument("--sd-config", default=str(DEFAULT_SD_CONFIG))
    parser.add_argument("--sd-checkpoint", default=str(DEFAULT_SD_CHECKPOINT))
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--snr-db", default=0.0, type=float)
    parser.add_argument(
        "--strength",
        default=0.35,
        type=float,
        help="Stable Diffusion img2img noise strength in [0, 1]",
    )
    parser.add_argument("--ddim-steps", default=50, type=int)
    parser.add_argument("--ddim-eta", default=0.0, type=float)
    parser.add_argument(
        "--guidance-scale", "--scale", dest="guidance_scale", default=5.0, type=float
    )
    parser.add_argument("--num-samples", default=4, type=int)
    parser.add_argument("--transmit-channel-num", default=16, type=int)
    parser.add_argument("--feature-channels", default=256, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--precision", choices=("full", "autocast"), default="autocast")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose-loading", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("Current execution parameters:")
    for name, value in sorted(vars(args).items()):
        print("{}: {}".format(name, value))
    run_inference(args)


if __name__ == "__main__":
    main()
