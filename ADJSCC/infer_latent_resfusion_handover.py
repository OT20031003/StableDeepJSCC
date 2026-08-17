"""Directly hand any 0--5 step latent Resfusion state to Stable Diffusion."""

import argparse
import gc
import math
from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor
from torchvision.utils import make_grid


from adjscc_sd_img2img import (
    DEFAULT_SD_CHECKPOINT,
    DEFAULT_SD_CONFIG,
    DeviceAwareDDIMSampler,
    load_adjscc,
    load_init_image,
    load_stable_diffusion,
    precision_scope,
    to_display_range,
)
from latent_resfusion import (
    RESFUSION_REVERSE_STEPS,
    FiveStepResfusionSchedule,
    generate_resfusion_states,
    load_latent_resfusion_checkpoint,
    parse_dim_mults,
    timestep_mapping_records,
)
from training import resolve_device, save_image, save_json, seed_everything


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "latent_resfusion_direct_handover"
)


def raw_timestep_to_t_start(raw_timestep: int, total_timesteps: int = 1000) -> int:
    """Convert an inclusive raw timestep to DDIMSampler.decode's step count."""

    if raw_timestep < 0 or raw_timestep >= total_timesteps:
        raise ValueError(
            "raw timestep must be in [0, {}]".format(total_timesteps - 1)
        )
    return raw_timestep + 1


def prepare_raw_timestep_sampler(
    stable_diffusion, ddim_eta: float
) -> DeviceAwareDDIMSampler:
    """Prepare upstream DDIMSampler for exact raw-timestep reverse sampling."""

    sampler = DeviceAwareDDIMSampler(stable_diffusion)
    # The selected DDIM grid is unused when decode(..., use_original_steps=True),
    # but make_schedule also creates the full-step sigma array.
    sampler.make_schedule(
        ddim_num_steps=50,
        ddim_eta=ddim_eta,
        verbose=False,
    )
    # Upstream p_sample_ddim reads this full-step array from model rather than
    # sampler in its use_original_steps branch.  Expose the already-computed
    # buffer there without changing its values.
    stable_diffusion.ddim_sigmas_for_original_num_steps = (
        sampler.ddim_sigmas_for_original_num_steps
    )
    return sampler


@torch.no_grad()
def decode_from_raw_timestep(
    sampler: DeviceAwareDDIMSampler,
    latent: Tensor,
    conditioning,
    raw_timestep: int,
    guidance_scale: float,
    unconditional_conditioning=None,
) -> Tensor:
    """Run Stable Diffusion directly from raw timestep tau down through zero."""

    total_timesteps = int(sampler.ddpm_num_timesteps)
    return sampler.decode(
        latent,
        conditioning,
        raw_timestep_to_t_start(raw_timestep, total_timesteps),
        unconditional_guidance_scale=guidance_scale,
        unconditional_conditioning=unconditional_conditioning,
        use_original_steps=True,
    )


def validate_args(args) -> None:
    if args.image_size <= 0 or args.image_size % 32:
        raise ValueError("--image-size must be a positive multiple of 32")
    if not math.isfinite(args.snr_db):
        raise ValueError("--snr-db must be finite")
    if args.handover_step < 0 or args.handover_step > RESFUSION_REVERSE_STEPS:
        raise ValueError(
            "--handover-step must be between 0 and {}".format(
                RESFUSION_REVERSE_STEPS
            )
        )
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.guidance_scale < 0 or not math.isfinite(args.guidance_scale):
        raise ValueError("--guidance-scale must be finite and non-negative")
    if args.ddim_eta < 0 or not math.isfinite(args.ddim_eta):
        raise ValueError("--ddim-eta must be finite and non-negative")
    if args.transmit_channel_num <= 0 or args.feature_channels <= 0:
        raise ValueError("ADJSCC channel counts must be positive")
    args.dim_mults = parse_dim_mults(args.dim_mults)


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
    adjscc, adjscc_info = load_adjscc(
        args.adjscc_checkpoint,
        device,
        args.transmit_channel_num,
        args.feature_channels,
    )
    resfusion, resfusion_payload, resfusion_info = load_latent_resfusion_checkpoint(
        args.resfusion_checkpoint,
        device,
        fallback_dim=args.model_dim,
        fallback_dim_mults=args.dim_mults,
        fallback_resnet_groups=args.resnet_block_groups,
        freeze=True,
    )
    del resfusion_payload
    gc.collect()
    schedule = FiveStepResfusionSchedule().to(device)
    mapping = timestep_mapping_records(
        schedule, stable_diffusion.alphas_cumprod
    )
    print(
        "Direct handover mapping k=0..5 -> SD raw timestep: {}".format(
            [record["sd_raw_timestep"] for record in mapping]
        )
    )

    selected_steps = (
        list(range(RESFUSION_REVERSE_STEPS + 1))
        if args.all_handover_steps
        else [args.handover_step]
    )
    max_completed_steps = max(selected_steps)
    init_image = load_init_image(args.init_img, args.image_size).to(
        device, non_blocking=device.type == "cuda"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with stable_diffusion.ema_scope():
        clean_latent = stable_diffusion.get_first_stage_encoding(
            stable_diffusion.encode_first_stage(init_image)
        )
        snr_condition = torch.full(
            (1, 1),
            args.snr_db,
            device=device,
            dtype=clean_latent.dtype,
        )
        degraded_latent = adjscc(clean_latent, snr_condition)
        adjscc_reconstruction = stable_diffusion.decode_first_stage(
            degraded_latent
        )

        with precision_scope(device, args.precision):
            states = generate_resfusion_states(
                resfusion,
                schedule,
                degraded_latent,
                max_completed_steps=max_completed_steps,
            )

        conditioning = stable_diffusion.get_learned_conditioning(
            [args.prompt] * args.num_samples
        )
        unconditional_conditioning = None
        if args.guidance_scale != 1.0:
            unconditional_conditioning = (
                stable_diffusion.get_learned_conditioning(
                    [args.negative_prompt] * args.num_samples
                )
            )
        sampler = prepare_raw_timestep_sampler(
            stable_diffusion, args.ddim_eta
        )

        stage_outputs = []
        for completed_steps in selected_steps:
            raw_timestep = int(
                mapping[completed_steps]["sd_raw_timestep"]
            )
            state = states[completed_steps]
            state_reconstruction = stable_diffusion.decode_first_stage(state)
            direct_latent = state.repeat(
                args.num_samples, 1, 1, 1
            )
            print(
                (
                    "handover k={} directly to SD raw t={} "
                    "({} Stable Diffusion U-Net calls)"
                ).format(
                    completed_steps,
                    raw_timestep,
                    raw_timestep + 1,
                ),
                flush=True,
            )
            with precision_scope(device, args.precision):
                final_latent = decode_from_raw_timestep(
                    sampler,
                    direct_latent,
                    conditioning,
                    raw_timestep,
                    args.guidance_scale,
                    unconditional_conditioning,
                )
                final_images = stable_diffusion.decode_first_stage(
                    final_latent
                )
            stage_outputs.append(
                {
                    "completed_steps": completed_steps,
                    "raw_timestep": raw_timestep,
                    "state_reconstruction": state_reconstruction,
                    "final_images": final_images,
                }
            )

    input_01 = to_display_range(init_image).cpu()
    adjscc_01 = to_display_range(adjscc_reconstruction).cpu()
    save_image(str(output_dir / "input.png"), input_01[0])
    save_image(
        str(output_dir / "adjscc_received.png"), adjscc_01[0]
    )

    output_records: List[Dict[str, object]] = []
    summary_images = [input_01[0], adjscc_01[0]]
    for output in stage_outputs:
        completed_steps = int(output["completed_steps"])
        raw_timestep = int(output["raw_timestep"])
        stage_name = "handover_k{}_sd_t{:04d}".format(
            completed_steps, raw_timestep
        )
        stage_dir = output_dir / stage_name
        state_01 = to_display_range(output["state_reconstruction"]).cpu()
        final_01 = to_display_range(output["final_images"]).cpu()
        save_image(str(stage_dir / "resfusion_state.png"), state_01[0])
        sample_paths = []
        for index, image in enumerate(final_01):
            relative_path = "{}/sample_{:03d}.png".format(
                stage_name, index
            )
            save_image(str(output_dir / relative_path), image)
            sample_paths.append(relative_path)

        comparison = torch.cat(
            (
                input_01.repeat(args.num_samples, 1, 1, 1),
                adjscc_01.repeat(args.num_samples, 1, 1, 1),
                state_01.repeat(args.num_samples, 1, 1, 1),
                final_01,
            ),
            dim=0,
        )
        save_image(
            str(stage_dir / "comparison_grid.png"),
            make_grid(comparison, nrow=args.num_samples),
        )
        summary_images.append(final_01[0])
        output_records.append(
            {
                "completed_resfusion_steps": completed_steps,
                "sd_raw_timestep": raw_timestep,
                "sd_reverse_steps": raw_timestep + 1,
                "resfusion_state_decode": (
                    "{}/resfusion_state.png".format(stage_name)
                ),
                "samples": sample_paths,
                "comparison_grid": (
                    "{}/comparison_grid.png".format(stage_name)
                ),
            }
        )

    save_image(
        str(output_dir / "summary_grid.png"),
        make_grid(torch.stack(summary_images), nrow=len(summary_images)),
    )
    results = {
        "pipeline": (
            "image -> scaled SD VAE latent -> ADJSCC -> five-step latent "
            "Resfusion -> direct Stable Diffusion raw-timestep handover -> "
            "SD VAE decoder"
        ),
        "handover": "direct; no bridge, no re-noise, no residual subtraction",
        "arguments": {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in vars(args).items()
        },
        "device": str(device),
        "sd_checkpoint": str(Path(args.sd_checkpoint).resolve()),
        "adjscc_checkpoint": adjscc_info,
        "resfusion_checkpoint": resfusion_info,
        "input_shape": list(init_image.shape),
        "clean_latent_shape": list(clean_latent.shape),
        "degraded_latent_shape": list(degraded_latent.shape),
        "timestep_mapping": mapping,
        "selected_handover_steps": selected_steps,
        "outputs": {
            "input": "input.png",
            "adjscc_received": "adjscc_received.png",
            "summary_grid": "summary_grid.png",
            "handover_results": output_records,
        },
    }
    save_json(str(output_dir / "metadata.json"), results)
    print("Saved direct-handover results to {}".format(output_dir.resolve()))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-img", required=True)
    parser.add_argument("--adjscc-checkpoint", required=True)
    parser.add_argument("--resfusion-checkpoint", required=True)
    parser.add_argument("--sd-config", default=str(DEFAULT_SD_CONFIG))
    parser.add_argument("--sd-checkpoint", default=str(DEFAULT_SD_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--snr-db", default=0.0, type=float)
    parser.add_argument("--handover-step", default=0, type=int)
    parser.add_argument("--all-handover-steps", action="store_true")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument(
        "--guidance-scale",
        "--scale",
        dest="guidance_scale",
        default=1.0,
        type=float,
    )
    parser.add_argument("--ddim-eta", default=0.0, type=float)
    parser.add_argument("--num-samples", default=1, type=int)

    parser.add_argument("--transmit-channel-num", default=32, type=int)
    parser.add_argument("--feature-channels", default=256, type=int)
    parser.add_argument("--model-dim", default=64, type=int)
    parser.add_argument("--dim-mults", default="1,2,4,8")
    parser.add_argument("--resnet-block-groups", default=8, type=int)

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--precision", choices=("full", "autocast"), default="autocast"
    )
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
