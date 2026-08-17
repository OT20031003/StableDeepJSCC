"""Train five-step latent Resfusion with the paper's residual-noise MSE only."""

import argparse
import gc
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW


from adjscc_sd_img2img import load_adjscc
from adjscc_sd_vae_ffhq import (
    DEFAULT_DATA_DIR,
    DEFAULT_SD_CHECKPOINT,
    DEFAULT_SD_CONFIG,
    FrozenFirstStageVAE,
    build_datasets,
    make_loader,
    sample_snr,
)
from latent_resfusion import (
    RESFUSION_REVERSE_STEPS,
    RESFUSION_TOTAL_STEPS,
    FiveStepResfusionSchedule,
    LatentResfusionUNet,
    load_latent_resfusion_checkpoint,
    parse_dim_mults,
)
from training import (
    parameter_count,
    resolve_device,
    save_checkpoint,
    save_json,
    seed_everything,
)


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "outputs" / "latent_resfusion_5step"
)


def _effective_batch_count(loader, max_batches: Optional[int]) -> int:
    if max_batches is None:
        return len(loader)
    return min(len(loader), max_batches)


def _format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return "{}:{:02d}:{:02d}".format(hours, minutes, seconds)
    return "{:02d}:{:02d}".format(minutes, seconds)


def _load_history(path: Path) -> Dict[str, list]:
    if not path.is_file():
        return {"epochs": []}
    with path.open("r", encoding="utf-8") as handle:
        history = json.load(handle)
    if not isinstance(history, dict) or not isinstance(
        history.get("epochs"), list
    ):
        raise ValueError("invalid history file: {}".format(path))
    return history


def _compute_batch(
    model: LatentResfusionUNet,
    schedule: FiveStepResfusionSchedule,
    vae: FrozenFirstStageVAE,
    adjscc,
    images: Tensor,
    device: torch.device,
    snr_low: float,
    snr_high: float,
    autocast_enabled: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return scalar MSE, prediction, and target for one image batch."""

    with torch.no_grad():
        clean_latent = vae.encode(images)
        snr_db = sample_snr(
            clean_latent.shape[0],
            snr_low,
            snr_high,
            device,
            clean_latent.dtype,
        )
        degraded_latent = adjscc(clean_latent, snr_db)
        timesteps = torch.randint(
            0,
            schedule.num_steps,
            (clean_latent.shape[0],),
            device=device,
            dtype=torch.long,
        )
        noise = torch.randn_like(clean_latent)
        state = schedule.forward_state(
            clean_latent, degraded_latent, timesteps, noise
        )
        target = schedule.residual_noise_target(
            clean_latent, degraded_latent, timesteps, noise
        )

    autocast_context = (
        torch.cuda.amp.autocast() if autocast_enabled else nullcontext()
    )
    with autocast_context:
        prediction = model(state, degraded_latent, timesteps)
    # Keep the only optimization objective, residual-noise MSE, in float32.
    loss = F.mse_loss(prediction.float(), target.float())
    return loss, prediction, target


def run_epoch(
    model: LatentResfusionUNet,
    schedule: FiveStepResfusionSchedule,
    vae: FrozenFirstStageVAE,
    adjscc,
    loader,
    device: torch.device,
    snr_low: float,
    snr_high: float,
    precision: str,
    optimizer: Optional[AdamW] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    accumulation_steps: int = 1,
    grad_clip: float = 0.0,
    max_batches: Optional[int] = None,
    log_every: int = 100,
    label: str = "train",
) -> float:
    """Run one train or validation epoch using only residual-noise MSE."""

    training = optimizer is not None
    model.train(training)
    vae.eval()
    adjscc.eval()
    total_batches = _effective_batch_count(loader, max_batches)
    if total_batches <= 0:
        raise ValueError("{} loader produced no batches".format(label))
    autocast_enabled = precision == "autocast" and device.type == "cuda"
    if training:
        optimizer.zero_grad()

    squared_error = 0.0
    elements = 0
    start_time = time.monotonic()
    for batch_index, batch in enumerate(loader):
        if batch_index >= total_batches:
            break
        images = batch["image"].to(
            device, non_blocking=device.type == "cuda"
        )

        grad_context = torch.enable_grad() if training else torch.no_grad()
        with grad_context:
            loss, prediction, target = _compute_batch(
                model,
                schedule,
                vae,
                adjscc,
                images,
                device,
                snr_low,
                snr_high,
                autocast_enabled,
            )
            if training:
                group_start = (
                    batch_index // accumulation_steps
                ) * accumulation_steps
                group_size = min(
                    accumulation_steps, total_batches - group_start
                )
                scaled_loss = loss / float(group_size)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

        with torch.no_grad():
            squared_error += F.mse_loss(
                prediction.float(), target.float(), reduction="sum"
            ).item()
            elements += target.numel()

        processed = batch_index + 1
        optimizer_step = training and (
            processed % accumulation_steps == 0
            or processed == total_batches
        )
        if optimizer_step:
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        if log_every and (
            processed % log_every == 0 or processed == total_batches
        ):
            elapsed = time.monotonic() - start_time
            rate = processed / max(elapsed, 1e-9)
            remaining = (total_batches - processed) / max(rate, 1e-9)
            print(
                "{} {}/{}: resnoise_mse={:.8f}, elapsed={}, ETA={}".format(
                    label,
                    processed,
                    total_batches,
                    squared_error / max(elements, 1),
                    _format_duration(elapsed),
                    _format_duration(remaining),
                ),
                flush=True,
            )

    if elements == 0:
        raise ValueError("{} loader produced no usable batches".format(label))
    return squared_error / elements


def validate_args(args) -> None:
    if args.image_size <= 0 or args.image_size % 32:
        raise ValueError("--image-size must be a positive multiple of 32")
    if args.val_count <= 0:
        raise ValueError("--val-count must be positive")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.accumulation_steps <= 0:
        raise ValueError("--accumulation-steps must be positive")
    if args.learning_rate <= 0 or not math.isfinite(args.learning_rate):
        raise ValueError("--learning-rate must be finite and positive")
    if args.weight_decay < 0 or not math.isfinite(args.weight_decay):
        raise ValueError("--weight-decay must be finite and non-negative")
    if args.grad_clip < 0 or not math.isfinite(args.grad_clip):
        raise ValueError("--grad-clip must be finite and non-negative")
    if args.val_every <= 0:
        raise ValueError("--val-every must be positive")
    if args.log_every < 0:
        raise ValueError("--log-every cannot be negative")
    if args.snr_up_train < args.snr_low_train:
        raise ValueError("--snr-up-train must be >= --snr-low-train")
    if not all(
        math.isfinite(value)
        for value in (
            args.snr_low_train,
            args.snr_up_train,
            args.snr_val,
        )
    ):
        raise ValueError("all SNR values must be finite")
    if args.transmit_channel_num <= 0 or args.feature_channels <= 0:
        raise ValueError("ADJSCC channel counts must be positive")
    args.dim_mults = parse_dim_mults(args.dim_mults)


def train(args) -> None:
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print("device: {}".format(device))

    train_dataset, val_dataset, discovered = build_datasets(args)
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        True,
        args.num_workers,
        device,
    )
    val_loader = make_loader(
        val_dataset,
        args.eval_batch_size,
        False,
        args.num_workers,
        device,
    )
    print(
        "FFHQ images: {} discovered, {} train, {} validation".format(
            discovered, len(train_dataset), len(val_dataset)
        )
    )

    vae = FrozenFirstStageVAE.from_stable_diffusion(
        args.sd_config, args.sd_checkpoint, device
    )
    adjscc, adjscc_info = load_adjscc(
        args.adjscc_checkpoint,
        device,
        args.transmit_channel_num,
        args.feature_channels,
    )
    schedule = FiveStepResfusionSchedule().to(device)

    resume_payload = None
    if args.resume:
        model, resume_payload, resume_info = load_latent_resfusion_checkpoint(
            args.resume,
            device,
            fallback_dim=args.model_dim,
            fallback_dim_mults=args.dim_mults,
            fallback_resnet_groups=args.resnet_block_groups,
            freeze=False,
        )
        print(
            "Resumed Latent Resfusion architecture: dim={}, dim_mults={}, groups={}".format(
                resume_info["model_dim"],
                resume_info["dim_mults"],
                resume_info["resnet_block_groups"],
            )
        )
    else:
        model = LatentResfusionUNet(
            dim=args.model_dim,
            dim_mults=args.dim_mults,
            resnet_block_groups=args.resnet_block_groups,
        ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_epoch = 0
    best_loss = float("inf")
    if resume_payload is not None:
        start_epoch = int(resume_payload.get("epoch", 0))
        best_loss = float(resume_payload.get("best_loss", best_loss))
        if (
            not args.reset_optimizer
            and "optimizer_state_dict" in resume_payload
        ):
            optimizer.load_state_dict(
                resume_payload["optimizer_state_dict"]
            )
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate
                group["weight_decay"] = args.weight_decay
        if args.reset_best:
            best_loss = float("inf")
        print("Resuming after completed epoch {}".format(start_epoch))
        del resume_payload
        gc.collect()

    use_scaler = args.precision == "autocast" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"
    history = _load_history(history_path) if args.resume else {"epochs": []}

    metadata = dict(vars(args))
    metadata.update(
        {
            "model_dim": model.dim,
            "dim_mults": list(model.dim_mults),
            "resnet_block_groups": model.resnet_block_groups,
            "resfusion_total_steps": RESFUSION_TOTAL_STEPS,
            "resfusion_reverse_steps": RESFUSION_REVERSE_STEPS,
            "objective": "residual_noise_mse_only",
            "adjscc_checkpoint_info": adjscc_info,
            "sd_checkpoint_resolved": str(
                Path(args.sd_checkpoint).resolve()
            ),
            "checkpoint_kind": "epoch",
        }
    )

    print("Latent Resfusion parameters: {:,}".format(parameter_count(model)))
    print("VAE trainable parameters: 0")
    print("ADJSCC trainable parameters: 0")
    print(
        "Resfusion schedule: T={} -> {} reverse steps".format(
            schedule.total_steps, schedule.num_steps
        )
    )
    print("Training objective: residual-noise MSE only")

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        train_loss = run_epoch(
            model,
            schedule,
            vae,
            adjscc,
            train_loader,
            device,
            args.snr_low_train,
            args.snr_up_train,
            args.precision,
            optimizer=optimizer,
            scaler=scaler,
            accumulation_steps=args.accumulation_steps,
            grad_clip=args.grad_clip,
            max_batches=args.max_train_batches,
            log_every=args.log_every,
            label="epoch {} train".format(epoch),
        )

        val_loss = None
        if epoch % args.val_every == 0:
            val_loss = run_epoch(
                model,
                schedule,
                vae,
                adjscc,
                val_loader,
                device,
                args.snr_val,
                args.snr_val,
                args.precision,
                optimizer=None,
                scaler=None,
                accumulation_steps=1,
                grad_clip=0.0,
                max_batches=args.max_eval_batches,
                log_every=args.log_every,
                label="epoch {} validation".format(epoch),
            )

        row = {
            "epoch": epoch,
            "train_resnoise_mse": train_loss,
            "validation_resnoise_mse": val_loss,
        }
        history["epochs"].append(row)
        save_json(str(history_path), history)

        save_checkpoint(
            str(output_dir / "last.pt"),
            model,
            optimizer,
            epoch=epoch,
            best_loss=best_loss,
            metadata=metadata,
        )
        message = "epoch {}: train_resnoise_mse={:.8f}".format(
            epoch, train_loss
        )
        if val_loss is not None:
            message += ", validation_resnoise_mse={:.8f}".format(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(
                    str(output_dir / "best.pt"),
                    model,
                    optimizer,
                    epoch=epoch,
                    best_loss=best_loss,
                    metadata=metadata,
                )
                # Keep last.pt's best-loss field consistent with best.pt.
                save_checkpoint(
                    str(output_dir / "last.pt"),
                    model,
                    optimizer,
                    epoch=epoch,
                    best_loss=best_loss,
                    metadata=metadata,
                )
                message += " (saved best)"
        print(message, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--sd-config", default=str(DEFAULT_SD_CONFIG))
    parser.add_argument("--sd-checkpoint", default=str(DEFAULT_SD_CHECKPOINT))
    parser.add_argument("--adjscc-checkpoint", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--val-count", default=1000, type=int)
    parser.add_argument("--split-seed", default=0, type=int)
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--limit-val-samples", type=int)
    parser.add_argument("--no-augment", action="store_true")

    parser.add_argument("--transmit-channel-num", default=32, type=int)
    parser.add_argument("--feature-channels", default=256, type=int)
    parser.add_argument("--snr-low-train", default=-10.0, type=float)
    parser.add_argument("--snr-up-train", default=20.0, type=float)
    parser.add_argument("--snr-val", default=0.0, type=float)

    parser.add_argument("--model-dim", default=64, type=int)
    parser.add_argument("--dim-mults", default="1,2,4,8")
    parser.add_argument("--resnet-block-groups", default=8, type=int)

    parser.add_argument("--learning-rate", default=1.1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--eval-batch-size", default=1, type=int)
    parser.add_argument("--accumulation-steps", default=8, type=int)
    parser.add_argument("--grad-clip", default=1.0, type=float)
    parser.add_argument("--val-every", default=1, type=int)
    parser.add_argument("--log-every", default=100, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--precision", choices=("full", "autocast"), default="autocast")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)

    parser.add_argument("--resume")
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--reset-best", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("Current execution parameters:")
    for name, value in sorted(vars(args).items()):
        print("{}: {}".format(name, value))
    train(args)


if __name__ == "__main__":
    main()
