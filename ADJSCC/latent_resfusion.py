"""Shared five-step latent Resfusion model, schedule, and checkpoint helpers.

The implementation follows the restoration path in the official Resfusion
repository.  It deliberately predicts only residual noise.  Stable Diffusion
handover is a direct tensor handoff; this module adds no bridge or extra loss.
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


REPO_ROOT = Path(__file__).resolve().parents[1]
RESFUSION_DENOISING_ROOT = (
    REPO_ROOT / "Resfusion" / "model" / "denoising_module"
)
if str(RESFUSION_DENOISING_ROOT) not in sys.path:
    sys.path.insert(0, str(RESFUSION_DENOISING_ROOT))

# Import the official RDDM U-Net without importing Resfusion's RGB Lightning
# wrapper.  The wrapper contains RGB range conversion and image-space metrics
# that must not be applied to scaled Stable Diffusion latents.
from RDDM.RDDM_model import RDDM_Unet


LATENT_CHANNELS = 4
RESFUSION_TOTAL_STEPS = 12
RESFUSION_REVERSE_STEPS = 5
TRUNCATION_TARGET = 0.5
TRUNCATION_THRESHOLD = 0.01


def parse_dim_mults(value) -> Tuple[int, ...]:
    """Parse comma-separated U-Net multipliers."""

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        multipliers = tuple(int(part) for part in parts)
    else:
        multipliers = tuple(int(part) for part in value)
    if not multipliers or any(multiplier <= 0 for multiplier in multipliers):
        raise ValueError("dim multipliers must be positive integers")
    return multipliers


class LatentResfusionUNet(nn.Module):
    """Official RDDM U-Net adapted to 4-channel scaled VAE latents."""

    def __init__(
        self,
        dim: int = 64,
        dim_mults: Sequence[int] = (1, 2, 4, 8),
        resnet_block_groups: int = 8,
    ) -> None:
        super().__init__()
        dim_mults = parse_dim_mults(dim_mults)
        if dim <= 0:
            raise ValueError("model dim must be positive")
        if resnet_block_groups <= 0:
            raise ValueError("resnet block groups must be positive")
        if any(
            (dim * multiplier) % resnet_block_groups
            for multiplier in dim_mults
        ):
            raise ValueError(
                "every U-Net width must be divisible by resnet_block_groups"
            )

        self.dim = int(dim)
        self.dim_mults = dim_mults
        self.resnet_block_groups = int(resnet_block_groups)
        self.denoiser = RDDM_Unet(
            dim=self.dim,
            dim_mults=self.dim_mults,
            out_dim=LATENT_CHANNELS,
            channels=LATENT_CHANNELS,
            input_condition=True,
            input_condition_channels=LATENT_CHANNELS,
            resnet_block_groups=self.resnet_block_groups,
        )

    def forward(
        self, state: Tensor, degraded_latent: Tensor, timesteps: Tensor
    ) -> Tensor:
        if state.ndim != 4 or state.shape[1] != LATENT_CHANNELS:
            raise ValueError(
                "state must have shape [B, 4, H, W], got {}".format(
                    tuple(state.shape)
                )
            )
        if degraded_latent.shape != state.shape:
            raise ValueError(
                "degraded latent shape {} does not match state shape {}".format(
                    tuple(degraded_latent.shape), tuple(state.shape)
                )
            )
        if timesteps.ndim != 1 or timesteps.shape[0] != state.shape[0]:
            raise ValueError(
                "timesteps must have shape [B], got {}".format(
                    tuple(timesteps.shape)
                )
            )
        spatial_multiple = 2 ** (len(self.dim_mults) - 1)
        if (
            state.shape[-2] % spatial_multiple
            or state.shape[-1] % spatial_multiple
        ):
            raise ValueError(
                "latent spatial dimensions must be divisible by {}".format(
                    spatial_multiple
                )
            )
        return self.denoiser(
            x=state,
            time=timesteps,
            input_cond=degraded_latent,
        )


class FiveStepResfusionSchedule(nn.Module):
    """The official T=12 LinearPro schedule truncated to five reverse calls."""

    def __init__(
        self,
        total_steps: int = RESFUSION_TOTAL_STEPS,
        target: float = TRUNCATION_TARGET,
        threshold: float = TRUNCATION_THRESHOLD,
    ) -> None:
        super().__init__()
        if total_steps != RESFUSION_TOTAL_STEPS:
            raise ValueError(
                "easy.tex fixes the Resfusion schedule to T={}, got {}".format(
                    RESFUSION_TOTAL_STEPS, total_steps
                )
            )

        scale = 1000.0 / float(total_steps)
        betas = torch.linspace(
            scale * 0.0001,
            scale * 0.02,
            total_steps,
            dtype=torch.float64,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        valid = alphas_cumprod >= 0
        sqrt_valid = torch.zeros_like(alphas_cumprod)
        sqrt_valid[valid] = torch.sqrt(alphas_cumprod[valid])
        candidates = torch.nonzero(
            valid & (sqrt_valid <= float(target)), as_tuple=False
        ).flatten()
        if candidates.numel() == 0:
            raise RuntimeError("LinearPro schedule never reaches the target")
        crossing = int(candidates[0].item())

        # Values beyond the acceleration point are unused.  The official
        # scheduler forces the crossing value to exactly 0.5 when the gap is
        # larger than 0.01, which is the T=12 restoration configuration.
        alphas = alphas[: crossing + 1].clone()
        alphas_cumprod = alphas_cumprod[: crossing + 1].clone()
        gap = float(target) - math.sqrt(float(alphas_cumprod[-1]))
        if gap > float(threshold):
            alphas_cumprod[-1] = float(target) ** 2
            alphas[-1] = alphas_cumprod[-1] / alphas_cumprod[-2]
        betas = 1.0 - alphas

        if alphas_cumprod.numel() != RESFUSION_REVERSE_STEPS:
            raise RuntimeError(
                "T=12 must produce five Resfusion steps, got {}".format(
                    alphas_cumprod.numel()
                )
            )

        alphas_cumprod_prev = torch.roll(alphas_cumprod, shifts=1, dims=0)
        # Reproduce the official implementation.  Its sampler suppresses the
        # final stochastic term explicitly by setting the sampled noise to
        # zero at t=0 (see reverse_step below).
        alphas_cumprod_prev[0] = alphas_cumprod_prev[1]
        posterior_betas = (
            (1.0 - alphas_cumprod_prev)
            / (1.0 - alphas_cumprod)
            * betas
        )

        self.total_steps = int(total_steps)
        self.register_buffer("betas", betas.to(torch.float32))
        self.register_buffer("alphas", alphas.to(torch.float32))
        self.register_buffer(
            "alphas_cumprod", alphas_cumprod.to(torch.float32)
        )
        self.register_buffer(
            "alphas_cumprod_prev", alphas_cumprod_prev.to(torch.float32)
        )
        self.register_buffer(
            "posterior_betas", posterior_betas.to(torch.float32)
        )

    @property
    def num_steps(self) -> int:
        return int(self.alphas_cumprod.numel())

    @property
    def acceleration_index(self) -> int:
        return self.num_steps - 1

    @property
    def stage_alphas_cumprod(self) -> Tensor:
        """Return alpha-bar for completed denoise counts k=0,...,5."""

        clean = torch.ones(
            1,
            device=self.alphas_cumprod.device,
            dtype=self.alphas_cumprod.dtype,
        )
        return torch.cat((torch.flip(self.alphas_cumprod, dims=(0,)), clean))

    @staticmethod
    def _validate_latent_pair(clean_latent: Tensor, degraded_latent: Tensor) -> None:
        if clean_latent.ndim != 4 or clean_latent.shape[1] != LATENT_CHANNELS:
            raise ValueError(
                "clean latent must have shape [B, 4, H, W], got {}".format(
                    tuple(clean_latent.shape)
                )
            )
        if degraded_latent.shape != clean_latent.shape:
            raise ValueError(
                "degraded latent shape {} does not match clean shape {}".format(
                    tuple(degraded_latent.shape), tuple(clean_latent.shape)
                )
            )

    def _extract(self, values: Tensor, timesteps: Tensor, reference: Tensor) -> Tensor:
        timesteps = timesteps.to(device=values.device, dtype=torch.long).reshape(-1)
        if timesteps.shape[0] != reference.shape[0]:
            raise ValueError("one timestep is required for every batch element")
        if timesteps.numel() and (
            int(timesteps.min().item()) < 0
            or int(timesteps.max().item()) >= self.num_steps
        ):
            raise ValueError(
                "Resfusion timestep must be in [0, {}]".format(
                    self.num_steps - 1
                )
            )
        shape = (reference.shape[0],) + (1,) * (reference.ndim - 1)
        return values.gather(0, timesteps).reshape(shape).to(
            device=reference.device, dtype=reference.dtype
        )

    def forward_state(
        self,
        clean_latent: Tensor,
        degraded_latent: Tensor,
        timesteps: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """Sample v_t using the Resfusion closed-form forward process."""

        self._validate_latent_pair(clean_latent, degraded_latent)
        if noise.shape != clean_latent.shape:
            raise ValueError("noise shape must match the latent shape")
        alpha_bar = self._extract(
            self.alphas_cumprod, timesteps, clean_latent
        )
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        residual = degraded_latent - clean_latent
        return (
            sqrt_alpha_bar * clean_latent
            + (1.0 - sqrt_alpha_bar) * residual
            + torch.sqrt(1.0 - alpha_bar) * noise
        )

    def residual_noise_target(
        self,
        clean_latent: Tensor,
        degraded_latent: Tensor,
        timesteps: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """Build the paper's residual-noise regression target."""

        self._validate_latent_pair(clean_latent, degraded_latent)
        if noise.shape != clean_latent.shape:
            raise ValueError("noise shape must match the latent shape")
        alpha = self._extract(self.alphas, timesteps, clean_latent)
        beta = self._extract(self.betas, timesteps, clean_latent)
        alpha_bar = self._extract(
            self.alphas_cumprod, timesteps, clean_latent
        )
        residual = degraded_latent - clean_latent
        return noise + (
            (1.0 - torch.sqrt(alpha))
            * torch.sqrt(1.0 - alpha_bar)
            / beta
            * residual
        )

    def initial_state(
        self, degraded_latent: Tensor, noise: Optional[Tensor] = None
    ) -> Tensor:
        """Create v_4 = 0.5 * z_hat + sqrt(0.75) * epsilon."""

        if degraded_latent.ndim != 4 or degraded_latent.shape[1] != LATENT_CHANNELS:
            raise ValueError(
                "degraded latent must have shape [B, 4, H, W]"
            )
        if noise is None:
            noise = torch.randn_like(degraded_latent)
        if noise.shape != degraded_latent.shape:
            raise ValueError("noise shape must match the degraded latent")
        alpha_bar = self.alphas_cumprod[-1].to(
            device=degraded_latent.device, dtype=degraded_latent.dtype
        )
        return (
            torch.sqrt(alpha_bar) * degraded_latent
            + torch.sqrt(1.0 - alpha_bar) * noise
        )

    def reverse_step(
        self,
        state: Tensor,
        predicted_residual_noise: Tensor,
        timestep: int,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply one stochastic Resfusion reverse update."""

        if timestep < 0 or timestep >= self.num_steps:
            raise ValueError(
                "Resfusion timestep must be in [0, {}]".format(
                    self.num_steps - 1
                )
            )
        if predicted_residual_noise.shape != state.shape:
            raise ValueError("predicted residual-noise shape must match state")
        if noise is None:
            noise = torch.randn_like(state) if timestep > 0 else torch.zeros_like(state)
        if noise.shape != state.shape:
            raise ValueError("reverse noise shape must match state")

        alpha = self.alphas[timestep].to(
            device=state.device, dtype=state.dtype
        )
        beta = self.betas[timestep].to(
            device=state.device, dtype=state.dtype
        )
        alpha_bar = self.alphas_cumprod[timestep].to(
            device=state.device, dtype=state.dtype
        )
        posterior_beta = self.posterior_betas[timestep].to(
            device=state.device, dtype=state.dtype
        )
        mean = (
            state
            - beta
            / torch.sqrt(1.0 - alpha_bar)
            * predicted_residual_noise
        ) / torch.sqrt(alpha)
        return mean + torch.sqrt(posterior_beta) * noise


@torch.no_grad()
def generate_resfusion_states(
    model: LatentResfusionUNet,
    schedule: FiveStepResfusionSchedule,
    degraded_latent: Tensor,
    initial_noise: Optional[Tensor] = None,
    max_completed_steps: int = RESFUSION_REVERSE_STEPS,
) -> List[Tensor]:
    """Generate direct-handover states from k=0 through the requested step."""

    if (
        max_completed_steps < 0
        or max_completed_steps > schedule.num_steps
    ):
        raise ValueError(
            "max_completed_steps must be in [0, {}]".format(
                schedule.num_steps
            )
        )

    state = schedule.initial_state(degraded_latent, noise=initial_noise)
    states = [state]
    timesteps = range(
        schedule.num_steps - 1,
        schedule.num_steps - 1 - max_completed_steps,
        -1,
    )
    for timestep in timesteps:
        timestep_batch = torch.full(
            (state.shape[0],),
            timestep,
            device=state.device,
            dtype=torch.long,
        )
        prediction = model(state, degraded_latent, timestep_batch)
        state = schedule.reverse_step(state, prediction, timestep)
        states.append(state)
    return states


def match_stable_diffusion_timesteps(
    schedule: FiveStepResfusionSchedule,
    sd_alphas_cumprod: Tensor,
) -> List[int]:
    """Map k=0,...,5 to the nearest Stable Diffusion raw timestep."""

    if sd_alphas_cumprod.ndim != 1 or sd_alphas_cumprod.numel() == 0:
        raise ValueError("Stable Diffusion alphas_cumprod must be a non-empty vector")
    sd_alpha = sd_alphas_cumprod.detach().to(
        device="cpu", dtype=torch.float64
    )
    if bool(torch.any(sd_alpha <= 0)) or bool(torch.any(sd_alpha > 1)):
        raise ValueError("Stable Diffusion alpha-bar values must be in (0, 1]")

    res_alpha = schedule.stage_alphas_cumprod.detach().to(
        device="cpu", dtype=torch.float64
    )
    res_coefficients = torch.stack(
        (
            1.0 - torch.sqrt(res_alpha),
            torch.sqrt(1.0 - res_alpha),
        ),
        dim=1,
    )
    sd_coefficients = torch.stack(
        (
            1.0 - torch.sqrt(sd_alpha),
            torch.sqrt(1.0 - sd_alpha),
        ),
        dim=1,
    )
    distances = (
        res_coefficients[:, None, :] - sd_coefficients[None, :, :]
    ).square().sum(dim=2)
    return [int(value) for value in torch.argmin(distances, dim=1).tolist()]


def timestep_mapping_records(
    schedule: FiveStepResfusionSchedule,
    sd_alphas_cumprod: Tensor,
) -> List[Dict[str, float]]:
    """Return JSON-serializable details for the six handover stages."""

    mapping = match_stable_diffusion_timesteps(schedule, sd_alphas_cumprod)
    res_alpha = schedule.stage_alphas_cumprod.detach().cpu().to(torch.float64)
    sd_alpha = sd_alphas_cumprod.detach().cpu().to(torch.float64)
    records = []
    for completed_steps, raw_timestep in enumerate(mapping):
        alpha_value = float(res_alpha[completed_steps])
        sd_value = float(sd_alpha[raw_timestep])
        records.append(
            {
                "completed_resfusion_steps": completed_steps,
                "resfusion_state_index": (
                    schedule.num_steps - 1 - completed_steps
                ),
                "resfusion_alpha_bar": alpha_value,
                "resfusion_sqrt_alpha_bar": math.sqrt(alpha_value),
                "resfusion_noise_coefficient": math.sqrt(1.0 - alpha_value),
                "sd_raw_timestep": raw_timestep,
                "sd_alpha_bar": sd_value,
                "sd_sqrt_alpha_bar": math.sqrt(sd_value),
                "sd_noise_coefficient": math.sqrt(1.0 - sd_value),
            }
        )
    return records


def load_latent_resfusion_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    fallback_dim: int = 64,
    fallback_dim_mults: Sequence[int] = (1, 2, 4, 8),
    fallback_resnet_groups: int = 8,
    freeze: bool = True,
) -> Tuple[LatentResfusionUNet, Dict, Dict[str, object]]:
    """Construct and load a LatentResfusionUNet from a project checkpoint."""

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(
            "Latent Resfusion checkpoint not found: {}".format(checkpoint_file)
        )
    payload = torch.load(str(checkpoint_file), map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        metadata = payload.get("metadata", {})
    else:
        state_dict = payload
        payload = {"model_state_dict": state_dict, "metadata": {}}
        metadata = {}
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint does not contain a model state_dict")
    if not isinstance(metadata, dict):
        metadata = {}

    dim = int(metadata.get("model_dim", fallback_dim))
    dim_mults = parse_dim_mults(
        metadata.get("dim_mults", fallback_dim_mults)
    )
    resnet_groups = int(
        metadata.get("resnet_block_groups", fallback_resnet_groups)
    )
    model = LatentResfusionUNet(
        dim=dim,
        dim_mults=dim_mults,
        resnet_block_groups=resnet_groups,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    if freeze:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    best_loss = float(payload.get("best_loss", float("inf")))
    info = {
        "path": str(checkpoint_file.resolve()),
        "epoch": int(payload.get("epoch", 0)),
        "best_loss": best_loss if math.isfinite(best_loss) else None,
        "model_dim": dim,
        "dim_mults": list(dim_mults),
        "resnet_block_groups": resnet_groups,
        "resfusion_total_steps": int(
            metadata.get("resfusion_total_steps", RESFUSION_TOTAL_STEPS)
        ),
        "resfusion_reverse_steps": int(
            metadata.get(
                "resfusion_reverse_steps", RESFUSION_REVERSE_STEPS
            )
        ),
    }
    return model, payload, info
