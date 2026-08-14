"""Differentiable wireless channel models implemented with PyTorch."""

from typing import Optional

import torch
from torch import Tensor, nn


def _column(value: Tensor, batch_size: int, name: str) -> Tensor:
    """Return a condition tensor with shape ``[batch_size, 1]``."""
    if not torch.is_tensor(value):
        raise TypeError("{} must be a torch.Tensor".format(name))
    if value.ndim == 0:
        value = value.expand(batch_size)
    if value.ndim == 1:
        value = value.unsqueeze(1)
    if value.ndim != 2 or value.shape != (batch_size, 1):
        raise ValueError(
            "{} must have shape [batch] or [batch, 1], got {}".format(
                name, tuple(value.shape)
            )
        )
    return value


def _complex_noise_like(x: Tensor) -> Tensor:
    """Unit-power circularly symmetric complex Gaussian noise."""
    scale = 2.0 ** -0.5
    return torch.complex(
        torch.randn_like(x.real) * scale,
        torch.randn_like(x.real) * scale,
    )


def _noise_std(snr_db: Tensor) -> Tensor:
    # sqrt(10 ** (-snr_db / 10)) == 10 ** (-snr_db / 20)
    return torch.pow(10.0, -snr_db / 20.0)


def awgn(x: Tensor, snr_db: Tensor) -> Tensor:
    """Add complex additive white Gaussian noise at the requested SNR."""
    std = _noise_std(snr_db).to(dtype=x.real.dtype)
    return x + torch.complex(std, torch.zeros_like(std)) * _complex_noise_like(x)


def slow_fading(
    x: Tensor, snr_db: Tensor, h_real: Tensor, h_imag: Tensor
) -> Tensor:
    """Apply a per-codeword complex slow-fading coefficient and AWGN."""
    std = _noise_std(snr_db).to(dtype=x.real.dtype)
    h = torch.complex(h_real, h_imag)
    return h * x + torch.complex(std, torch.zeros_like(std)) * _complex_noise_like(x)


def slow_fading_eq(
    x: Tensor, snr_db: Tensor, h_real: Tensor, h_imag: Tensor, eps: float = 1e-8
) -> Tensor:
    """Apply AWGN after ideal equalization of a slow-fading channel."""
    std = _noise_std(snr_db).to(dtype=x.real.dtype)
    h = torch.complex(h_real, h_imag)
    # Avoid a numerical singularity for an extremely unlikely exact zero fade.
    magnitude = torch.abs(h)
    safe_h = torch.where(
        magnitude < eps,
        torch.complex(torch.full_like(h.real, eps), torch.zeros_like(h.real)),
        h,
    )
    return x + (
        torch.complex(std, torch.zeros_like(std)) * _complex_noise_like(x) / safe_h
    )


def burst(
    x: Tensor,
    snr_db: Tensor,
    b_prob: Tensor,
    b_stddev: Tensor,
) -> Tensor:
    """Add AWGN and codeword-level Bernoulli-Gaussian burst noise."""
    std = _noise_std(snr_db).to(dtype=x.real.dtype)
    indicator = torch.bernoulli(b_prob.clamp(0.0, 1.0))
    burst_scale = indicator * b_stddev
    return (
        x
        + torch.complex(std, torch.zeros_like(std)) * _complex_noise_like(x)
        + torch.complex(burst_scale, torch.zeros_like(burst_scale))
        * _complex_noise_like(x)
    )


class Channel(nn.Module):
    """Power-normalize a real latent tensor and transmit it over a channel.

    The flattened latent is split into equal real and imaginary halves, matching
    the TensorFlow implementation. Every codeword is normalized to unit average
    complex-symbol power before channel corruption.
    """

    VALID_CHANNELS = ("awgn", "slow_fading", "slow_fading_eq", "burst")

    def __init__(self, channel_type: str = "awgn", eps: float = 1e-8) -> None:
        super().__init__()
        if channel_type not in self.VALID_CHANNELS:
            raise ValueError(
                "channel_type must be one of {}, got {!r}".format(
                    self.VALID_CHANNELS, channel_type
                )
            )
        self.channel_type = channel_type
        self.eps = eps

    def forward(
        self,
        features: Tensor,
        snr_db: Tensor,
        h_real: Optional[Tensor] = None,
        h_imag: Optional[Tensor] = None,
        b_prob: Optional[Tensor] = None,
        b_stddev: Optional[Tensor] = None,
    ) -> Tensor:
        if features.ndim < 2:
            raise ValueError("features must include batch and feature dimensions")

        batch_size = features.shape[0]
        flat = features.reshape(batch_size, -1)
        if flat.shape[1] % 2:
            raise ValueError(
                "the flattened latent size must be even, got {}".format(flat.shape[1])
            )

        dim_z = flat.shape[1] // 2
        z_in = torch.complex(flat[:, :dim_z], flat[:, dim_z:])
        power = torch.sum(torch.abs(z_in).square(), dim=1, keepdim=True)
        z_in = z_in * torch.sqrt(
            torch.as_tensor(dim_z, dtype=features.dtype, device=features.device)
            / power.clamp_min(self.eps)
        )

        snr_db = _column(snr_db, batch_size, "snr_db").to(
            device=features.device, dtype=features.dtype
        )
        if self.channel_type == "awgn":
            z_out = awgn(z_in, snr_db)
        elif self.channel_type in ("slow_fading", "slow_fading_eq"):
            if h_real is None or h_imag is None:
                raise ValueError(
                    "h_real and h_imag are required for {}".format(self.channel_type)
                )
            h_real = _column(h_real, batch_size, "h_real").to(
                device=features.device, dtype=features.dtype
            )
            h_imag = _column(h_imag, batch_size, "h_imag").to(
                device=features.device, dtype=features.dtype
            )
            channel_fn = (
                slow_fading if self.channel_type == "slow_fading" else slow_fading_eq
            )
            z_out = channel_fn(z_in, snr_db, h_real, h_imag)
        else:
            if b_prob is None or b_stddev is None:
                raise ValueError("b_prob and b_stddev are required for burst")
            b_prob = _column(b_prob, batch_size, "b_prob").to(
                device=features.device, dtype=features.dtype
            )
            b_stddev = _column(b_stddev, batch_size, "b_stddev").to(
                device=features.device, dtype=features.dtype
            )
            z_out = burst(z_in, snr_db, b_prob, b_stddev)

        real_features = torch.cat((z_out.real, z_out.imag), dim=1)
        return real_features.reshape_as(features)
