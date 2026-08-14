"""PyTorch modules for baseline and attention-based Deep JSCC."""

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from util_channel import Channel


class _LowerBound(torch.autograd.Function):
    """Lower bound with the gradient rule used by TensorFlow Compression."""

    @staticmethod
    def forward(ctx, inputs: Tensor, bound: Tensor) -> Tensor:
        ctx.save_for_backward(inputs, bound)
        return torch.maximum(inputs, bound)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        inputs, bound = ctx.saved_tensors
        pass_through = (inputs >= bound) | (grad_output < 0)
        return pass_through.to(grad_output.dtype) * grad_output, None


class _NonNegativeParametrizer(nn.Module):
    def __init__(self, minimum: float, reparam_offset: float = 2 ** -18) -> None:
        super().__init__()
        self.pedestal = reparam_offset ** 2
        self.bound = (minimum + self.pedestal) ** 0.5

    def init(self, value: Tensor) -> Tensor:
        return torch.sqrt(torch.clamp(value + self.pedestal, min=self.pedestal))

    def forward(self, value: Tensor) -> Tensor:
        bound = torch.as_tensor(self.bound, dtype=value.dtype, device=value.device)
        value = _LowerBound.apply(value, bound)
        return value.square() - self.pedestal


class GDN(nn.Module):
    """Generalized divisive normalization.

    This is a self-contained PyTorch equivalent of ``tensorflow_compression.GDN``
    and therefore does not require CompressAI or TensorFlow Compression.
    """

    def __init__(
        self,
        channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.inverse = inverse
        self.beta_reparam = _NonNegativeParametrizer(beta_min)
        self.gamma_reparam = _NonNegativeParametrizer(0.0)

        beta = torch.ones(channels)
        gamma = gamma_init * torch.eye(channels)
        self.beta = nn.Parameter(self.beta_reparam.init(beta))
        self.gamma = nn.Parameter(self.gamma_reparam.init(gamma))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError("GDN expects an NCHW tensor, got {}D".format(inputs.ndim))
        channels = inputs.shape[1]
        if channels != self.beta.numel():
            raise ValueError(
                "GDN was built for {} channels, got {}".format(
                    self.beta.numel(), channels
                )
            )
        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma).reshape(channels, channels, 1, 1)
        norm = F.conv2d(inputs.square(), gamma, beta)
        norm = torch.sqrt(norm) if self.inverse else torch.rsqrt(norm)
        return inputs * norm


class GFRModule(nn.Module):
    """Signal convolution followed by GDN/IGDN and an optional activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        decoder: bool = False,
        activation: Optional[str] = None,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("only odd kernels are supported for same-zero padding")
        padding = kernel_size // 2
        if decoder:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=stride - 1,
                bias=True,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            )
        self.gdn = GDN(out_channels, inverse=decoder)

        if activation is None:
            self.activation = nn.Identity()
        elif activation == "prelu":
            self.activation = nn.PReLU(num_parameters=out_channels)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("unsupported activation: {!r}".format(activation))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.gdn(self.conv(inputs)))


class AFModule(nn.Module):
    """Channel-wise attention feature (AF) module."""

    def __init__(self, channels: int, condition_features: int = 1) -> None:
        super().__init__()
        hidden = max(channels // 16, 1)
        self.condition_features = condition_features
        self.dense1 = nn.Linear(channels + condition_features, hidden)
        self.dense2 = nn.Linear(hidden, channels)

    def forward(self, inputs: Tensor, conditions: Sequence[Tensor]) -> Tensor:
        if len(conditions) != self.condition_features:
            raise ValueError(
                "expected {} attention conditions, got {}".format(
                    self.condition_features, len(conditions)
                )
            )
        pooled = F.adaptive_avg_pool2d(inputs, 1).flatten(1)
        columns = []
        for condition in conditions:
            if condition.ndim == 1:
                condition = condition.unsqueeze(1)
            if condition.ndim != 2 or condition.shape[1] != 1:
                raise ValueError(
                    "each attention condition must have shape [batch] or [batch, 1]"
                )
            columns.append(condition.to(dtype=inputs.dtype, device=inputs.device))
        attention = torch.cat([pooled] + columns, dim=1)
        attention = F.relu(self.dense1(attention), inplace=True)
        attention = torch.sigmoid(self.dense2(attention)).unsqueeze(-1).unsqueeze(-1)
        return inputs * attention


class BasicEncoder(nn.Module):
    def __init__(
        self, transmit_channels: int, feature_channels: int = 256
    ) -> None:
        super().__init__()
        c = feature_channels
        self.layers = nn.Sequential(
            GFRModule(3, c, 9, 2, activation="prelu"),
            GFRModule(c, c, 5, 2, activation="prelu"),
            GFRModule(c, c, 5, 1, activation="prelu"),
            GFRModule(c, c, 5, 1, activation="prelu"),
            GFRModule(c, transmit_channels, 5, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class BasicDecoder(nn.Module):
    def __init__(
        self, transmit_channels: int, feature_channels: int = 256
    ) -> None:
        super().__init__()
        c = feature_channels
        self.layers = nn.Sequential(
            GFRModule(transmit_channels, c, 5, 1, decoder=True, activation="prelu"),
            GFRModule(c, c, 5, 1, decoder=True, activation="prelu"),
            GFRModule(c, c, 5, 1, decoder=True, activation="prelu"),
            GFRModule(c, c, 5, 2, decoder=True, activation="prelu"),
            GFRModule(c, 3, 9, 2, decoder=True, activation="sigmoid"),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class AttentionEncoder(nn.Module):
    def __init__(
        self,
        transmit_channels: int,
        feature_channels: int = 256,
        condition_on_fading: bool = False,
    ) -> None:
        super().__init__()
        c = feature_channels
        condition_features = 3 if condition_on_fading else 1
        self.condition_on_fading = condition_on_fading
        self.blocks = nn.ModuleList(
            [
                GFRModule(3, c, 9, 2, activation="prelu"),
                GFRModule(c, c, 5, 2, activation="prelu"),
                GFRModule(c, c, 5, 1, activation="prelu"),
                GFRModule(c, c, 5, 1, activation="prelu"),
            ]
        )
        self.attention = nn.ModuleList(
            [AFModule(c, condition_features) for _ in range(4)]
        )
        self.final = GFRModule(c, transmit_channels, 5, 1)

    def _conditions(
        self,
        snr_db: Tensor,
        h_real: Optional[Tensor],
        h_imag: Optional[Tensor],
    ) -> Sequence[Tensor]:
        if not self.condition_on_fading:
            return (snr_db,)
        if h_real is None or h_imag is None:
            raise ValueError("fading-aware attention requires h_real and h_imag")
        return (snr_db, h_real, h_imag)

    def forward(
        self,
        inputs: Tensor,
        snr_db: Tensor,
        h_real: Optional[Tensor] = None,
        h_imag: Optional[Tensor] = None,
    ) -> Tensor:
        conditions = self._conditions(snr_db, h_real, h_imag)
        outputs = inputs
        for block, attention in zip(self.blocks, self.attention):
            outputs = attention(block(outputs), conditions)
        return self.final(outputs)


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        transmit_channels: int,
        feature_channels: int = 256,
        condition_on_fading: bool = False,
    ) -> None:
        super().__init__()
        c = feature_channels
        condition_features = 3 if condition_on_fading else 1
        self.condition_on_fading = condition_on_fading
        self.blocks = nn.ModuleList(
            [
                GFRModule(
                    transmit_channels, c, 5, 1, decoder=True, activation="prelu"
                ),
                GFRModule(c, c, 5, 1, decoder=True, activation="prelu"),
                GFRModule(c, c, 5, 1, decoder=True, activation="prelu"),
                GFRModule(c, c, 5, 2, decoder=True, activation="prelu"),
            ]
        )
        self.attention = nn.ModuleList(
            [AFModule(c, condition_features) for _ in range(4)]
        )
        self.final = GFRModule(c, 3, 9, 2, decoder=True, activation="sigmoid")

    def _conditions(
        self,
        snr_db: Tensor,
        h_real: Optional[Tensor],
        h_imag: Optional[Tensor],
    ) -> Sequence[Tensor]:
        if not self.condition_on_fading:
            return (snr_db,)
        if h_real is None or h_imag is None:
            raise ValueError("fading-aware attention requires h_real and h_imag")
        return (snr_db, h_real, h_imag)

    def forward(
        self,
        inputs: Tensor,
        snr_db: Tensor,
        h_real: Optional[Tensor] = None,
        h_imag: Optional[Tensor] = None,
    ) -> Tensor:
        conditions = self._conditions(snr_db, h_real, h_imag)
        outputs = inputs
        for block, attention in zip(self.blocks, self.attention):
            outputs = attention(block(outputs), conditions)
        return self.final(outputs)


def _match_spatial_size(inputs: Tensor, height: int, width: int) -> Tensor:
    """Crop/pad decoder output to the source size for non-multiples of four."""
    inputs = inputs[..., :height, :width]
    pad_height = max(height - inputs.shape[-2], 0)
    pad_width = max(width - inputs.shape[-1], 0)
    if pad_height or pad_width:
        inputs = F.pad(inputs, (0, pad_width, 0, pad_height))
    return inputs


class DeepJSCC(nn.Module):
    """End-to-end baseline or adaptive Deep JSCC model.

    Input and output images use PyTorch's conventional NCHW layout and a
    floating-point range of ``[0, 1]``.
    """

    def __init__(
        self,
        transmit_channels: int = 16,
        channel_type: str = "awgn",
        attention: bool = True,
        feature_channels: int = 256,
        condition_on_fading: bool = True,
    ) -> None:
        super().__init__()
        fading_attention = attention and condition_on_fading and channel_type in (
            "slow_fading",
            "slow_fading_eq",
        )
        if attention:
            self.encoder = AttentionEncoder(
                transmit_channels, feature_channels, fading_attention
            )
            self.decoder = AttentionDecoder(
                transmit_channels, feature_channels, fading_attention
            )
        else:
            self.encoder = BasicEncoder(transmit_channels, feature_channels)
            self.decoder = BasicDecoder(transmit_channels, feature_channels)
        self.channel = Channel(channel_type)
        self.attention = attention
        self.condition_on_fading = fading_attention

    def forward(
        self,
        images: Tensor,
        snr_db: Tensor,
        h_real: Optional[Tensor] = None,
        h_imag: Optional[Tensor] = None,
        b_prob: Optional[Tensor] = None,
        b_stddev: Optional[Tensor] = None,
    ) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "images must have shape [batch, 3, height, width], got {}".format(
                    tuple(images.shape)
                )
            )
        height, width = images.shape[-2:]
        if self.attention:
            latent = self.encoder(images, snr_db, h_real, h_imag)
        else:
            latent = self.encoder(images)
        received = self.channel(
            latent,
            snr_db,
            h_real=h_real,
            h_imag=h_imag,
            b_prob=b_prob,
            b_stddev=b_stddev,
        )
        if self.attention:
            reconstruction = self.decoder(received, snr_db, h_real, h_imag)
        else:
            reconstruction = self.decoder(received)
        return _match_spatial_size(reconstruction, height, width)


# Backward-friendly aliases for names used by the original implementation.
Basic_Encoder = BasicEncoder
Basic_Decoder = BasicDecoder
Attention_Encoder = AttentionEncoder
Attention_Decoder = AttentionDecoder
AF_Module = AFModule
