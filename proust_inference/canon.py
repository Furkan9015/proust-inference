"""Canon layers (short depthwise causal convs) for inference."""

from __future__ import annotations
from typing import Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None


class ShortConvolution(nn.Conv1d):
    """Depthwise causal Conv1d with optional fast CUDA kernel."""

    def __init__(
        self, hidden_size: int, kernel_size: int, bias: bool = False,
        activation: Optional[str] = "silu", use_fast_conv1d: Optional[bool] = True,
        device=None, dtype=None,
    ):
        super().__init__(
            in_channels=hidden_size, out_channels=hidden_size,
            kernel_size=kernel_size, groups=hidden_size, bias=bias,
            padding=kernel_size - 1, device=device, dtype=dtype,
        )
        self.hidden_size = hidden_size
        self.activation = None
        if activation is not None:
            assert activation in ["silu", "swish"]
            self.activation = activation

        if causal_conv1d_fn is None and use_fast_conv1d:
            raise RuntimeError("Install `causal-conv1d>=1.4.0` or set use_fast_conv1d=False.")
        self.use_fast_conv1d = use_fast_conv1d and causal_conv1d_fn is not None

    def forward(
        self, x: torch.Tensor, mask=None, cache=None,
        output_final_state: bool = False, seq_idx=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        if cache is not None and x.shape[1] == 1:
            return self.step(x, cache)

        x = x.transpose(1, 2)  # (B, D, T)

        if self.use_fast_conv1d:
            weight = self.weight.squeeze(1)
            x = causal_conv1d_fn(
                x=x, weight=weight, bias=self.bias,
                activation=self.activation, seq_idx=seq_idx,
            )
        else:
            x = self._conv_forward(x, self.weight, self.bias)[..., :x.shape[-1]]
            if self.activation is not None:
                x = F.silu(x)
        return x.transpose(1, 2), cache

    def step(self, x: torch.Tensor, cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[1] == 1
        x = x.squeeze(1)
        if self.use_fast_conv1d:
            x = causal_conv1d_update(
                x=x, conv_state=cache,
                weight=self.weight.squeeze(1), bias=self.bias,
                activation=self.activation,
            )
        else:
            cache.copy_(torch.roll(cache, shifts=-1, dims=-1))
            cache[:, :, -1] = x
            x = torch.sum(cache * self.weight.squeeze(1), dim=-1)
            if self.bias is not None:
                x = x + self.bias
            if self.activation is not None:
                x = F.silu(x)
        return x.unsqueeze(1), cache


def create_canon(dim: int, config) -> ShortConvolution:
    canon = ShortConvolution(
        hidden_size=dim,
        kernel_size=config.canon_kernel,
        bias=config.canon_bias,
        activation="silu" if config.canon_activation else None,
        use_fast_conv1d=causal_conv1d_fn is not None and config.canon_kernel in [2, 3, 4],
    )
    canon._zeyuan_residual = config.canon_residual
    return canon


def apply_canon(
    canon: ShortConvolution, hidden_states: torch.Tensor,
    seq_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if seq_idx is not None and not canon.use_fast_conv1d:
        raise RuntimeError("Packed sequences require `causal-conv1d` for Canon layers.")

    input_was_2d = hidden_states.dim() == 2
    if input_was_2d:
        hidden_states = hidden_states.unsqueeze(0)
    if seq_idx is not None and seq_idx.dim() == 1:
        seq_idx = seq_idx.unsqueeze(0)

    hidden_states2, _ = canon(
        x=hidden_states, cache=None, output_final_state=False, seq_idx=seq_idx,
    )

    if canon._zeyuan_residual:
        out = hidden_states + hidden_states2
    else:
        out = hidden_states2

    if input_was_2d:
        out = out.squeeze(0)
    return out


def cu_seqlens_to_seq_idx(cu_seqlens: torch.Tensor, total_tokens: int | None = None) -> torch.Tensor:
    """Compute per-token sequence IDs from cu_seqlens."""
    if total_tokens is None:
        total_tokens = int(cu_seqlens[-1].item())
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if lengths.numel() == 0:
        return torch.empty(0, device=cu_seqlens.device, dtype=torch.int32)
    seq_ids = torch.arange(lengths.numel(), device=cu_seqlens.device, dtype=torch.int64)
    seq_idx = torch.repeat_interleave(seq_ids, lengths.to(torch.int64))
    return seq_idx.to(torch.int32)
