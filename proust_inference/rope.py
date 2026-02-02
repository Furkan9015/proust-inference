"""Rotary Position Embeddings (RoPE)."""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for batched and varlen modes."""

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._update_cache(max_seq_len)

    def _update_cache(self, seq_len: int):
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        is_varlen = q.dim() == 3
        if is_varlen:
            return self._forward_varlen(q, k, cu_seqlens, position_ids)
        return self._forward_batched(q, k, position_ids)

    def _forward_batched(self, q, k, position_ids=None):
        seq_len = q.shape[1]
        if seq_len > self.max_seq_len:
            self._update_cache(seq_len)

        if position_ids is not None:
            cos = self.cos_cached[position_ids].unsqueeze(2)
            sin = self.sin_cached[position_ids].unsqueeze(2)
        else:
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _forward_varlen(self, q, k, cu_seqlens=None, position_ids=None):
        total_tokens = q.shape[0]

        if position_ids is None:
            if cu_seqlens is not None:
                position_ids = self._culen_indices(cu_seqlens, total_tokens)
            else:
                position_ids = torch.arange(total_tokens, device=q.device, dtype=torch.long)

        cos = self.cos_cached[position_ids].unsqueeze(1)
        sin = self.sin_cached[position_ids].unsqueeze(1)

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _culen_indices(self, cu_seqlens: torch.Tensor, total_tokens: int) -> torch.Tensor:
        device = cu_seqlens.device
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        seq_ids = torch.repeat_interleave(
            torch.arange(len(lengths), device=device), lengths.long()
        )
        seq_starts = cu_seqlens[:-1][seq_ids]
        positions = torch.arange(total_tokens, device=device, dtype=torch.long)
        return positions - seq_starts.long()
