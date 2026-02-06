"""GQA-S2: GQA with Partial RoPE and KV-Sharing (S2 scheme) for inference."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm
from .canon import create_canon, apply_canon

# Backend detection
_BACKEND = "naive"

try:
    from .fa4_backend import (
        is_available as _fa4_is_available,
        activate_fa4 as _activate_fa4,
        flash_attn_func as _flash_attn_func_fa4,
        flash_attn_varlen_func as _flash_attn_varlen_func_fa4,
    )
    if _fa4_is_available():
        _activate_fa4()
        _BACKEND = "fa4"
except ImportError:
    pass

try:
    from flash_attn import flash_attn_func as _flash_attn_func_fa2
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func_fa2
    if _BACKEND == "naive":
        _BACKEND = "fa2"
except ImportError:
    pass


class PartialRotaryEmbeddingS2(nn.Module):
    """Partial RoPE for S2 scheme: applies RoPE to Q, K, V rope portions + inverse RoPE on output."""

    def __init__(self, rope_dim: int, nope_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.rope_dim = rope_dim
        self.nope_dim = nope_dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
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

    def _get_cos_sin(self, x, cu_seqlens, max_seqlen, position_ids):
        is_varlen = x.dim() == 3 and cu_seqlens is not None

        if max_seqlen is not None and max_seqlen > self.max_seq_len:
            self._update_cache(max_seqlen)

        if is_varlen:
            if position_ids is None:
                position_ids = self._compute_varlen_positions(cu_seqlens, x.shape[0])
            cos = self.cos_cached[position_ids].unsqueeze(1)
            sin = self.sin_cached[position_ids].unsqueeze(1)
        else:
            if position_ids is not None:
                cos = self.cos_cached[position_ids].unsqueeze(2)
                sin = self.sin_cached[position_ids].unsqueeze(2)
            else:
                seq_len = x.shape[-3]
                cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2)
                sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)
        return cos, sin

    def forward(self, q, k, v, cu_seqlens=None, max_seqlen=None, position_ids=None):
        nope = self.nope_dim
        cos, sin = self._get_cos_sin(q, cu_seqlens, max_seqlen, position_ids)

        q_rope = q[..., nope:]
        k_rope = k[..., nope:]
        v_rope = v[..., nope:]

        q = torch.cat([q[..., :nope], q_rope * cos + self._rotate_half(q_rope) * sin], dim=-1)
        k = torch.cat([k[..., :nope], k_rope * cos + self._rotate_half(k_rope) * sin], dim=-1)
        v = torch.cat([v[..., :nope], v_rope * cos + self._rotate_half(v_rope) * sin], dim=-1)
        return q, k, v

    def inverse_rope(self, x, cu_seqlens=None, max_seqlen=None, position_ids=None):
        nope = self.nope_dim
        cos, sin = self._get_cos_sin(x, cu_seqlens, max_seqlen, position_ids)
        x_rope = x[..., nope:]
        return torch.cat([x[..., :nope], x_rope * cos - self._rotate_half(x_rope) * sin], dim=-1)

    def _compute_varlen_positions(self, cu_seqlens, total_tokens):
        device = cu_seqlens.device
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        seq_ids = torch.repeat_interleave(torch.arange(len(lengths), device=device), lengths.long())
        starts = torch.cat([torch.zeros(1, device=device, dtype=cu_seqlens.dtype), lengths.cumsum(0)[:-1]])
        positions = torch.arange(total_tokens, device=device, dtype=torch.long)
        return positions - starts[seq_ids].long()


def _should_use_value_embed(layer_idx: int, num_value_embeds: int, num_layers: int) -> bool:
    if num_value_embeds <= 0:
        return False
    return layer_idx < num_value_embeds or layer_idx >= num_layers - num_value_embeds


class GQAS2Attention(nn.Module):
    """GQA with S2 KV-Sharing and Partial RoPE for inference."""

    def __init__(
        self, hidden_size=1024, num_heads=16, num_kv_heads=2,
        nope_head_dim=192, rope_head_dim=64, max_position_embeddings=2048,
        rope_theta=10000.0, layer_idx=0, value_residual_lambda_init=0.5,
        key_offset=False, num_value_embeds=0, num_layers=24,
        canon_set="", canon_kernel=4, canon_bias=False,
        canon_activation=False, canon_residual=True,
        query_dependent_gate=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.nope_head_dim = nope_head_dim
        self.rope_head_dim = rope_head_dim
        self.layer_idx = layer_idx
        self.key_offset = key_offset
        self.query_dependent_gate = query_dependent_gate
        self.head_dim = nope_head_dim + rope_head_dim

        assert num_heads % num_kv_heads == 0
        self.gqa_ratio = num_heads // num_kv_heads
        self.scaling = self.head_dim ** (-0.5)

        self.q_dim = num_heads * self.head_dim
        self.kv_dim = num_kv_heads * self.head_dim
        self.gate_dim = num_heads if query_dependent_gate else 0
        self.qkv_proj = nn.Linear(hidden_size, self.q_dim + self.gate_dim + self.kv_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.post_sdpa_norm = RMSNorm(self.head_dim)

        self.rotary = PartialRotaryEmbeddingS2(
            rope_dim=rope_head_dim, nope_dim=nope_head_dim,
            max_seq_len=max_position_embeddings, base=rope_theta,
        )

        if layer_idx > 0:
            self.value_lambda = nn.Parameter(
                torch.tensor([value_residual_lambda_init, -value_residual_lambda_init])
            )

        if not query_dependent_gate:
            self.gate = nn.Parameter(torch.zeros(num_heads))

        self.use_value_embed = _should_use_value_embed(layer_idx, num_value_embeds, num_layers)
        if self.use_value_embed:
            self.value_embed_gate = nn.Linear(16, num_kv_heads, bias=False)

        self.canonB = None
        if "B" in canon_set:
            class _Cfg:
                pass
            cfg = _Cfg()
            cfg.canon_kernel = canon_kernel
            cfg.canon_bias = canon_bias
            cfg.canon_activation = canon_activation
            cfg.canon_residual = canon_residual
            self.canonB = create_canon(self.q_dim + self.gate_dim + self.kv_dim, cfg)

    def _apply_key_offset(self, k, cu_seqlens, is_varlen, position_ids=None):
        nope_dim = self.nope_head_dim
        if is_varlen and position_ids is not None:
            total_tokens = k.shape[0]
            if total_tokens == 0 or nope_dim == 0:
                return k
            k_shifted = torch.empty_like(k)
            k_shifted[:, :, nope_dim:] = k[:, :, nope_dim:]
            k_shifted[0, :, :nope_dim] = k[0, :, :nope_dim]
            if total_tokens > 1:
                k_shifted[1:, :, :nope_dim] = k[:-1, :, :nope_dim]
            start_mask = (position_ids == 0).view(total_tokens, 1, 1)
            k_shifted[:, :, :nope_dim] = torch.where(
                start_mask, k[:, :, :nope_dim], k_shifted[:, :, :nope_dim],
            )
            return k_shifted
        else:
            k_shifted = k.clone()
            k_shifted[:, 1:, :, :nope_dim] = k[:, :-1, :, :nope_dim]
            return k_shifted

    def forward(
        self, hidden_states, v_first=None, causal=True,
        cu_seqlens=None, max_seqlen=None, position_ids=None,
        value_embed=None, seq_idx=None,
    ):
        is_varlen = cu_seqlens is not None
        if is_varlen:
            total_tokens = hidden_states.shape[0]
            batch_size, seq_len = 1, total_tokens
        else:
            batch_size, seq_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        if self.canonB is not None:
            qkv = apply_canon(self.canonB, qkv, seq_idx=seq_idx)

        q_end = self.q_dim
        gate_end = q_end + self.gate_dim
        if is_varlen:
            q = qkv[..., :q_end].view(total_tokens, self.num_heads, self.head_dim)
            if self.query_dependent_gate:
                qd_gate_logits = qkv[..., q_end:gate_end]
            kv = qkv[..., gate_end:].view(total_tokens, self.num_kv_heads, self.head_dim)
        else:
            q = qkv[..., :q_end].view(batch_size, seq_len, self.num_heads, self.head_dim)
            if self.query_dependent_gate:
                qd_gate_logits = qkv[..., q_end:gate_end]
            kv = qkv[..., gate_end:].view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        k = kv
        v = kv

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k, v = self.rotary(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, position_ids=position_ids)

        if self.key_offset:
            k = self._apply_key_offset(k, cu_seqlens, is_varlen, position_ids)

        if self.use_value_embed and value_embed is not None:
            gate_input = hidden_states[..., :16]
            ve_gate = 2.0 * torch.sigmoid(self.value_embed_gate(gate_input))
            if is_varlen:
                ve = value_embed.view(total_tokens, self.num_kv_heads, self.head_dim)
                ve_gate = ve_gate.view(total_tokens, self.num_kv_heads, 1)
            else:
                ve = value_embed.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
                ve_gate = ve_gate.view(batch_size, seq_len, self.num_kv_heads, 1)
            v = v + ve_gate * ve

        if v_first is None:
            v_first_out = v
        else:
            lam_v, lam_first = torch.sigmoid(self.value_lambda)
            v = (lam_v * v) + (lam_first * v_first)
            v_first_out = v_first

        if is_varlen:
            attn_out = self._attn_varlen(q, k, v, cu_seqlens, max_seqlen, causal)
        else:
            attn_out = self._attn(q, k, v, causal)

        attn_out = self.rotary.inverse_rope(attn_out, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, position_ids=position_ids)
        attn_out = self.post_sdpa_norm(attn_out)

        if self.query_dependent_gate:
            if is_varlen:
                attn_out = (torch.sigmoid(qd_gate_logits).unsqueeze(-1) * attn_out).view(total_tokens, -1)
            else:
                attn_out = (torch.sigmoid(qd_gate_logits).unsqueeze(-1) * attn_out).view(batch_size, seq_len, -1)
        else:
            if is_varlen:
                attn_out = (torch.sigmoid(self.gate).view(1, self.num_heads, 1) * attn_out).view(total_tokens, -1)
            else:
                attn_out = (torch.sigmoid(self.gate).view(1, 1, self.num_heads, 1) * attn_out).view(batch_size, seq_len, -1)

        return self.o_proj(attn_out), v_first_out

    def _attn(self, q, k, v, causal):
        if _BACKEND == "fa4":
            return _flash_attn_func_fa4(q, k, v, softmax_scale=self.scaling, causal=causal)
        elif _BACKEND == "fa2":
            return _flash_attn_func_fa2(q, k, v, softmax_scale=self.scaling, causal=causal)
        return self._naive_attn(q, k, v, causal)

    def _attn_varlen(self, q, k, v, cu_seqlens, max_seqlen, causal):
        if _BACKEND == "fa4":
            return _flash_attn_varlen_func_fa4(
                q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                softmax_scale=self.scaling, causal=causal,
            )
        elif _BACKEND == "fa2":
            return _flash_attn_varlen_func_fa2(
                q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                softmax_scale=self.scaling, causal=causal,
            )
        return self._naive_attn_varlen(q, k, v, cu_seqlens, max_seqlen, causal)

    def _naive_attn(self, q, k, v, causal):
        batch, seq, heads, head_dim = q.shape
        if k.shape[2] != heads:
            k = k.repeat_interleave(self.gqa_ratio, dim=2)
            v = v.repeat_interleave(self.gqa_ratio, dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if causal:
            mask = torch.triu(torch.ones(seq, seq, device=q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        out = torch.matmul(F.softmax(scores, dim=-1), v)
        return out.transpose(1, 2)

    def _naive_attn_varlen(self, q, k, v, cu_seqlens, max_seqlen, causal):
        total_tokens, heads, head_dim = q.shape
        device = q.device
        if k.shape[1] != heads:
            k = k.repeat_interleave(self.gqa_ratio, dim=1)
            v = v.repeat_interleave(self.gqa_ratio, dim=1)
        mask = torch.zeros(total_tokens, total_tokens, device=device, dtype=torch.bool)
        for i in range(len(cu_seqlens) - 1):
            s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            mask[s:e, s:e] = True
        if causal:
            mask = mask & torch.tril(torch.ones_like(mask))
        attn_mask = ~mask
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        scores = scores.masked_fill(attn_mask.unsqueeze(0), float("-inf"))
        out = torch.matmul(F.softmax(scores, dim=-1), v)
        return out.transpose(0, 1)
