"""GQA Attention with QK-norm, gated output, and value residuals."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .norm import RMSNorm
from .rope import RotaryEmbedding
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
        _BACKEND = "fa4"
except ImportError:
    _fa4_is_available = lambda: False
    _activate_fa4 = lambda: False

try:
    from flash_attn import flash_attn_func as _flash_attn_func_fa2
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func_fa2
    if _BACKEND == "naive":
        _BACKEND = "fa2"
except ImportError:
    pass


def get_attention_backend() -> str:
    return _BACKEND


def init_attention_backend() -> str:
    global _BACKEND, _FA4_REGISTERED
    if _BACKEND == "fa4":
        _activate_fa4()
    return _BACKEND


class GQAAttention(nn.Module):
    """Grouped Query Attention with QK-norm, gated output, value residuals."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.gqa_ratio = config.gqa_ratio

        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim

        self.qkv_proj = nn.Linear(self.hidden_dim, self.q_dim + 2 * self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.q_dim, self.hidden_dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.post_sdpa_norm = RMSNorm(self.head_dim)

        self.gate = nn.Parameter(torch.zeros(self.num_heads))

        if layer_idx > 0:
            self.value_lambda = nn.Parameter(
                torch.tensor([config.value_residual_lambda_init, -config.value_residual_lambda_init])
            )

        self.use_value_embed = _should_use_value_embed(layer_idx, config.num_value_embeds, config.num_layers)
        if self.use_value_embed:
            self.value_embed_gate = nn.Linear(16, self.num_kv_heads, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.canonB = None
        if "B" in config.canon_set:
            self.canonB = create_canon(self.q_dim + 2 * self.kv_dim, config)

    def forward(
        self, x, v_first=None, causal=True,
        cu_seqlens=None, max_seqlen=None, position_ids=None,
        value_embed=None, seq_idx=None,
    ):
        is_varlen = cu_seqlens is not None

        if is_varlen:
            total_tokens = x.shape[0]
        else:
            batch, seq, _ = x.shape

        qkv = self.qkv_proj(x)
        if self.canonB is not None:
            qkv = apply_canon(self.canonB, qkv, seq_idx=seq_idx)

        if is_varlen:
            q = qkv[..., :self.q_dim].view(total_tokens, self.num_heads, self.head_dim)
            k = qkv[..., self.q_dim:self.q_dim + self.kv_dim].view(total_tokens, self.num_kv_heads, self.head_dim)
            v = qkv[..., self.q_dim + self.kv_dim:].view(total_tokens, self.num_kv_heads, self.head_dim)
        else:
            q = qkv[..., :self.q_dim].view(batch, seq, self.num_heads, self.head_dim)
            k = qkv[..., self.q_dim:self.q_dim + self.kv_dim].view(batch, seq, self.num_kv_heads, self.head_dim)
            v = qkv[..., self.q_dim + self.kv_dim:].view(batch, seq, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rope(q, k, position_ids=position_ids, cu_seqlens=cu_seqlens)

        if self.use_value_embed and value_embed is not None:
            gate_input = x[..., :16]
            ve_gate = 2.0 * torch.sigmoid(self.value_embed_gate(gate_input))
            if is_varlen:
                ve = value_embed.view(total_tokens, self.num_kv_heads, self.head_dim)
                ve_gate = ve_gate.view(total_tokens, self.num_kv_heads, 1)
            else:
                ve = value_embed.view(batch, seq, self.num_kv_heads, self.head_dim)
                ve_gate = ve_gate.view(batch, seq, self.num_kv_heads, 1)
            v = v + ve_gate * ve

        if v_first is None:
            v_first_out = v
        else:
            lam_v, lam_first = torch.sigmoid(self.value_lambda)
            v = (lam_v * v) + (lam_first * v_first)
            v_first_out = v_first

        if is_varlen:
            attn_out = self._attn_varlen(q, k, v, cu_seqlens, max_seqlen, causal)
            attn_out = self.post_sdpa_norm(attn_out)
            gate = torch.sigmoid(self.gate).view(1, self.num_heads, 1)
            attn_out = (gate * attn_out).reshape(total_tokens, self.q_dim)
        else:
            attn_out = self._attn(q, k, v, causal)
            attn_out = self.post_sdpa_norm(attn_out)
            gate = torch.sigmoid(self.gate).view(1, 1, self.num_heads, 1)
            attn_out = (gate * attn_out).reshape(batch, seq, self.q_dim)

        return self.o_proj(attn_out), v_first_out

    def _attn(self, q, k, v, causal):
        if _BACKEND == "fa4":
            return _flash_attn_func_fa4(q, k, v, softmax_scale=self.scale, causal=causal)
        elif _BACKEND == "fa2":
            return _flash_attn_func_fa2(q, k, v, softmax_scale=self.scale, causal=causal)
        return self._naive_attn(q, k, v, causal)

    def _attn_varlen(self, q, k, v, cu_seqlens, max_seqlen, causal):
        if _BACKEND == "fa4":
            return _flash_attn_varlen_func_fa4(
                q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                softmax_scale=self.scale, causal=causal,
            )
        elif _BACKEND == "fa2":
            return _flash_attn_varlen_func_fa2(
                q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                softmax_scale=self.scale, causal=causal,
            )
        return self._naive_attn_varlen(q, k, v, cu_seqlens, max_seqlen, causal)

    def _naive_attn(self, q, k, v, causal):
        batch, seq, num_heads, head_dim = q.shape
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.gqa_ratio, dim=2)
            v = v.repeat_interleave(self.gqa_ratio, dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if causal:
            mask = torch.triu(torch.ones(seq, seq, device=q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        out = torch.matmul(F.softmax(scores, dim=-1), v)
        return out.transpose(1, 2)

    def _naive_attn_varlen(self, q, k, v, cu_seqlens, max_seqlen, causal):
        total_tokens = q.shape[0]
        device = q.device
        if self.num_kv_heads != self.num_heads:
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
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(attn_mask.unsqueeze(0), float("-inf"))
        out = torch.matmul(F.softmax(scores, dim=-1), v)
        return out.transpose(0, 1)


def _should_use_value_embed(layer_idx: int, num_value_embeds: int, num_layers: int) -> bool:
    if num_value_embeds <= 0:
        return False
    return layer_idx < num_value_embeds or layer_idx >= num_layers - num_value_embeds
