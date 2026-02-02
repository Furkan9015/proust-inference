"""GQA/MLA/GQA-S2 Transformer model for inference."""

import math
import torch
import torch.nn as nn

from .config import ModelConfig
from .attention import GQAAttention
from .ffn import ReluSquaredFFN
from .norm import RMSNorm
from .canon import create_canon, apply_canon, cu_seqlens_to_seq_idx


class TransformerBlock(nn.Module):
    """Transformer block with GQA attention."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn = GQAAttention(config, layer_idx)
        self.ffn = ReluSquaredFFN(config)
        self.attn_pre_norm = RMSNorm(config.hidden_dim)
        self.ffn_pre_norm = RMSNorm(config.hidden_dim)
        self.canonA = create_canon(config.hidden_dim, config) if "A" in config.canon_set else None
        self.canonC = create_canon(config.hidden_dim, config) if "C" in config.canon_set else None

    def forward(self, x, v_first=None, causal=True, cu_seqlens=None, max_seqlen=None,
                position_ids=None, seq_idx=None, value_embed=None):
        attn_in = self.attn_pre_norm(x)
        if self.canonA is not None:
            attn_in = apply_canon(self.canonA, attn_in, seq_idx=seq_idx)
        attn_out, v_first_out = self.attn(
            attn_in, v_first=v_first, causal=causal, cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen, position_ids=position_ids,
            value_embed=value_embed, seq_idx=seq_idx,
        )
        x = x + attn_out

        ffn_in = self.ffn_pre_norm(x)
        if self.canonC is not None:
            ffn_in = apply_canon(self.canonC, ffn_in, seq_idx=seq_idx)
        x = x + self.ffn(ffn_in, seq_idx=seq_idx)
        return x, v_first_out


class GQAS2TransformerBlock(nn.Module):
    """Transformer block with GQA-S2 attention."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        from .gqa_s2 import GQAS2Attention
        self.attn = GQAS2Attention(
            hidden_size=config.hidden_dim, num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            nope_head_dim=config.gqa_s2_nope_head_dim,
            rope_head_dim=config.gqa_s2_rope_head_dim,
            max_position_embeddings=config.max_seq_len, layer_idx=layer_idx,
            value_residual_lambda_init=config.value_residual_lambda_init,
            key_offset=config.key_offset,
            num_value_embeds=config.num_value_embeds,
            num_layers=config.num_layers,
            canon_set=config.canon_set, canon_kernel=config.canon_kernel,
            canon_bias=config.canon_bias, canon_activation=config.canon_activation,
            canon_residual=config.canon_residual,
        )
        self.ffn = ReluSquaredFFN(config)
        self.attn_pre_norm = RMSNorm(config.hidden_dim)
        self.ffn_pre_norm = RMSNorm(config.hidden_dim)
        self.canonA = create_canon(config.hidden_dim, config) if "A" in config.canon_set else None
        self.canonC = create_canon(config.hidden_dim, config) if "C" in config.canon_set else None

    def forward(self, x, v_first=None, causal=True, cu_seqlens=None, max_seqlen=None,
                position_ids=None, seq_idx=None, value_embed=None):
        attn_in = self.attn_pre_norm(x)
        if self.canonA is not None:
            attn_in = apply_canon(self.canonA, attn_in, seq_idx=seq_idx)
        attn_out, v_first_out = self.attn(
            attn_in, v_first=v_first, causal=causal, cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen, position_ids=position_ids,
            value_embed=value_embed, seq_idx=seq_idx,
        )
        x = x + attn_out

        ffn_in = self.ffn_pre_norm(x)
        if self.canonC is not None:
            ffn_in = apply_canon(self.canonC, ffn_in, seq_idx=seq_idx)
        x = x + self.ffn(ffn_in, seq_idx=seq_idx)
        return x, v_first_out


def _get_transformer_block(config: ModelConfig, layer_idx: int) -> nn.Module:
    if config.attention_type == "gqa_s2":
        return GQAS2TransformerBlock(config, layer_idx)
    elif config.attention_type == "gqa":
        return TransformerBlock(config, layer_idx)
    raise ValueError(f"Unsupported attention_type: {config.attention_type}")


class GQATransformer(nn.Module):
    """Transformer with GQA/MLA/GQA-S2 attention for inference.

    Args:
        config: Model configuration
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.use_canon = bool(config.canon_set)

        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.embed_scale = math.sqrt(config.hidden_dim)

        self.use_post_embed_norm = config.post_embed_norm
        if self.use_post_embed_norm:
            self.post_embed_norm = RMSNorm(config.hidden_dim)

        self.layers = nn.ModuleList([
            _get_transformer_block(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        self.final_norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # Value embeddings
        self.num_value_embeds = config.num_value_embeds
        self.layer_to_value_embed = {}
        if self.num_value_embeds > 0:
            if config.attention_type == "gqa_s2":
                ve_dim = config.num_kv_heads * config.gqa_s2_head_dim
            else:
                ve_dim = config.num_kv_heads * config.head_dim
            self.value_embeds = nn.ModuleList([
                nn.Embedding(config.vocab_size, ve_dim) for _ in range(self.num_value_embeds)
            ])
            n = self.num_value_embeds
            for i in range(n):
                self.layer_to_value_embed[i] = i
                self.layer_to_value_embed[config.num_layers - n + i] = i

    def _softcap_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.config.softcap_a > 0:
            A, B, C = self.config.softcap_a, self.config.softcap_b, self.config.softcap_c
            return A * torch.sigmoid((logits + B) / C)
        softcap = self.config.logit_softcap
        if softcap is None or softcap <= 0:
            return logits
        return softcap * torch.tanh(logits / softcap)

    def forward(
        self, input_ids: torch.Tensor,
        causal: bool = True,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        """Forward pass (inference only, no loss computation).

        Args:
            input_ids: (batch, seq) or (total_tokens,) for varlen
            causal: Whether to use causal masking
            cu_seqlens: Cumulative sequence lengths for packed varlen
            max_seqlen: Maximum sequence length for varlen

        Returns:
            Logits tensor
        """
        input_ids = input_ids.clamp(0, self.config.vocab_size - 1)

        x = self.embed(input_ids) * self.embed_scale
        if self.use_post_embed_norm:
            x = self.post_embed_norm(x)

        # Compute position_ids once for GQA-S2
        position_ids = None
        if cu_seqlens is not None and self.config.attention_type == "gqa_s2":
            total_tokens = input_ids.shape[0]
            position_ids = self._compute_positions(cu_seqlens, total_tokens)

        # Compute seq_idx for Canon layers
        seq_idx = None
        if cu_seqlens is not None and self.use_canon:
            total_tokens = input_ids.shape[0]
            seq_idx = cu_seqlens_to_seq_idx(cu_seqlens, total_tokens=total_tokens).unsqueeze(0)

        # Precompute value embeddings
        value_embed_cache = {}
        if self.num_value_embeds > 0:
            for ve_idx, ve in enumerate(self.value_embeds):
                value_embed_cache[ve_idx] = ve(input_ids)

        v_first = None
        for i, layer in enumerate(self.layers):
            ve = value_embed_cache.get(self.layer_to_value_embed.get(i)) if self.num_value_embeds > 0 else None
            x, v_first = layer(
                x, v_first, causal=causal, cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen, position_ids=position_ids,
                seq_idx=seq_idx, value_embed=ve,
            )

        x = self.final_norm(x)
        logits = self.lm_head(x)
        logits = self._softcap_logits(logits)
        return logits

    def get_embeddings(
        self, input_ids: torch.Tensor, causal: bool = False,
    ) -> torch.Tensor:
        """Extract hidden state embeddings (before lm_head).

        Args:
            input_ids: (batch, seq) token IDs
            causal: Whether to use causal masking (False for bidirectional)

        Returns:
            Hidden states (batch, seq, hidden_dim)
        """
        input_ids = input_ids.clamp(0, self.config.vocab_size - 1)
        x = self.embed(input_ids) * self.embed_scale
        if self.use_post_embed_norm:
            x = self.post_embed_norm(x)

        v_first = None
        for layer in self.layers:
            x, v_first = layer(x, v_first, causal=causal)
        return self.final_norm(x)

    def _compute_positions(self, cu_seqlens, total_tokens):
        """Compute per-token position IDs from cu_seqlens."""
        device = cu_seqlens.device
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        seq_ids = torch.repeat_interleave(
            torch.arange(len(lengths), device=device), lengths.long()
        )
        seq_starts = cu_seqlens[:-1][seq_ids]
        positions = torch.arange(total_tokens, device=device, dtype=torch.long)
        return positions - seq_starts.long()

    def load_state_dict_compat(self, state_dict: dict, strict: bool = True) -> None:
        """Load state dict with backward compatibility for older checkpoints."""
        model_state = self.state_dict()
        model_keys = set(model_state.keys())
        ckpt_keys = set(state_dict.keys())
        missing = model_keys - ckpt_keys
        extra = ckpt_keys - model_keys

        for key in missing:
            if "norm" in key and key.endswith(".weight"):
                state_dict[key] = torch.ones(model_state[key].shape)

        for key in extra:
            del state_dict[key]

        # Handle fused -> separate value_embeds migration
        if self.num_value_embeds > 0:
            if "value_embeds" in ckpt_keys and "value_embeds.0.weight" not in ckpt_keys:
                fused = state_dict.pop("value_embeds")
                for i, chunk in enumerate(fused.chunk(self.num_value_embeds, dim=0)):
                    state_dict[f"value_embeds.{i}.weight"] = chunk

        # Handle value_embed_gate shape migration (per-Q-head -> per-KV-head)
        for key in list(state_dict.keys()):
            if not key.endswith("value_embed_gate.weight") or key not in model_state:
                continue
            ckpt_w, model_w = state_dict[key], model_state[key]
            if ckpt_w.shape == model_w.shape:
                continue
            if (ckpt_w.ndim == 2 and model_w.ndim == 2
                    and ckpt_w.shape[1] == model_w.shape[1]
                    and ckpt_w.shape[0] == self.config.num_heads
                    and model_w.shape[0] == self.config.num_kv_heads):
                ratio = self.config.num_heads // self.config.num_kv_heads
                state_dict[key] = ckpt_w[::ratio].contiguous()

        self.load_state_dict(state_dict, strict=strict)
