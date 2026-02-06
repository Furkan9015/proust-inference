"""Model configuration."""

import math
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class ModelConfig:
    """Configuration for proust Transformer model."""

    # Architecture
    hidden_dim: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 4
    head_dim: Optional[int] = None
    ffn_dim: Optional[int] = None
    vocab_size: int = 50257
    max_seq_len: int = 2048

    # Attention type
    attention_type: Literal["gqa", "gqa_s2"] = "gqa"

    # GQA-S2 parameters
    gqa_s2_nope_head_dim: int = 96
    gqa_s2_rope_head_dim: int = 32

    # Value residual
    value_residual_lambda_init: float = 0.5
    value_residual_lambda_lr_mult: float = 30.0

    # dtype
    dtype: str = "bfloat16"

    # U-net skip connections
    unet_skip_connections: bool = False

    # Hyper-Connections
    hyper_connections_streams: int = 1

    # Key offset
    key_offset: bool = False

    # Value embeddings
    num_value_embeds: int = 0

    # Sigmoid softcap
    softcap_a: float = 0.0
    softcap_b: float = 5.0
    softcap_c: float = 7.5

    # Legacy tanh softcap
    logit_softcap: float = 0.0

    # Canon layers
    canon_set: str = ""
    canon_kernel: int = 4
    canon_bias: bool = False
    canon_activation: bool = False
    canon_residual: bool = True

    # Query-dependent gating (NeurIPS 2025 Gated Attention)
    query_dependent_gate: bool = False

    # Post-embedding RMSNorm
    post_embed_norm: bool = False

    # Weight tying
    tie_embeddings: bool = False

    # muP scaling (not used at inference, kept for checkpoint compat)
    base_hidden_dim: int = 1024
    mup_enabled: bool = False
    mup_scale_init: bool = False

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_dim // self.num_heads
        if self.ffn_dim is None:
            self.ffn_dim = 4 * self.hidden_dim

        assert self.hidden_dim % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0

        if self.attention_type == "gqa_s2":
            assert self.gqa_s2_nope_head_dim > 0
            assert self.gqa_s2_rope_head_dim > 0

    @property
    def init_std(self) -> float:
        return 0.5 / math.sqrt(self.hidden_dim)

    @property
    def sandwich_norm_scale(self) -> float:
        return 1.0 / math.sqrt(self.num_layers)

    @property
    def gqa_ratio(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def gqa_s2_head_dim(self) -> int:
        return self.gqa_s2_nope_head_dim + self.gqa_s2_rope_head_dim


# Preset model configurations
MODEL_CONFIGS = {
    "50m": ModelConfig(
        hidden_dim=512, num_layers=16, num_heads=8, num_kv_heads=2,
        base_hidden_dim=512, attention_type="gqa",
    ),
    "gqa-s2-50m": ModelConfig(
        hidden_dim=512, num_layers=16, num_heads=8, num_kv_heads=2,
        base_hidden_dim=512, attention_type="gqa_s2",
        gqa_s2_nope_head_dim=48, gqa_s2_rope_head_dim=16,
    ),
    "350m": ModelConfig(
        hidden_dim=1024, num_layers=24, num_heads=16, num_kv_heads=4,
        base_hidden_dim=1024, attention_type="gqa",
    ),
    "1.5b": ModelConfig(
        hidden_dim=2048, num_layers=28, num_heads=32, num_kv_heads=8,
        base_hidden_dim=1024, attention_type="gqa",
    ),
    "7b": ModelConfig(
        hidden_dim=4096, num_layers=32, num_heads=32, num_kv_heads=8,
        base_hidden_dim=1024, attention_type="gqa",
    ),
    "gqa-s2-350m": ModelConfig(
        hidden_dim=1024, num_layers=24, num_heads=16, num_kv_heads=2,
        base_hidden_dim=1024, attention_type="gqa_s2",
        gqa_s2_nope_head_dim=96, gqa_s2_rope_head_dim=32,
    ),
    "gqa-s2-350m-large": ModelConfig(
        hidden_dim=1024, num_layers=24, num_heads=16, num_kv_heads=2,
        base_hidden_dim=1024, attention_type="gqa_s2",
        gqa_s2_nope_head_dim=192, gqa_s2_rope_head_dim=64,
    ),
    "gqa-s2-350m-g4": ModelConfig(
        hidden_dim=1024, num_layers=24, num_heads=16, num_kv_heads=4,
        base_hidden_dim=1024, attention_type="gqa_s2",
        gqa_s2_nope_head_dim=96, gqa_s2_rope_head_dim=32,
    ),
    "gqa-s2-1.5b": ModelConfig(
        hidden_dim=2048, num_layers=28, num_heads=32, num_kv_heads=4,
        base_hidden_dim=1024, attention_type="gqa_s2",
        gqa_s2_nope_head_dim=96, gqa_s2_rope_head_dim=32,
    ),
}


def get_model_config(name: str) -> ModelConfig:
    """Get a preset model configuration by name."""
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model config: {name}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[name]
