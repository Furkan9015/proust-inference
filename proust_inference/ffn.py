"""Feed-Forward Network with ReLU^2 activation for inference."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .canon import create_canon, apply_canon


class ReluSquaredFFN(nn.Module):
    """FFN: Linear -> ReLU^2 -> Linear, 4x expansion."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.ffn_dim = config.ffn_dim

        self.up_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False)

        self.canonD = None
        if "D" in config.canon_set:
            self.canonD = create_canon(self.ffn_dim, config)

    def forward(self, x: torch.Tensor, seq_idx: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up_proj(x)
        if self.canonD is not None:
            x = apply_canon(self.canonD, x, seq_idx=seq_idx)
        x = F.relu(x).square()
        x = self.down_proj(x)
        return x
