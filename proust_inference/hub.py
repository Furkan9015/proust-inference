"""Load proust models from HuggingFace Hub."""

import torch
from pathlib import Path

from .config import ModelConfig
from .model import GQATransformer


DEFAULT_REPO = "nappenstance/proust_v0"
DEFAULT_FILENAME = "checkpoint_step62474_loss2.3879_final.pt"


def load_model(
    repo_id: str = DEFAULT_REPO,
    filename: str = DEFAULT_FILENAME,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    local_path: str | None = None,
) -> GQATransformer:
    """Load a proust model from HuggingFace Hub or local path.

    Args:
        repo_id: HuggingFace repo ID
        filename: Checkpoint filename in the repo
        device: Device to load model onto
        dtype: Model dtype (bfloat16 recommended)
        local_path: If set, load from this local .pt file instead of HF

    Returns:
        Loaded model in eval mode
    """
    if local_path is not None:
        ckpt_path = local_path
    else:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "config" not in checkpoint:
        raise ValueError("Checkpoint missing 'config' key. Not a proust checkpoint?")

    # Filter to only fields ModelConfig knows about (checkpoints may have extra fields)
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(ModelConfig)}
    config_dict = {k: v for k, v in checkpoint["config"].items() if k in valid_fields}
    config = ModelConfig(**config_dict)
    model = GQATransformer(config)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict_compat(state_dict)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model
