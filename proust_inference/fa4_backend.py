"""FA4 backend using PyTorch's native attention dispatcher."""

import torch

_FA4_AVAILABLE = False
_FA4_ACTIVATED = False


def is_available() -> bool:
    global _FA4_AVAILABLE
    if _FA4_ACTIVATED:
        return True
    if _FA4_AVAILABLE:
        return True
    try:
        from torch.nn.attention import current_flash_attention_impl
        if current_flash_attention_impl() == "FA4":
            return True
    except (ImportError, AttributeError):
        pass
    try:
        from torch.nn.attention import list_flash_attention_impls
        _FA4_AVAILABLE = "FA4" in list_flash_attention_impls()
        return _FA4_AVAILABLE
    except (ImportError, AttributeError):
        return False


def activate_fa4() -> bool:
    global _FA4_ACTIVATED
    if _FA4_ACTIVATED:
        return True
    if not is_available():
        return False
    try:
        from torch.nn.attention import activate_flash_attention_impl
        activate_flash_attention_impl("FA4")
        _FA4_ACTIVATED = True
        return True
    except Exception:
        return False


def flash_attn_varlen_func(
    q, k, v, cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    softmax_scale=None, causal=False, **_kw,
) -> torch.Tensor:
    if not _FA4_ACTIVATED:
        activate_fa4()
    out, *_ = torch.ops.aten._flash_attention_forward(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        0.0, causal, return_debug_mask=False, scale=softmax_scale,
    )
    return out


def flash_attn_func(
    q, k, v, softmax_scale=None, causal=False, **_kw,
) -> torch.Tensor:
    if not _FA4_ACTIVATED:
        activate_fa4()
    batch, seqlen, num_heads, head_dim = q.shape
    cu_seqlens = torch.arange(
        0, (batch + 1) * seqlen, seqlen, dtype=torch.int32, device=q.device,
    )
    q_flat = q.reshape(-1, num_heads, head_dim)
    k_flat = k.reshape(-1, k.shape[2], head_dim)
    v_flat = v.reshape(-1, v.shape[2], head_dim)
    out_flat = flash_attn_varlen_func(
        q_flat, k_flat, v_flat, cu_seqlens, cu_seqlens,
        seqlen, seqlen, softmax_scale=softmax_scale, causal=causal,
    )
    return out_flat.reshape(batch, seqlen, num_heads, head_dim)
