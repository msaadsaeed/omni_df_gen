"""
utils/pooling.py
─────────────────
Pooling and masking utilities shared across model modules.
"""

from __future__ import annotations

from typing import Optional

import torch


def mean_pool(
    x: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Length-aware mean pooling over the time dimension.

    Args:
        x       : (B, T, D)
        lengths : (B,)  number of *valid* tokens; None → all tokens valid

    Returns:
        (B, D)
    """
    if lengths is None:
        return x.mean(dim=1)

    mask = (
        torch.arange(x.size(1), device=x.device)
        .unsqueeze(0)
        .lt(lengths.unsqueeze(1))
    )                                    # (B, T)  bool
    mask_f = mask.unsqueeze(-1).float()  # (B, T, 1)
    return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


def make_key_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Build a key-padding mask suitable for nn.MultiheadAttention.

    Returns a bool tensor where True = PADDING (position should be ignored).

    Args:
        lengths : (B,)  number of valid tokens
        max_len : int   sequence length T

    Returns:
        (B, T)  bool
    """
    return (
        torch.arange(max_len, device=lengths.device)
        .unsqueeze(0)
        .ge(lengths.unsqueeze(1))
    )


def downsample_lengths(
    lengths: torch.Tensor,
    original_len: int,
    downsampled_len: int,
) -> torch.Tensor:
    """
    Scale lengths from original sequence space to downsampled space.
    Useful when audio encoders subsample with a fixed stride.

    Args:
        lengths        : (B,)  lengths in original space
        original_len   : T before downsampling
        downsampled_len: T' after downsampling

    Returns:
        (B,)  lengths clipped to [0, downsampled_len]
    """
    if original_len == 0:
        return lengths.clamp(max=downsampled_len)
    scale = downsampled_len / original_len
    return (lengths.float() * scale).long().clamp(max=downsampled_len)
