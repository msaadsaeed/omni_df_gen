"""
models/projection.py
─────────────────────
Projection head: maps encoder output to shared latent dimension d.

Uses a two-layer MLP with LayerNorm and GELU activation.
This is more expressive than a plain linear projection and avoids the
need for the backbone and the cross-modal generators to share a vocabulary.

Input : (*, in_dim)
Output: (*, out_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection with LayerNorm and GELU activations.

    Args:
        in_dim  : input feature dimension (backbone output dim)
        out_dim : target shared dimension d
        dropout : optional dropout between layers (0 = disabled)
    """

    def __init__(
        self,
        in_dim:  int,
        out_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)