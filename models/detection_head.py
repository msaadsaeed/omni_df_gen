"""
models/detection_head.py
─────────────────────────
Detection Head: shared MLP trunk + three task-specific output layers.

                  h  (B, 2d)
                  │
            ┌─────▼─────┐
            │  Trunk MLP │
            └─────┬─────┘
        ┌─────────┼─────────┐
        ▼         ▼         ▼
  head_audio  head_video  head_av
   (B, 2)      (B, 2)     (B, 4)

Task-head selection at inference time:
  • Audio-only  → use logits["audio"]
  • Video-only  → use logits["video"]
  • AV          → use logits["av"]

All three heads are computed every forward pass so all three losses
can be applied jointly during training.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    """
    Shared trunk MLP + three task-specific linear output heads.

    Args:
        in_dim     : input dimension (2*d from fusion)
        hidden_dim : trunk hidden dimension
        dropout    : dropout probability in the trunk
    """

    def __init__(
        self,
        in_dim:     int,
        hidden_dim: int   = 512,
        dropout:    float = 0.3,
    ) -> None:
        super().__init__()

        half = hidden_dim // 2

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, half),
            nn.LayerNorm(half),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Task heads
        self.head_audio = nn.Linear(half, 2)   # binary: real / fake audio
        self.head_video = nn.Linear(half, 2)   # binary: real / fake video
        self.head_av    = nn.Linear(half, 4)   # 4-class RARV/FAFV/RAFV/FARV

        self._init_weights()

    def _init_weights(self) -> None:
        for head in (self.head_audio, self.head_video, self.head_av):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            h : (B, 2*d)

        Returns:
            dict with keys 'audio', 'video', 'av' each containing raw logits
        """
        z = self.trunk(h)
        return {
            "audio": self.head_audio(z),   # (B, 2)
            "video": self.head_video(z),   # (B, 2)
            "av":    self.head_av(z),      # (B, 4)
        }
