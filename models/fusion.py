"""
models/fusion.py
─────────────────
Uncertainty-Aware Fusion module.

Fuses real and generated modality features according to the modality mask
(ma, mv) and uncertainty-derived reliability weights (wa, wv).

Fusion rules
────────────
  ma=1, mv=1 :  wa·ha  ⊕  wv·hv       (both real, still reliability-weighted)
  ma=0, mv=1 :  wa·ĥa  ⊕  wv·hv       (audio generated, video real)
  ma=1, mv=0 :  wa·ha  ⊕  wv·ĥv       (audio real, video generated)

  where  wm = exp(−α · mean(sm))
         sm  = mean token-level log-variance from the generator

  α is a learnable scalar (initialised from the config value) so the
  model can adapt how aggressively uncertainty suppresses features.

The output is the concatenation [audio_feat, video_feat] of shape (B, 2d),
which is fed directly to the DetectionHead.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class UncertaintyAwareFusion(nn.Module):
    """
    Reliability-weighted fusion of real and generated modality features.

    Args:
        d     : per-modality feature dimension
        alpha : initial reliability sharpness  (learnable)
    """

    def __init__(self, d: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.d         = d
        self.log_alpha = nn.Parameter(torch.tensor(math.log(max(alpha, 1e-6))))

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        ha:     torch.Tensor,   # (B, d)  real audio features
        hv:     torch.Tensor,   # (B, d)  real video features
        ha_hat: torch.Tensor,   # (B, d)  generated audio features
        hv_hat: torch.Tensor,   # (B, d)  generated video features
        sa:     torch.Tensor,   # (B,)    mean log-variance (audio generator)
        sv:     torch.Tensor,   # (B,)    mean log-variance (video generator)
        ma:     torch.Tensor,   # (B,)    float 1 = audio present, 0 = absent
        mv:     torch.Tensor,   # (B,)    float 1 = video present, 0 = absent
    ) -> torch.Tensor:
        """
        Returns:
            h : (B, 2*d)  fused representation
        """
        alpha = self.log_alpha.exp()

        # Reliability weights — high σ ⟹ high log σ² ⟹ low weight
        wa = torch.exp(-alpha * sa).unsqueeze(-1)   # (B, 1)
        wv = torch.exp(-alpha * sv).unsqueeze(-1)   # (B, 1)

        ma = ma.unsqueeze(-1).float()               # (B, 1)
        mv = mv.unsqueeze(-1).float()               # (B, 1)

        # Audio component
        #   present  → wa * real ha
        #   absent   → wa * generated ĥa   (already uncertainty-suppressed)
        audio_feat = ma * (wa * ha) + (1.0 - ma) * (wa * ha_hat)

        # Video component
        video_feat = mv * (wv * hv) + (1.0 - mv) * (wv * hv_hat)

        return torch.cat([audio_feat, video_feat], dim=-1)   # (B, 2d)

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())
