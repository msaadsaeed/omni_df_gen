"""
losses/losses.py
─────────────────
Multi-task loss for OmniDeepfakeModel.

Total loss:
    L = λ_audio · CE(audio)  +  λ_video · CE(video)
      + λ_av    · CE(av)     +  λ_rec   · (NLL_a + NLL_v)

All three classification heads are trained jointly every step because
FakeAVCelebDataModule always returns all three labels.

The NLL reconstruction terms calibrate the generators: the model is
penalised both for inaccurate reconstructions AND for over-estimating
its uncertainty (the Gaussian NLL prevents trivially large σ).

Optionally applies class-frequency weighting to the AV head via
'av_class_weights' to handle any label imbalance in FakeAVCeleb.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class OmniLoss(nn.Module):
    """
    Weighted multi-task loss.

    Args:
        lambda_audio    : weight for audio binary CE
        lambda_video    : weight for video binary CE
        lambda_av       : weight for AV 4-class CE
        lambda_rec      : weight for reconstruction NLL terms
        av_class_weights: optional (4,) tensor of per-class weights for AV CE
                          (pass torch.tensor([w0,w1,w2,w3]) to address imbalance)
        label_smoothing : label smoothing for cross-entropy (0 = disabled)
    """

    def __init__(
        self,
        lambda_audio:     float = 0.3,
        lambda_video:     float = 0.3,
        lambda_av:        float = 1.0,
        lambda_rec:       float = 0.1,
        av_class_weights: Optional[torch.Tensor] = None,
        label_smoothing:  float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_audio    = lambda_audio
        self.lambda_video    = lambda_video
        self.lambda_av       = lambda_av
        self.lambda_rec      = lambda_rec
        self.label_smoothing = label_smoothing

        if av_class_weights is not None:
            self.register_buffer("av_class_weights", av_class_weights.float())
        else:
            self.av_class_weights = None

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch:   Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs : model output dict (logits_audio, logits_video,
                      logits_av, rec_loss_a, rec_loss_v)
            batch   : dataloader batch dict (audio_label, video_label, av_label)

        Returns:
            dict with keys: total, ce_audio, ce_video, ce_av, rec
        """
        ce_audio = F.cross_entropy(
            outputs["logits_audio"],
            batch["audio_label"],
            label_smoothing=self.label_smoothing,
        )
        ce_video = F.cross_entropy(
            outputs["logits_video"],
            batch["video_label"],
            label_smoothing=self.label_smoothing,
        )
        ce_av = F.cross_entropy(
            outputs["logits_av"],
            batch["av_label"],
            weight=(
                self.av_class_weights.to(outputs["logits_av"].device)
                if self.av_class_weights is not None else None
            ),
            label_smoothing=self.label_smoothing,
        )
        rec = outputs["rec_loss_a"] + outputs["rec_loss_v"]

        total = (
            self.lambda_audio * ce_audio
            + self.lambda_video * ce_video
            + self.lambda_av   * ce_av
            + self.lambda_rec  * rec
        )

        return {
            "total":    total,
            "ce_audio": ce_audio,
            "ce_video": ce_video,
            "ce_av":    ce_av,
            "rec":      rec,
        }
