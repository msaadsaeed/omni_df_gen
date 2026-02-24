"""
models/omni_model.py
─────────────────────
OmniDeepfakeModel — assembles all modules into the full pipeline.

Pipeline summary
────────────────
  1. AudioBackbone (Wav2Vec2 + HuBERT)   →  (B, T_a', D_audio)
  2. VideoBackbone (CLIP + FaceNet)       →  (B, T_v,  D_video)
  3. Audio ProjectionHead                 →  (B, T_a', d)  ← ha_seq
  4. Video ProjectionHead                 →  (B, T_v,  d)  ← hv_seq
  5. Ga←v  (video → audio features)      →  (B, T_a', d),  log_var_a
  6. Gv←a  (audio → video features)      →  (B, T_v,  d),  log_var_v
  7. Pool  ha, hv, ĥa, ĥv               →  (B, d) each
  8. UncertaintyAwareFusion               →  (B, 2d)
  9. DetectionHead                        →  logits_audio, logits_video, logits_av

All three heads are computed every forward pass.
Modality masks (ma, mv) control whether real or generated features are used
in the fusion step; they default to 1 (both present) for training.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.audio_backbone import AudioBackbone
from backbones.video_backbone import VideoBackbone
from models.projection import ProjectionHead
from models.generator import CrossModalGenerator
from models.fusion import UncertaintyAwareFusion
from models.detection_head import DetectionHead
from utils.pooling import mean_pool, make_key_padding_mask, downsample_lengths


class OmniDeepfakeModel(nn.Module):
    """
    Unified audio-visual deepfake detection model.

    Args:
        d                  : shared projection dimension
        freeze_backbones   : freeze pretrained backbone weights on init
        alpha              : initial reliability sharpness (learnable)
        max_tgt_len        : max sequence length for generator query tokens
        gen_nhead          : attention heads in cross-modal generators
        gen_enc_layers     : transformer encoder layers in generators
        gen_dec_layers     : transformer decoder layers in generators
        gen_dim_ff         : FFN hidden size in generators
        head_hidden        : hidden dim of the detection head trunk
        head_dropout       : dropout in the detection head
        wav2vec2_name      : HuggingFace id for Wav2Vec2
        hubert_name        : HuggingFace id for HuBERT
        audio_mode         : 'wav2vec2' or 'wav2vec2+hubert'
        clip_name          : HuggingFace id for CLIP ViT
        facenet_pretrained : 'vggface2' or 'casia-webface'
    """

    def __init__(
        self,
        d:                  int   = 256,
        freeze_backbones:   bool  = True,
        alpha:              float = 1.0,
        max_tgt_len:        int   = 128,
        gen_nhead:          int   = 4,
        gen_enc_layers:     int   = 2,
        gen_dec_layers:     int   = 2,
        gen_dim_ff:         int   = 512,
        head_hidden:        int   = 512,
        head_dropout:       float = 0.3,
        wav2vec2_name:      str   = "facebook/wav2vec2-base",
        hubert_name:        str   = "facebook/hubert-base-ls960",
        audio_mode:         str   = "wav2vec2+hubert",
        clip_name:          str   = "openai/clip-vit-base-patch32",
        facenet_pretrained: str   = "vggface2",
    ) -> None:
        super().__init__()
        self.d = d

        # Backbones
        self.audio_backbone = AudioBackbone(
            mode=audio_mode,
            wav2vec2_name=wav2vec2_name,
            hubert_name=hubert_name,
            freeze=freeze_backbones,
        )
        self.video_backbone = VideoBackbone(
            clip_name=clip_name,
            facenet_pretrained=facenet_pretrained,
            freeze=freeze_backbones,
        )

        # ── Projection heads ──────────────────────────────────────────────
        self.audio_proj = ProjectionHead(self.audio_backbone.out_dim, d)
        self.video_proj = ProjectionHead(self.video_backbone.out_dim, d)

        # ── Cross-modal generators ────────────────────────────────────────
        gen_kwargs = dict(
            d_model=d,
            nhead=gen_nhead,
            num_enc_layers=gen_enc_layers,
            num_dec_layers=gen_dec_layers,
            dim_feedforward=gen_dim_ff,
            max_tgt_len=max_tgt_len,
        )
        self.gen_audio_from_video = CrossModalGenerator(**gen_kwargs)   # Ga←v
        self.gen_video_from_audio = CrossModalGenerator(**gen_kwargs)   # Gv←a

        # ── Fusion & detection ────────────────────────────────────────────
        self.fusion         = UncertaintyAwareFusion(d=d, alpha=alpha)
        self.detection_head = DetectionHead(
            in_dim=2 * d,
            hidden_dim=head_hidden,
            dropout=head_dropout,
        )

    # ── Encoding helpers ──────────────────────────────────────────────────

    def _encode_audio(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            feat_seq  : (B, T_a', d)
            feat_pool : (B, d)
            pad_mask  : (B, T_a') bool  True=padding  |  None
        """
        raw      = self.audio_backbone(audio, audio_lengths)   # (B, T', D_audio)
        feat_seq = self.audio_proj(raw)                        # (B, T', d)

        T_prime  = feat_seq.size(1)
        pad_mask = None
        ds_len   = None

        if audio_lengths is not None:
            ds_len   = downsample_lengths(audio_lengths, audio.size(1), T_prime)
            pad_mask = make_key_padding_mask(ds_len, T_prime)

        feat_pool = mean_pool(feat_seq, ds_len)
        return feat_seq, feat_pool, pad_mask

    def _encode_video(
        self,
        video: torch.Tensor,
        video_lengths: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            feat_seq  : (B, T_v, d)
            feat_pool : (B, d)
            pad_mask  : (B, T_v) bool  True=padding  |  None
        """
        raw      = self.video_backbone(video)     # (B, T_v, D_video)
        feat_seq = self.video_proj(raw)           # (B, T_v, d)

        T_v      = feat_seq.size(1)
        pad_mask = None

        if video_lengths is not None:
            pad_mask = make_key_padding_mask(video_lengths, T_v)

        feat_pool = mean_pool(feat_seq, video_lengths)
        return feat_seq, feat_pool, pad_mask

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Accepts a batch dict from FakeAVCelebDataModule.

        Required keys:
            audio         (B, T_a)
            video         (B, T_v, 1, H, W)

        Optional keys (provided by dataloader):
            audio_lengths (B,)
            video_lengths (B,)
            ma            (B,)  float  1=audio present  0=absent  (default: all 1)
            mv            (B,)  float  1=video present  0=absent  (default: all 1)

        Returns dict:
            logits_audio  (B, 2)    binary audio-fake logits
            logits_video  (B, 2)    binary video-fake logits
            logits_av     (B, 4)    4-class AV logits
            rec_loss_a    scalar    audio reconstruction NLL loss
            rec_loss_v    scalar    video reconstruction NLL loss
        """
        audio         = batch["audio"]
        video         = batch["video"]
        audio_lengths = batch.get("audio_lengths")
        video_lengths = batch.get("video_lengths")
        B, device     = audio.size(0), audio.device

        # ── Step 1–4: Encode + project ────────────────────────────────────
        a_seq, ha, a_pad = self._encode_audio(audio, audio_lengths)
        v_seq, hv, v_pad = self._encode_video(video, video_lengths)

        T_a = a_seq.size(1)
        T_v = v_seq.size(1)

        # ── Step 5–6: Cross-modal generation ─────────────────────────────
        # Ga←v : video → audio
        ha_hat, log_var_a = self.gen_audio_from_video(
            src=v_seq, tgt_len=T_a, src_key_padding_mask=v_pad
        )   # (B, T_a, d),  (B, T_a, d)

        # Gv←a : audio → video
        hv_hat, log_var_v = self.gen_video_from_audio(
            src=a_seq, tgt_len=T_v, src_key_padding_mask=a_pad
        )   # (B, T_v, d),  (B, T_v, d)

        # ── Step 7: Pool generated features + compute mean uncertainty ────
        ha_hat_pool = ha_hat.mean(dim=1)      # (B, d)
        hv_hat_pool = hv_hat.mean(dim=1)      # (B, d)

        # Clamp log_var to a safe range before any use.
        # Without clamping, the model can predict very large negative values
        # (tiny σ) which cause exp(-s) to explode, making the NLL negative.
        # Range [-10, 10] allows σ in [~0.007, ~148] — more than sufficient.
        log_var_a = log_var_a.clamp(-4.0, 4.0)
        log_var_v = log_var_v.clamp(-4.0, 4.0)

        sa = log_var_a.mean(dim=(1, 2))       # (B,)  mean log σ² (audio)
        sv = log_var_v.mean(dim=(1, 2))       # (B,)  mean log σ² (video)

        # ── Reconstruction losses (Gaussian NLL) ──────────────────────────
        rec_loss_a = self._nll_loss(ha.unsqueeze(1).expand_as(ha_hat), ha_hat, log_var_a)
        rec_loss_v = self._nll_loss(hv.unsqueeze(1).expand_as(hv_hat), hv_hat, log_var_v)

        # ── Step 8: Modality masks ────────────────────────────────────────
        ma = batch.get("ma", torch.ones(B, device=device))
        mv = batch.get("mv", torch.ones(B, device=device))

        # ── Step 8: Uncertainty-aware fusion ─────────────────────────────
        h = self.fusion(ha, hv, ha_hat_pool, hv_hat_pool, sa, sv, ma, mv)

        # ── Step 9: Detection ─────────────────────────────────────────────
        logits = self.detection_head(h)

        return {
            "logits_audio": logits["audio"],
            "logits_video": logits["video"],
            "logits_av":    logits["av"],
            "rec_loss_a":   rec_loss_a,
            "rec_loss_v":   rec_loss_v,
        }

    @staticmethod
    def _nll_loss(
        target:  torch.Tensor,
        pred:    torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stabilised Gaussian NLL reconstruction loss.

        The raw projected features can have large magnitudes (std >> 1),
        which makes the squared-diff term dominate and drives the model to
        predict very negative log_var (tiny σ) to compensate — causing the
        NLL to go negative.

        Fix: L2-normalise both target and prediction along the feature dim
        before computing the loss.  This maps everything onto the unit
        hypersphere so squared distances are bounded in [0, 4], regardless
        of the original feature scale.  The loss is then well-behaved and
        always positive for reasonable log_var values.
        """
        # L2-normalise along feature dimension (last dim)
        target_n = F.normalize(target.detach(), dim=-1)
        pred_n   = F.normalize(pred,            dim=-1)

        diff    = (target_n - pred_n).pow(2)              # bounded in [0, 4]
        exp_neg = (-log_var).exp().clamp(max=1e2)         # safety net
        return 0.5 * (log_var + diff * exp_neg).mean()

    # ── Convenience: count trainable parameters ───────────────────────────

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())