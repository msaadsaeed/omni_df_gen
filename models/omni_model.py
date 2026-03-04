"""
models/omni_model.py
─────────────────────
OmniDeepfakeModel — assembles all modules into the full pipeline.

Pipeline summary
────────────────
  1. AudioBackbone (Wav2Vec2 + HuBERT)   →  (B, T_a', D_audio)   [if audio present]
  2. VideoBackbone (CLIP + FaceNet)       →  (B, T_v,  D_video)   [if video present]
  3. Audio ProjectionHead                 →  (B, T_a', d)
  4. Video ProjectionHead                 →  (B, T_v,  d)
  5. Ga←v  (video → audio features)      →  (B, T_a', d),  log_var_a
  6. Gv←a  (audio → video features)      →  (B, T_v,  d),  log_var_v
  7. Pool  ha, hv (real or generated)    →  (B, d) each
  8. UncertaintyAwareFusion               →  (B, 2d)
  9. DetectionHead                        →  logits_audio, logits_video, logits_av

Missing-modality handling
─────────────────────────
  Both present  : encode both → generate both → rec_loss → fuse real features
  Audio missing : encode video → generate audio via Ga←v → fuse (generated + real)
  Video missing : encode audio → generate video via Gv←a → fuse (real + generated)

  rec_loss is only computed when both modalities are real (no target otherwise).
  The CE loss provides the training signal through the generator when a modality
  is absent — forcing the generator to produce discriminatively useful features.
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

# Default target sequence lengths used when generating a missing modality.
_DEFAULT_TGT_LEN_AUDIO = 50   # ~1s of audio at Wav2Vec2 stride=320
_DEFAULT_TGT_LEN_VIDEO = 30   # ~1s of video at 30 fps


class OmniDeepfakeModel(nn.Module):
    """
    Unified audio-visual deepfake detection model with missing-modality support.

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
        """Returns: feat_seq (B,T',d), feat_pool (B,d), pad_mask (B,T')|None"""
        raw      = self.audio_backbone(audio, audio_lengths)
        feat_seq = self.audio_proj(raw)
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
        """Returns: feat_seq (B,T_v,d), feat_pool (B,d), pad_mask (B,T_v)|None"""
        raw      = self.video_backbone(video)
        feat_seq = self.video_proj(raw)
        T_v      = feat_seq.size(1)
        pad_mask = None
        if video_lengths is not None:
            pad_mask = make_key_padding_mask(video_lengths, T_v)
        feat_pool = mean_pool(feat_seq, video_lengths)
        return feat_seq, feat_pool, pad_mask

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Accepts a batch dict.  At least one of 'audio' or 'video' must be present
        (the other may be None or absent entirely).

        Required keys (at least one):
            audio         (B, T_a)           — None/absent for video-only mode
            video         (B, T_v, 1, H, W)  — None/absent for audio-only mode

        Optional keys:
            audio_lengths (B,)
            video_lengths (B,)
            default_tgt_len_audio  int  token budget when generating audio (default 50)
            default_tgt_len_video  int  token budget when generating video (default 30)

        Returns dict:
            logits_audio  (B, 2)    binary audio-fake logits
            logits_video  (B, 2)    binary video-fake logits
            logits_av     (B, 4)    4-class AV logits
            rec_loss_a    scalar    audio reconstruction NLL (0.0 when audio missing)
            rec_loss_v    scalar    video reconstruction NLL (0.0 when video missing)
        """
        audio         = batch.get("audio")
        video         = batch.get("video")
        audio_lengths = batch.get("audio_lengths")
        video_lengths = batch.get("video_lengths")

        audio_missing = audio is None
        video_missing = video is None

        if audio_missing and video_missing:
            raise ValueError("At least one of 'audio' or 'video' must be provided.")

        device = audio.device if not audio_missing else video.device
        B      = audio.size(0) if not audio_missing else video.size(0)

        tgt_len_audio = batch.get("default_tgt_len_audio", _DEFAULT_TGT_LEN_AUDIO)
        tgt_len_video = batch.get("default_tgt_len_video", _DEFAULT_TGT_LEN_VIDEO)

        # ── Case 1: both modalities present ──────────────────────────────
        if not audio_missing and not video_missing:
            a_seq, ha, a_pad = self._encode_audio(audio, audio_lengths)
            v_seq, hv, v_pad = self._encode_video(video, video_lengths)

            T_a = a_seq.size(1)
            T_v = v_seq.size(1)

            ha_hat, log_var_a = self.gen_audio_from_video(
                src=v_seq, tgt_len=T_a, src_key_padding_mask=v_pad
            )
            hv_hat, log_var_v = self.gen_video_from_audio(
                src=a_seq, tgt_len=T_v, src_key_padding_mask=a_pad
            )

            log_var_a = log_var_a.clamp(-4.0, 4.0)
            log_var_v = log_var_v.clamp(-4.0, 4.0)

            rec_loss_a = self._nll_loss(
                ha.unsqueeze(1).expand_as(ha_hat), ha_hat, log_var_a
            )
            rec_loss_v = self._nll_loss(
                hv.unsqueeze(1).expand_as(hv_hat), hv_hat, log_var_v
            )

            sa = log_var_a.mean(dim=(1, 2))   # (B,)
            sv = log_var_v.mean(dim=(1, 2))   # (B,)

            ha_hat_pool = ha_hat.mean(dim=1)
            hv_hat_pool = hv_hat.mean(dim=1)

            ma = batch.get("ma", torch.ones(B, device=device))
            mv = batch.get("mv", torch.ones(B, device=device))

            h = self.fusion(ha, hv, ha_hat_pool, hv_hat_pool, sa, sv, ma, mv)

        # ── Case 2: audio missing — generate audio from video ─────────────
        elif audio_missing:
            v_seq, hv, v_pad = self._encode_video(video, video_lengths)

            ha_hat, log_var_a = self.gen_audio_from_video(
                src=v_seq, tgt_len=tgt_len_audio, src_key_padding_mask=v_pad
            )
            log_var_a = log_var_a.clamp(-4.0, 4.0)

            # No real audio → no reconstruction target
            rec_loss_a = torch.tensor(0.0, device=device)
            rec_loss_v = torch.tensor(0.0, device=device)

            # Generator uncertainty is naturally high for a missing modality;
            # the fusion module will down-weight it accordingly.
            sa = log_var_a.mean(dim=(1, 2))        # (B,)  high uncertainty expected
            sv = torch.zeros(B, device=device)     # real video → no uncertainty

            ha_pool = ha_hat.mean(dim=1)           # (B, d)  generated audio
            # ma=0 signals fusion to use generated slot; mv=1 keeps real video
            h = self.fusion(
                ha_pool, hv,      # ha (generated), hv (real)
                ha_pool, hv,      # ha_hat, hv_hat — identical to keep fusion signature
                sa, sv,
                torch.zeros(B, device=device),   # ma=0: audio absent
                torch.ones(B,  device=device),   # mv=1: video present
            )

        # ── Case 3: video missing — generate video from audio ─────────────
        else:
            a_seq, ha, a_pad = self._encode_audio(audio, audio_lengths)

            hv_hat, log_var_v = self.gen_video_from_audio(
                src=a_seq, tgt_len=tgt_len_video, src_key_padding_mask=a_pad
            )
            log_var_v = log_var_v.clamp(-4.0, 4.0)

            rec_loss_a = torch.tensor(0.0, device=device)
            rec_loss_v = torch.tensor(0.0, device=device)

            sa = torch.zeros(B, device=device)     # real audio → no uncertainty
            sv = log_var_v.mean(dim=(1, 2))        # (B,)  high uncertainty expected

            hv_pool = hv_hat.mean(dim=1)           # (B, d)  generated video
            # ma=1: real audio present; mv=0: video absent (use generated slot)
            h = self.fusion(
                ha, hv_pool,      # ha (real), hv (generated)
                ha, hv_pool,      # ha_hat, hv_hat — identical to keep fusion signature
                sa, sv,
                torch.ones(B,  device=device),   # ma=1: audio present
                torch.zeros(B, device=device),   # mv=0: video absent
            )

        # ── Detection ─────────────────────────────────────────────────────
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
        L2-normalises features so squared distances are bounded in [0,4].
        """
        target_n = F.normalize(target.detach(), dim=-1)
        pred_n   = F.normalize(pred,            dim=-1)
        diff     = (target_n - pred_n).pow(2)
        exp_neg  = (-log_var).exp().clamp(max=1e2)
        return 0.5 * (log_var + diff * exp_neg).mean()

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())