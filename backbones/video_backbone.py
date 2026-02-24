"""
backbones/video_backbone.py
────────────────────────────
Dual video backbone: CLIP ViT + FaceNet (InceptionResnetV1).

  CLIP ViT  : semantic visual features from the CLS token
              strong on scene context and visual manipulation artefacts
  FaceNet   : identity / facial geometry embedding (512-d)
              strong on face-swap and identity inconsistencies

The two embeddings are concatenated per frame, giving downstream modules
a combined view of semantic content AND identity.

Input  : (B, T, 1, H, W)  grayscale face crops float32 [-1, 1]
Output : (B, T, D_c + D_f) per-frame concatenated features
         out_dim property   → int  (D_c + D_f)

Preprocessing (handled internally)
────────────────────────────────────
  Grayscale → RGB  (repeat channel dim)
  CLIP   needs 224×224 with ImageNet-style normalisation  (mean/std)
  FaceNet needs 160×160 in [-1, 1]

Freezing strategy
─────────────────
  freeze=True  (default)
      Freeze all backbone parameters; only projection layers are trained.
  freeze=False
      All parameters trainable (use with small LR).

Dependencies
────────────
  pip install transformers facenet-pytorch
"""

from __future__ import annotations
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel

VideoMode = Literal["clip", "clip+facenet"]


class VideoBackbone(nn.Module):
    """
    Configurable video backbone: CLIP-only or CLIP+FaceNet ensemble.

    Args:
        mode               : "clip"  or  "clip+facenet"
        clip_name          : HuggingFace model name for CLIP vision encoder
        facenet_pretrained : 'vggface2' or 'casia-webface'
                             (ignored when mode="clip")
        freeze             : Freeze all backbone parameters
    """

    def __init__(
        self,
        mode:               VideoMode = "clip",
        clip_name:          str       = "openai/clip-vit-base-patch32",
        facenet_pretrained: str       = "vggface2",
        freeze:             bool      = True,
    ) -> None:
        super().__init__()
        if mode not in ("clip", "clip+facenet"):
            raise ValueError(f"mode must be 'clip' or 'clip+facenet', got {mode!r}")
        self.mode = mode

        # ── CLIP ViT ──────────────────────────────────────────────────────
        self.clip     = CLIPVisionModel.from_pretrained(clip_name)
        self.clip_dim = self.clip.config.hidden_size   # 768 for base-patch32

        # ── FaceNet ───────────────────────────────────────────────────────
        self.facenet:     Optional[nn.Module] = None
        self.facenet_dim: int                 = 0
        if mode == "clip+facenet":
            try:
                from facenet_pytorch import InceptionResnetV1
            except ImportError as exc:
                raise ImportError(
                    "facenet-pytorch is required: pip install facenet-pytorch"
                ) from exc
            self.facenet     = InceptionResnetV1(pretrained=facenet_pretrained).train()
            self.facenet_dim = 512

        if freeze:
            self._freeze_all()

        self.out_dim: int = self.clip_dim + self.facenet_dim

        # Normalisation constants for CLIP (ImageNet-like)
        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )

    # ── Freeze helpers ────────────────────────────────────────────────────

    def _freeze_all(self) -> None:
        for p in self.clip.parameters():
            p.requires_grad_(False)
        if self.facenet is not None:
            for p in self.facenet.parameters():
                p.requires_grad_(False)

    def unfreeze_clip_top_k_layers(self, k: int) -> None:
        """
        Unfreeze the top-k vision transformer layers in CLIP for gradual
        fine-tuning after initial warm-up epochs.
        """
        layers = self.clip.vision_model.encoder.layers
        for layer in layers[-k:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        # Always unfreeze the final layer norm and projection
        for p in self.clip.vision_model.post_layernorm.parameters():
            p.requires_grad_(True)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video : (B, T, 1, H, W)  grayscale face crops float32 [-1, 1]

        Returns:
            (B, T, D_c + D_f)  per-frame concatenated features
        """
        B, T, C, H, W = video.shape

        # Flatten temporal dim for batched backbone processing
        frames = video.view(B * T, C, H, W)        # (B*T, 1, H, W)
        rgb    = frames.expand(-1, 3, -1, -1)       # (B*T, 3, H, W) — no copy

        # ── CLIP (expects 224×224, ImageNet normalisation) ─────────────────
        clip_in = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
        clip_in = (clip_in + 1.0) / 2.0                         # [-1,1] → [0,1]
        clip_in = (clip_in - self.clip_mean) / self.clip_std     # normalise
        clip_cls = self.clip(pixel_values=clip_in).last_hidden_state[:, 0, :]
        # → (B*T, D_c)   CLS token

        if self.facenet is None:
            return clip_cls.view(B, T, -1)
        
        # ── FaceNet (expects 160×160, range [-1, 1]) ───────────────────────
        fn_in  = F.interpolate(rgb, size=(160, 160), mode="bilinear", align_corners=False)
        # rgb was derived from [-1,1] expand; FaceNet expects [-1,1] → no extra norm needed
        fn_out = self.facenet(fn_in)                             # (B*T, 512)

        # ── Combine ───────────────────────────────────────────────────────
        combined = torch.cat([clip_cls, fn_out], dim=-1)         # (B*T, D_c+D_f)
        return combined.view(B, T, -1)                           # (B, T, D_c+D_f)
