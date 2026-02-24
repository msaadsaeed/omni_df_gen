"""
backbones/audio_backbone.py
----------------------------
Audio backbone with two operating modes:

  mode="wav2vec2"        : Wav2Vec2 only
  mode="wav2vec2+hubert" : Wav2Vec2 + HuBERT concatenated  (default)

Both models process raw 16 kHz waveforms and produce frame-level hidden
states via a shared convolutional feature extractor + Transformer.

  Wav2Vec2 : raw acoustic features, strong on phonetics / artefact detection
  HuBERT   : discrete self-supervised targets, strong on prosody / rhythm

Output dim
----------
  wav2vec2 only         : D_w   (768 for *-base, 1024 for *-large)
  wav2vec2 + hubert     : D_w + D_h

Freezing strategy
-----------------
  freeze=True  (default)
      Freeze the convolutional feature extractors in both models;
      Transformer layers remain trainable.
      NOTE: HubertModel does NOT expose freeze_feature_encoder() — we
      freeze its feature_extractor sub-module directly.

  freeze=False
      All parameters are trainable. Use with a very small learning rate.

  freeze_all=True
      Freeze every parameter (pure frozen feature extractor mode).
      Useful when training only the projection / generator heads.

Gradual unfreezing
------------------
  Call unfreeze_top_k_transformer_layers(k) after initial warm-up epochs
  to progressively unfreeze the top-k Transformer layers of each encoder.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
from transformers import HubertModel, Wav2Vec2Model

# Supported mode strings
AudioMode = Literal["wav2vec2", "wav2vec2+hubert"]


class AudioBackbone(nn.Module):
    """
    Configurable audio backbone: Wav2Vec2-only or Wav2Vec2+HuBERT ensemble.

    Args:
        mode           : "wav2vec2"  or  "wav2vec2+hubert"
        wav2vec2_name  : HuggingFace id / local path for Wav2Vec2
        hubert_name    : HuggingFace id / local path for HuBERT
                         (ignored when mode="wav2vec2")
        freeze         : Freeze CNN feature extractors; keep Transformers trainable
        freeze_all     : Freeze ALL parameters (overrides freeze)
    """

    def __init__(
        self,
        mode:          AudioMode = "wav2vec2+hubert",
        wav2vec2_name: str       = "facebook/wav2vec2-base",
        hubert_name:   str       = "facebook/hubert-base-ls960",
        freeze:        bool      = True,
        freeze_all:    bool      = False,
    ) -> None:
        super().__init__()

        if mode not in ("wav2vec2", "wav2vec2+hubert"):
            raise ValueError(
                f"mode must be 'wav2vec2' or 'wav2vec2+hubert', got {mode!r}"
            )
        self.mode = mode

        # ── Wav2Vec2 (always loaded) ──────────────────────────────────────
        # HuggingFace from_pretrained() calls model.eval() internally.
        # We call .train() immediately after so Lightning does not warn
        # about "modules in eval mode at the start of training".
        # Freezing is done via requires_grad=False, NOT via .eval().
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(wav2vec2_name)
        self.wav2vec2.train()

        # ── HuBERT (only when mode includes it) ───────────────────────────
        self.hubert: Optional[HubertModel] = None
        if mode == "wav2vec2+hubert":
            self.hubert = HubertModel.from_pretrained(hubert_name)
            self.hubert.train()

        # ── Freezing ──────────────────────────────────────────────────────
        # requires_grad=False keeps weights frozen without affecting
        # train/eval mode (which controls dropout & batch-norm behaviour).
        if freeze_all:
            self._freeze_all()
        elif freeze:
            self._freeze_cnn_extractors()

        # ── Output dimension ──────────────────────────────────────────────
        self.out_dim: int = self.wav2vec2.config.hidden_size
        if self.hubert is not None:
            self.out_dim += self.hubert.config.hidden_size

    # ── Freeze helpers ────────────────────────────────────────────────────

    def _freeze_cnn_extractors(self) -> None:
        """
        Freeze only the convolutional feature extractor sub-modules.

        Wav2Vec2Model exposes freeze_feature_encoder() officially.
        HubertModel does NOT — we freeze its feature_extractor directly.
        Both approaches produce the same effect: CNN weights frozen,
        Transformer weights remain trainable.
        """
        # Wav2Vec2 — official API
        self.wav2vec2.freeze_feature_encoder()

        # HuBERT — freeze feature_extractor sub-module manually
        if self.hubert is not None:
            for p in self.hubert.feature_extractor.parameters():
                p.requires_grad_(False)
            # HuBERT also has a feature_projection layer after the CNN;
            # freeze that too for parity with Wav2Vec2's behaviour.
            for p in self.hubert.feature_projection.parameters():
                p.requires_grad_(False)

    def _freeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze_top_k_transformer_layers(self, k: int) -> None:
        """
        Unfreeze the top-k Transformer encoder layers in each model.
        Call this after the initial warm-up phase for gradual fine-tuning.

        Args:
            k : number of layers to unfreeze from the top (most recent)
        """
        models = [self.wav2vec2]
        if self.hubert is not None:
            models.append(self.hubert)

        for model in models:
            layers = model.encoder.layers
            for layer in layers[-k:]:
                for p in layer.parameters():
                    p.requires_grad_(True)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        waveform: torch.Tensor,
        lengths:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            waveform : (B, T)  raw waveform, float32, ~[-1, 1], 16 kHz
            lengths  : (B,)    valid sample counts (None = all valid)

        Returns:
            mode="wav2vec2"        : (B, T', D_w)
            mode="wav2vec2+hubert" : (B, T', D_w + D_h)
            where T' = T // 320   (conv stride of the base models)
        """
        attention_mask: Optional[torch.Tensor] = None
        if lengths is not None:
            attention_mask = (
                torch.arange(waveform.size(1), device=waveform.device)
                .unsqueeze(0)
                .lt(lengths.unsqueeze(1))
                .long()
            )   # (B, T)  1=valid, 0=padding

        w_out = self.wav2vec2(
            waveform, attention_mask=attention_mask
        ).last_hidden_state                          # (B, T', D_w)

        if self.hubert is None:
            return w_out

        h_out = self.hubert(
            waveform, attention_mask=attention_mask
        ).last_hidden_state                          # (B, T', D_h)

        # Both models share the same CNN stride so T' should match;
        # clamp to the minimum in case of a rare off-by-one.
        T = min(w_out.size(1), h_out.size(1))
        return torch.cat([w_out[:, :T], h_out[:, :T]], dim=-1)   # (B, T', D_w+D_h)

    # ── Info ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return (
            f"{self.__class__.__name__}("
            f"mode={self.mode!r}, "
            f"out_dim={self.out_dim}, "
            f"trainable={trainable:,}/{total:,})"
        )