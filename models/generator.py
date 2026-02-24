"""
models/generator.py
────────────────────
Bidirectional cross-modal feature generator.

Two instances are created in the main model:
  Ga←v  : generates audio features from video (gen_audio_from_video)
  Gv←a  : generates video features from audio (gen_video_from_audio)

Architecture
────────────
  Transformer Encoder  — contextualises the source-modality token sequence
  Transformer Decoder  — cross-attends to the encoded source via N learned
                         query tokens (one per target time-step, DETR-style)
  Mean head            — linear → predicted target features  μ̂ ∈ ℝ^d
  Log-variance head    — linear → log σ² ∈ ℝ^d (token-level uncertainty)

Why learned queries instead of the source sequence as decoder input?
  → Decouples source and target sequence lengths, making the generator
    applicable regardless of the T_audio / T_video ratio.
  → Forces the decoder to attend globally to the source context rather
    than relying on positional alignment.

Uncertainty parameterisation
────────────────────────────
  s = log σ²    (log-variance)
  σ = exp(s/2)

  The log-variance is used in two ways:
    1. Reconstruction loss: Gaussian NLL  (see losses/losses.py)
    2. Reliability weighting in fusion:   w = exp(−α · mean(s))
       → high uncertainty ⟹ high s ⟹ low weight

Input:
  src : (B, T_src, d)    source-modality projected features
  tgt_len : int          number of target tokens to generate

Output:
  feat    : (B, tgt_len, d)  generated features  (μ̂)
  log_var : (B, tgt_len, d)  log-variance        (s = log σ²)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class CrossModalGenerator(nn.Module):
    """
    Transformer encoder-decoder cross-modal generator.

    Args:
        d_model        : shared feature dimension (must match projection heads)
        nhead          : number of attention heads
        num_enc_layers : depth of the source encoder
        num_dec_layers : depth of the target decoder
        dim_feedforward: FFN intermediate dimension
        dropout        : attention & FFN dropout
        max_tgt_len    : maximum number of target query tokens (pre-allocated)
                         Queries beyond tgt_len are simply sliced off.
    """

    def __init__(
        self,
        d_model:        int   = 256,
        nhead:          int   = 4,
        num_enc_layers: int   = 2,
        num_dec_layers: int   = 2,
        dim_feedforward: int  = 512,
        dropout:        float = 0.1,
        max_tgt_len:    int   = 512,
    ) -> None:
        super().__init__()

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True,   # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_enc_layers,
            norm=nn.LayerNorm(d_model),
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=num_dec_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Learned positional query tokens (DETR-style)
        self.tgt_queries = nn.Parameter(
            torch.randn(1, max_tgt_len, d_model) * 0.02
        )

        # Output heads
        self.feat_head    = nn.Linear(d_model, d_model)
        self.logvar_head  = nn.Linear(d_model, d_model)

    def forward(
        self,
        src:                    torch.Tensor,
        tgt_len:                int,
        src_key_padding_mask:   Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src                  : (B, T_src, d)  source features
            tgt_len              : number of target tokens to generate
            src_key_padding_mask : (B, T_src) bool, True = padding position

        Returns:
            feat    : (B, tgt_len, d)  generated feature means
            log_var : (B, tgt_len, d)  log-variance per token
        """
        B = src.size(0)

        # Encode source sequence
        memory  = self.encoder(
            src, src_key_padding_mask=src_key_padding_mask
        )   # (B, T_src, d)

        # Expand learned queries to batch
        queries = self.tgt_queries[:, :tgt_len, :].expand(B, -1, -1)

        # Decode: queries cross-attend to the encoded source
        decoded = self.decoder(
            queries, memory,
            memory_key_padding_mask=src_key_padding_mask,
        )   # (B, tgt_len, d)

        feat    = self.feat_head(decoded)                   # (B, tgt_len, d)  μ̂
        log_var = self.logvar_head(decoded).clamp(-4., 4.)  # (B, tgt_len, d)  s = log σ²
        # With L2-normalised targets, squared diffs are in [0, 4].
        # Clamping log_var to [-4, 4] keeps σ in [0.14, 7.4] — a sensible
        # uncertainty range that prevents both explosion and collapse.

        return feat, log_var