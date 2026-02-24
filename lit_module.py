"""
lit_module.py
──────────────
PyTorch Lightning Module wrapping OmniDeepfakeModel.

Training strategy
─────────────────
  Phase 1  (warm-up, default first ~10% of steps):
    Backbone parameters frozen; only projection heads, generators,
    fusion, and detection head are trained.

  Phase 2  (gradual unfreeze):
    Periodically unfreeze the top-k transformer layers in both
    audio and video backbones (configurable via unfreeze_top_k_at_epoch).

Optimiser
─────────
  AdamW with weight decay.
  Separate parameter groups:
    - backbone params : lower LR  (backbone_lr_scale × base_lr)
    - all other params: base_lr

  OneCycleLR scheduler with cosine annealing.

Logging
───────
  All loss components and accuracy/AUC metrics are logged to the
  Lightning logger (TensorBoard / W&B depending on Trainer config).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from models.omni_model import OmniDeepfakeModel
from losses.losses import OmniLoss
from utils.metrics import accuracy, compute_auc, EvalAccumulator


class OmniDeepfakeLitModel(pl.LightningModule):
    """
    Lightning module for OmniDeepfakeModel.

    Args:
        d                      : shared projection dimension
        lr                     : base learning rate
        weight_decay           : AdamW weight decay
        backbone_lr_scale      : backbone LR = backbone_lr_scale * lr
        lambda_audio/video/av  : classification loss weights
        lambda_rec             : reconstruction NLL weight
        label_smoothing        : cross-entropy label smoothing
        unfreeze_top_k_layers  : unfreeze top-k backbone transformer layers
                                 when this epoch is reached (None = never)
        unfreeze_at_epoch      : epoch at which to call unfreeze
        freeze_backbones       : start with frozen backbones
        model_kwargs           : forwarded to OmniDeepfakeModel.__init__
    """

    def __init__(
        self,
        d:                  int   = 256,
        lr:                 float = 2e-4,
        weight_decay:       float = 1e-4,
        backbone_lr_scale:  float = 0.1,
        lambda_audio:       float = 0.3,
        lambda_video:       float = 0.3,
        lambda_av:          float = 1.0,
        lambda_rec:         float = 0.1,
        label_smoothing:    float = 0.05,
        unfreeze_top_k_layers: Optional[int] = 2,
        unfreeze_at_epoch:  int   = 5,
        freeze_backbones:   bool  = True,
        **model_kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = OmniDeepfakeModel(
            d=d,
            freeze_backbones=freeze_backbones,
            **model_kwargs,
        )
        self.criterion = OmniLoss(
            lambda_audio=lambda_audio,
            lambda_video=lambda_video,
            lambda_av=lambda_av,
            lambda_rec=lambda_rec,
            label_smoothing=label_smoothing,
        )

        self.lr                   = lr
        self.weight_decay         = weight_decay
        self.backbone_lr_scale    = backbone_lr_scale
        self.unfreeze_top_k_layers = unfreeze_top_k_layers
        self.unfreeze_at_epoch    = unfreeze_at_epoch

        # Epoch-level accumulators for AUC computation
        self._val_acc  = EvalAccumulator()
        self._test_acc = EvalAccumulator()

    # ── Lifecycle callbacks ───────────────────────────────────────────────

    def on_train_epoch_start(self) -> None:
        """Gradually unfreeze backbone layers at the configured epoch."""
        if (
            self.unfreeze_top_k_layers is not None
            and self.current_epoch == self.unfreeze_at_epoch
        ):
            self.model.audio_backbone.unfreeze_top_k_transformer_layers(
                self.unfreeze_top_k_layers
            )
            self.model.video_backbone.unfreeze_clip_top_k_layers(
                self.unfreeze_top_k_layers
            )
            self.log(
                "unfreeze/top_k_layers",
                float(self.unfreeze_top_k_layers),
                on_step=False, on_epoch=True,
            )

    # ── Core step ─────────────────────────────────────────────────────────

    def _step(self, batch: Dict, stage: str) -> torch.Tensor:
        outputs = self.model(batch)
        losses  = self.criterion(outputs, batch)

        # Log all loss components
        on_step = stage == "train"

        self.log(f"{stage}/rec_a", outputs["rec_loss_a"],
                 on_step=on_step, on_epoch=True,
                 batch_size=batch["audio"].size(0))
        self.log(f"{stage}/rec_v", outputs["rec_loss_v"],
                 on_step=on_step, on_epoch=True,
                 batch_size=batch["audio"].size(0))

        self.log_dict(
            {f"{stage}/{k}": v for k, v in losses.items()},
            on_step=on_step, on_epoch=True, prog_bar=(stage == "train"),
            batch_size=batch["audio"].size(0),
        )

        # Log step-level accuracy (lightweight, no AUC)
        for task, logits_key, label_key in [
            ("audio", "logits_audio", "audio_label"),
            ("video", "logits_video", "video_label"),
            ("av",    "logits_av",    "av_label"),
        ]:
            acc = accuracy(outputs[logits_key], batch[label_key])
            self.log(
                f"{stage}/acc_{task}", acc,
                on_step=on_step, on_epoch=True, prog_bar=True,
                batch_size=batch["audio"].size(0),
            )

        return losses["total"]

    # ── Train / val / test ────────────────────────────────────────────────

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        outputs = self.model(batch)
        self._step(batch, "val")
        self._val_acc.update(
            logits_audio=outputs["logits_audio"],
            labels_audio=batch["audio_label"],
            logits_video=outputs["logits_video"],
            labels_video=batch["video_label"],
            logits_av=outputs["logits_av"],
            labels_av=batch["av_label"],
        )

    def on_validation_epoch_end(self) -> None:
        metrics = self._val_acc.compute()
        self.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            on_step=False, on_epoch=True,
        )
        self._val_acc.reset()

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        outputs = self.model(batch)
        self._step(batch, "test")
        self._test_acc.update(
            logits_audio=outputs["logits_audio"],
            labels_audio=batch["audio_label"],
            logits_video=outputs["logits_video"],
            labels_video=batch["video_label"],
            logits_av=outputs["logits_av"],
            labels_av=batch["av_label"],
        )

    def on_test_epoch_end(self) -> None:
        metrics = self._test_acc.compute()
        self.log_dict(
            {f"test/{k}": v for k, v in metrics.items()},
            on_step=False, on_epoch=True,
        )
        self._test_acc.reset()
        # Print a summary table to stdout
        print("\n── Test Results ─────────────────────────────")
        for k, v in sorted(metrics.items()):
            print(f"  {k:<30} {v:.4f}")
        print("─────────────────────────────────────────────\n")

    # ── Inference ─────────────────────────────────────────────────────────

    def predict_step(self, batch: Dict, batch_idx: int) -> Dict:
        """
        Returns softmax probabilities for all three task heads.
        To run single-modality inference:
            batch["ma"] = torch.zeros(B)   # disable audio
            batch["mv"] = torch.ones(B)    # keep video
        """
        outputs = self.model(batch)
        return {
            "prob_audio": F.softmax(outputs["logits_audio"], dim=-1),
            "prob_video": F.softmax(outputs["logits_video"], dim=-1),
            "prob_av":    F.softmax(outputs["logits_av"],    dim=-1),
        }

    # ── Optimiser ─────────────────────────────────────────────────────────

    def configure_optimizers(self):
        backbone_params = (
            list(self.model.audio_backbone.parameters())
            + list(self.model.video_backbone.parameters())
        )
        backbone_ids = {id(p) for p in backbone_params}

        other_params = [
            p for p in self.model.parameters()
            if p.requires_grad and id(p) not in backbone_ids
        ]
        backbone_trainable = [p for p in backbone_params if p.requires_grad]

        param_groups: List[Dict] = [
            {"params": other_params,       "lr": self.lr},
        ]
        if backbone_trainable:
            param_groups.append(
                {"params": backbone_trainable, "lr": self.lr * self.backbone_lr_scale}
            )

        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in param_groups],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.05,
            anneal_strategy="cos",
        )
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # ── Utilities ─────────────────────────────────────────────────────────

    def print_param_summary(self) -> None:
        total     = self.model.num_total_parameters()
        trainable = self.model.num_trainable_parameters()
        print(
            f"Parameters: {total:,} total | "
            f"{trainable:,} trainable ({100*trainable/total:.1f}%)"
        )
