"""
train.py
--------
Training entry point for OmniDeepfake.

Usage
-----
  # Basic run (all frames per clip)
  python train.py --root_dir /data/fakeavceleb_processed

  # Sample 16 frames per clip (fixed-length, faster batches)
  python train.py --root_dir /data/fakeavceleb_processed --n_frames 16

  # Full custom run
  python train.py \\
      --root_dir      /data/fakeavceleb_processed \\
      --n_frames      16                          \\
      --d             256                         \\
      --batch_size    16                          \\
      --max_epochs    30                          \\
      --lr            2e-4                        \\
      --gpus          1                           \\
      --precision     16-mixed                    \\
      --exp_name      omni_run_01

  # Resume from a checkpoint
  python train.py --root_dir /data/fakeavceleb_processed \\
      --resume checkpoints/last.ckpt
"""

from __future__ import annotations

import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from data.datamodule import FakeAVCelebDataModule
from lit_module import OmniDeepfakeLitModel


def _parse_n_frames(value: str):
    """Accept 'all' or a positive integer string."""
    if value.lower() == "all":
        return "all"
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"n_frames must be 'all' or a positive integer, got: {value!r}"
        )
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"n_frames must be >= 1, got: {n}"
        )
    return n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train OmniDeepfake")

    # Data
    p.add_argument("--root_dir",       type=str, required=True,
                   help="Dataset root (must contain train/ val/ test/ subdirs)")
    p.add_argument("--train_csv",      type=str, default="train/metadata.csv")
    p.add_argument("--val_csv",        type=str, default="val/metadata.csv")
    p.add_argument("--face_size",      type=int,             default=224,
                   help="Resize face crops to this size on load (default 224)")
    p.add_argument("--n_frames",       type=_parse_n_frames, default="all",
                   help="Frames per clip: 'all' or a positive int (e.g. 16)")
    p.add_argument("--max_audio_secs", type=float, default=16.0)
    p.add_argument("--batch_size",     type=int,   default=16)
    p.add_argument("--num_workers",    type=int,   default=8)
    p.add_argument("--no_augment",     action="store_true")

    # Model
    p.add_argument("--d",                  type=int,   default=256)
    p.add_argument("--max_tgt_len",        type=int,   default=128)
    p.add_argument("--gen_nhead",          type=int,   default=4)
    p.add_argument("--gen_enc_layers",     type=int,   default=2)
    p.add_argument("--gen_dec_layers",     type=int,   default=2)
    p.add_argument("--head_hidden",        type=int,   default=512)
    p.add_argument("--head_dropout",       type=float, default=0.3)
    p.add_argument("--audio_mode",         type=str,   default="wav2vec2+hubert",
                   choices=["wav2vec2", "wav2vec2+hubert"],
                   help="Use Wav2Vec2 only or Wav2Vec2+HuBERT ensemble")
    p.add_argument("--wav2vec2_name",      type=str,   default="facebook/wav2vec2-base")
    p.add_argument("--hubert_name",        type=str,   default="facebook/hubert-base-ls960")
    p.add_argument("--clip_name",          type=str,   default="openai/clip-vit-base-patch32")
    p.add_argument("--facenet_pretrained", type=str,   default="vggface2")
    p.add_argument("--no_freeze",          action="store_true",
                   help="Disable backbone freezing (not recommended from scratch)")
    p.add_argument("--unfreeze_top_k",     type=int,   default=2,
                   help="Unfreeze top-k backbone layers at --unfreeze_at_epoch")
    p.add_argument("--unfreeze_at_epoch",  type=int,   default=5)

    # Loss
    p.add_argument("--lambda_audio",    type=float, default=0.3)
    p.add_argument("--lambda_video",    type=float, default=0.3)
    p.add_argument("--lambda_av",       type=float, default=1.0)
    p.add_argument("--lambda_rec",      type=float, default=0.1)
    p.add_argument("--label_smoothing", type=float, default=0.05)

    # Optimiser
    p.add_argument("--lr",               type=float, default=2e-4)
    p.add_argument("--weight_decay",     type=float, default=1e-4)
    p.add_argument("--backbone_lr_scale",type=float, default=0.1)

    # Trainer
    p.add_argument("--max_epochs",          type=int,   default=30)
    p.add_argument("--gpus",               type=int,   default=1)
    p.add_argument("--precision",          type=str,   default="16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--grad_clip",          type=float, default=1.0)
    p.add_argument("--val_every_n_epochs", type=int,   default=1)

    # Logging / checkpoints
    p.add_argument("--log_dir",   type=str, default="./logs")
    p.add_argument("--exp_name",  type=str, default="omni_deepfake")
    p.add_argument("--ckpt_dir",  type=str, default="./checkpoints")
    p.add_argument("--resume",    type=str, default=None,
                   help="Path to checkpoint to resume from")

    # Misc
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    import warnings
    # Suppress harmless warnings from sklearn (AUC with 1 class)
    # and HuggingFace tokenisers parallelism noise.
    warnings.filterwarnings("ignore", category=UserWarning,
                            module="sklearn")
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    # DataModule
    dm = FakeAVCelebDataModule(
        root_dir       = args.root_dir,
        train_csv      = args.train_csv,
        val_csv        = args.val_csv,
        face_size      = args.face_size,
        n_frames       = args.n_frames,
        max_audio_secs = args.max_audio_secs,
        batch_size     = args.batch_size,
        num_workers    = args.num_workers,
        augment        = not args.no_augment,
    )

    # Model
    lit = OmniDeepfakeLitModel(
        d                     = args.d,
        lr                    = args.lr,
        weight_decay          = args.weight_decay,
        backbone_lr_scale     = args.backbone_lr_scale,
        lambda_audio          = args.lambda_audio,
        lambda_video          = args.lambda_video,
        lambda_av             = args.lambda_av,
        lambda_rec            = args.lambda_rec,
        label_smoothing       = args.label_smoothing,
        unfreeze_top_k_layers = args.unfreeze_top_k,
        unfreeze_at_epoch     = args.unfreeze_at_epoch,
        freeze_backbones      = not args.no_freeze,
        max_tgt_len           = args.max_tgt_len,
        gen_nhead             = args.gen_nhead,
        gen_enc_layers        = args.gen_enc_layers,
        gen_dec_layers        = args.gen_dec_layers,
        head_hidden           = args.head_hidden,
        head_dropout          = args.head_dropout,
        audio_mode            = args.audio_mode,
        wav2vec2_name         = args.wav2vec2_name,
        hubert_name           = args.hubert_name,
        clip_name             = args.clip_name,
        facenet_pretrained    = args.facenet_pretrained,
    )
    lit.print_param_summary()

    # Callbacks
    os.makedirs(args.ckpt_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath    = args.ckpt_dir,
        filename   = "omni-{epoch:02d}-{val/acc_av:.4f}",
        monitor    = "val/acc_av",
        mode       = "max",
        save_top_k = 3,
        save_last  = True,
    )
    early_stop_cb = EarlyStopping(
        monitor  = "val/acc_av",
        mode     = "max",
        patience = 5,
        verbose  = True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    progress   = RichProgressBar()

    # Logger
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.exp_name)

    # Trainer
    trainer = pl.Trainer(
        max_epochs              = args.max_epochs,
        accelerator             = "gpu" if args.gpus > 0 else "cpu",
        devices                 = args.gpus if args.gpus > 0 else "auto",
        precision               = args.precision,
        gradient_clip_val       = args.grad_clip,
        check_val_every_n_epoch = args.val_every_n_epochs,
        callbacks               = [checkpoint_cb, early_stop_cb, lr_monitor, progress],
        logger                  = logger,
        log_every_n_steps       = 10,
        # Disable Lightning's FLOPs profiler — it cannot trace through
        # HuggingFace models and always reports "Total FLOPs: 0".
        enable_model_summary    = True,   # keep param count summary
    )

    trainer.fit(lit, datamodule=dm, ckpt_path=args.resume)

    print(f"\nBest checkpoint : {checkpoint_cb.best_model_path}")
    print(f"Best val/acc_av : {checkpoint_cb.best_model_score:.4f}")

if __name__ == "__main__":
    main()
