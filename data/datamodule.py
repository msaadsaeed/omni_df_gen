"""
data/datamodule.py
──────────────────
PyTorch Lightning DataModule for pre-processed FakeAVCeleb.

Expected directory layout
──────────────────────────
    <root>/
        train/metadata.csv   ← produced by prepare_fakeav_splits.py
        val/metadata.csv
        test/metadata.csv
        RealVideo-RealAudio/
            African/men/id00076/
                00109.wav
                00109/
                    000000.png
                    000001.png
                    …
        FakeVideo-FakeAudio/
            …

All faces_path and audio_path values in the CSVs are relative to <root>.

n_frames parameter
──────────────────
    "all"  → every available PNG frame per clip (variable-length batches,
              handled by the collate_fn padding)
    int N  → uniformly subsample N frames per clip (fixed-length batches
              assuming all clips have ≥ N frames; collate_fn still pads
              clips shorter than N)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import FakeAVCelebDataset, NFrames, collate_fn


class FakeAVCelebDataModule(pl.LightningDataModule):
    """
    DataModule that reads pre-extracted frames and audio WAVs.
    No face detection at runtime.

    Args:
        root_dir       : dataset root directory (contains train/ val/ test/)
        n_frames       : frame sampling policy per clip
                           "all"  → every available frame
                           int N  → uniformly sample N frames
        max_audio_secs : truncate audio to at most this many seconds
        batch_size     : samples per batch
        num_workers    : DataLoader worker processes
        augment        : enable augmentations on the training set
        train_csv      : metadata CSV path relative to root_dir
                         (default: train/metadata.csv)
        val_csv        : (default: val/metadata.csv)
        test_csv       : (default: test/metadata.csv)
    """

    def __init__(
        self,
        root_dir:       Union[str, Path],
        n_frames:       NFrames = "all",
        face_size:      int     = 224,
        max_audio_secs: float   = 16.0,
        batch_size:     int     = 16,
        num_workers:    int     = 8,
        augment:        bool    = True,
        train_csv:      str     = "train/metadata.csv",
        val_csv:        str     = "val/metadata.csv",
        test_csv:       str     = "test/metadata.csv",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.root_dir       = Path(root_dir)
        self.n_frames       = n_frames
        self.face_size      = face_size
        self.max_audio_secs = max_audio_secs
        self.batch_size     = batch_size
        self.num_workers    = num_workers
        self.augment        = augment
        self.train_csv      = train_csv
        self.val_csv        = val_csv
        self.test_csv       = test_csv

    # ── Internal builder ─────────────────────────────────────────────────

    def _build(self, csv_rel: str, augment: bool = False) -> FakeAVCelebDataset:
        return FakeAVCelebDataset(
            metadata_csv   = self.root_dir / csv_rel,
            root_dir       = self.root_dir,
            n_frames       = self.n_frames,
            face_size      = self.face_size,
            max_audio_secs = self.max_audio_secs,
            augment        = augment,
        )

    # ── Setup ────────────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self.train_ds = self._build(self.train_csv, augment=self.augment)
            self.val_ds   = self._build(self.val_csv,   augment=False)
        if stage in ("test", None):
            self.test_ds  = self._build(self.test_csv,  augment=False)

    # ── DataLoaders ──────────────────────────────────────────────────────

    def _loader(self, dataset: FakeAVCelebDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size         = self.batch_size,
            shuffle            = shuffle,
            num_workers        = self.num_workers,
            collate_fn         = collate_fn,
            pin_memory         = True,
            drop_last          = shuffle,       # drop only for train
            persistent_workers = self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds,   shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds,  shuffle=False)

    # ── Info ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"root={self.root_dir}, "
            f"n_frames={self.n_frames!r}, "
            f"batch_size={self.batch_size}, "
            f"num_workers={self.num_workers})"
        )