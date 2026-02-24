"""
data/dataset.py
───────────────
FakeAVCeleb Dataset — pre-processed version.

Reads pre-extracted face frames (PNG) and audio (WAV) files directly.
No face detection is performed at runtime.

Metadata CSV columns (produced by prepare_fakeav_splits.py):
    av_class    : e.g. "RealAudio-RealVideo"
    rel_parent  : relative path to the clip folder from dataset root
    stem        : clip stem  e.g. "id00076_00109_000000"
    faces_path  : relative path to the frames folder
                  e.g. "RealVideo-RealAudio/African/men/id00076/00109"
    audio_path  : relative path to the .wav file
                  e.g. "RealVideo-RealAudio/African/men/id00076/00109.wav"
    num_frames  : total number of extracted PNG frames
    av_label    : int 0-3
    audio_label : int 0/1  (is audio fake?)
    video_label : int 0/1  (is video fake?)

Disk layout
───────────
    <root>/<audio_path>                → .wav file
    <root>/<faces_path>/<id>.png       → face crop PNGs (e.g. 000000.png …)

Frame sampling  (n_frames parameter)
────────────────────────────────────
    "all"   → load every available PNG in sorted order
    int N   → uniformly subsample exactly N frames (temporal coverage preserved)
              if N >= num available frames, all frames are returned

Labels  (all three returned every __getitem__)
──────────────────────────────────────────────
    av_label    4-class : RARV=0, FAFV=1, RAFV=2, FARV=3
    audio_label binary  : 1 if audio is fake, else 0
    video_label binary  : 1 if video is fake, else 0
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
import torchaudio
from PIL import Image
from torch.utils.data import Dataset

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_SR   = 16_000
MAX_SECONDS = 16.0

NFrames = Union[int, str]   # type alias: positive int or "all"

NUM_CLASSES = {"audio": 2, "video": 2, "audiovisual": 4}
CLASS_NAMES = {
    "audio":       ["RealAudio", "FakeAudio"],
    "video":       ["RealVideo", "FakeVideo"],
    "audiovisual": ["RARV", "FAFV", "RAFV", "FARV"],
}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _sorted_pngs(faces_dir: Path) -> List[Path]:
    """All *.png files in faces_dir, sorted numerically by stem."""
    paths = sorted(faces_dir.glob("*.png"), key=lambda p: p.stem)
    if not paths:
        raise FileNotFoundError(f"No PNG frames found in: {faces_dir}")
    return paths


def _select_frames(paths: List[Path], n_frames: NFrames) -> List[Path]:
    """
    Apply the n_frames sampling policy.

    Args:
        paths    : sorted list of all available PNGs
        n_frames : "all"  → return all paths unchanged
                   int N  → uniformly subsample N paths
                            (returns all if N >= len(paths))
    """
    if n_frames == "all":
        return paths

    if not isinstance(n_frames, int) or n_frames < 1:
        raise ValueError(
            f"n_frames must be a positive int or 'all', got {n_frames!r}"
        )

    T = len(paths)
    if n_frames >= T:
        return paths

    indices = torch.linspace(0, T - 1, n_frames).long().tolist()
    return [paths[i] for i in indices]


FACE_SIZE = 224   # all frames are resized to this spatial resolution on load


def load_frames(
    faces_dir: Path,
    n_frames:  NFrames = "all",
    face_size: int     = FACE_SIZE,
) -> torch.Tensor:
    """
    Load pre-extracted face-crop PNGs from faces_dir.

    Frames are resized to (face_size x face_size) on load so that every
    sample in a batch has identical spatial dimensions regardless of the
    resolution at which the crops were originally saved.

    Args:
        faces_dir : directory containing *.png face crops
        n_frames  : "all" or int N  (see _select_frames)
        face_size : output H and W after resize (default 224)

    Returns:
        (T, 1, face_size, face_size)  float32  in [-1, 1]
    """
    selected = _select_frames(_sorted_pngs(faces_dir), n_frames)

    frames: List[np.ndarray] = []
    for p in selected:
        img = Image.open(p).convert("L")            # grayscale (H, W)
        if img.size != (face_size, face_size):       # resize only when needed
            img = img.resize((face_size, face_size), Image.BILINEAR)
        frames.append(np.array(img, dtype=np.uint8))

    arr = np.stack(frames)                           # (T, H, W)  uint8
    t   = torch.from_numpy(arr[:, None]).float()     # (T, 1, H, W)
    return t / 127.5 - 1.0                          # [-1, 1]


def load_audio(
    audio_path: Path,
    target_sr:  int   = TARGET_SR,
    max_secs:   float = MAX_SECONDS,
) -> torch.Tensor:
    """
    Load a pre-extracted WAV file, resample if needed, convert to mono.

    Returns:
        (T_audio,)  float32
    """
    waveform, sr = torchaudio.load(str(audio_path))

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    if waveform.shape[0] > 1:                  # stereo → mono
        waveform = waveform.mean(dim=0, keepdim=True)

    max_samples = int(max_secs * target_sr)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]

    return waveform.squeeze(0)                  # (T_audio,)


# ── Dataset ───────────────────────────────────────────────────────────────────

class FakeAVCelebDataset(Dataset):
    """
    FakeAVCeleb dataset over pre-extracted face frames and audio WAVs.
    No face detection is performed at runtime.

    Args:
        metadata_csv   : path to a split metadata CSV
                         e.g. fakeavceleb_processed/train/metadata.csv
        root_dir       : dataset root; all faces_path / audio_path values
                         in the CSV are relative to this directory
        n_frames       : frame sampling policy
                           "all"  → load every available PNG
                           int N  → uniformly sample N frames per clip
        max_audio_secs : truncate audio to at most this many seconds
        augment        : apply light augmentations (set True only for train)
    """

    def __init__(
        self,
        metadata_csv:   Union[str, Path],
        root_dir:       Union[str, Path],
        n_frames:       NFrames = "all",
        face_size:      int     = FACE_SIZE,
        max_audio_secs: float   = MAX_SECONDS,
        augment:        bool    = False,
    ) -> None:
        self.root_dir       = Path(root_dir)
        self.n_frames       = n_frames
        self.face_size      = face_size
        self.max_audio_secs = max_audio_secs
        self.augment        = augment

        # validate n_frames early to catch typos before the first worker crash
        if n_frames != "all" and (not isinstance(n_frames, int) or n_frames < 1):
            raise ValueError(
                f"n_frames must be a positive int or 'all', got {n_frames!r}"
            )

        df = pd.read_csv(metadata_csv, sep=None, engine="python")
        if df.empty:
            raise RuntimeError(f"Empty metadata CSV: {metadata_csv}")

        self.samples: List[Dict] = df.to_dict(orient="records")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        row = self.samples[idx]

        faces_dir  = self.root_dir / str(row["faces_path"])
        audio_file = self.root_dir / str(row["audio_path"])

        frames   = load_frames(faces_dir,  n_frames=self.n_frames, face_size=self.face_size)
        waveform = load_audio(audio_file,  max_secs=self.max_audio_secs)

        if self.augment:
            waveform = self._augment_audio(waveform)
            frames   = self._augment_video(frames)

        return {
            "audio":       waveform,                                            # (T_a,)
            "video":       frames,                                              # (T, 1, H, W)
            "audio_label": torch.tensor(int(row["audio_label"]), dtype=torch.long),
            "video_label": torch.tensor(int(row["video_label"]), dtype=torch.long),
            "av_label":    torch.tensor(int(row["av_label"]),    dtype=torch.long),
            # passthrough strings — useful for per-sample error analysis
            "stem":        str(row["stem"]),
            "av_class":    str(row["av_class"]),
        }

    # ── Augmentations ────────────────────────────────────────────────────

    @staticmethod
    def _augment_audio(waveform: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            waveform = waveform + 0.005 * torch.randn_like(waveform)
        if random.random() < 0.5:
            waveform = waveform * random.uniform(0.8, 1.2)
        return waveform.clamp(-1.0, 1.0)

    @staticmethod
    def _augment_video(frames: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            frames = torch.flip(frames, dims=[-1])              # horizontal flip
        if random.random() < 0.5:
            frames = (frames * random.uniform(0.85, 1.15)).clamp(-1.0, 1.0)
        return frames


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Pad variable-length audio and video to the longest sequence in the batch.
    Provides length tensors for masked attention in the model.
    String metadata fields (stem, av_class) are returned as plain lists.
    """
    # ── Audio ─────────────────────────────────────────────────────────────
    audios    = [b["audio"] for b in batch]
    max_a     = max(a.shape[0] for a in audios)
    pad_a     = torch.zeros(len(audios), max_a)
    for i, a in enumerate(audios):
        pad_a[i, :a.shape[0]] = a
    a_lengths = torch.tensor([a.shape[0] for a in audios], dtype=torch.long)

    # ── Video ─────────────────────────────────────────────────────────────
    videos         = [b["video"] for b in batch]
    max_t          = max(v.shape[0] for v in videos)
    _, C, H, W     = videos[0].shape
    pad_v          = torch.zeros(len(videos), max_t, C, H, W)
    for i, v in enumerate(videos):
        pad_v[i, :v.shape[0]] = v
    v_lengths = torch.tensor([v.shape[0] for v in videos], dtype=torch.long)

    return {
        "audio":         pad_a,                                         # (B, T_a)
        "audio_lengths": a_lengths,                                     # (B,)
        "video":         pad_v,                                         # (B, T_v, 1, H, W)
        "video_lengths": v_lengths,                                     # (B,)
        "audio_label":   torch.stack([b["audio_label"] for b in batch]),
        "video_label":   torch.stack([b["video_label"] for b in batch]),
        "av_label":      torch.stack([b["av_label"]    for b in batch]),
        "stem":          [b["stem"]     for b in batch],
        "av_class":      [b["av_class"] for b in batch],
    }