"""
utils/metrics.py
─────────────────
Metric helpers for deepfake detection evaluation.

Provides:
  accuracy        — top-1 accuracy
  compute_auc     — macro-averaged one-vs-rest AUC
  per_class_acc   — per-class accuracy dict
  EvalAccumulator — stateful accumulator for epoch-level metrics
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def confusion_matrix(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    """
    Returns a (num_classes, num_classes) confusion matrix as a numpy array.
    Rows = true class, Columns = predicted class.
    """
    preds = logits.argmax(dim=-1).cpu().numpy()
    labs  = labels.cpu().numpy()
    cm    = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labs, preds):
        cm[int(t), int(p)] += 1
    return cm


# ── Stateless helpers ─────────────────────────────────────────────────────────

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Top-1 accuracy. Returns a scalar tensor."""
    return (logits.argmax(dim=-1) == labels).float().mean()


def per_class_acc(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Dict[int, float]:
    """
    Per-class accuracy as a dict {class_idx: acc_float}.
    Useful for spotting which manipulation type the model struggles on.
    """
    preds = logits.argmax(dim=-1)
    result: Dict[int, float] = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            result[c] = float("nan")
        else:
            result[c] = (preds[mask] == c).float().mean().item()
    return result


def compute_auc(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """
    Macro-averaged one-vs-rest ROC-AUC.
    For binary tasks (num_classes=2) this is equivalent to standard AUC.

    Returns nan silently (no warning) when:
      - only one class is present in labels (sklearn UndefinedMetricWarning)
      - any other ValueError (e.g. mismatched shapes)
    """
    import warnings
    probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
    labs  = labels.detach().cpu().numpy()

    # Guard: sklearn raises UndefinedMetricWarning AND ValueError when
    # only one class is present. Check explicitly to avoid the warning.
    unique_classes = np.unique(labs)
    if len(unique_classes) < 2:
        return float("nan")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if num_classes == 2:
                return float(roc_auc_score(labs, probs[:, 1]))
            return float(
                roc_auc_score(labs, probs, multi_class="ovr", average="macro")
            )
    except ValueError:
        return float("nan")


# ── Stateful accumulator ──────────────────────────────────────────────────────

class EvalAccumulator:
    """
    Accumulates logits and labels across batches for epoch-level metric
    computation (AUC, per-class accuracy) without keeping everything in GPU
    memory.

    Usage::

        acc = EvalAccumulator()
        for batch in loader:
            logits = model(batch)
            acc.update(logits_audio=logits["logits_audio"],
                       labels_audio=batch["audio_label"],
                       logits_video=logits["logits_video"],
                       labels_video=batch["video_label"],
                       logits_av=logits["logits_av"],
                       labels_av=batch["av_label"])
        metrics = acc.compute()
        acc.reset()
    """

    def __init__(self) -> None:
        self._store: Dict[str, List[torch.Tensor]] = {
            "logits_audio": [], "labels_audio": [],
            "logits_video": [], "labels_video": [],
            "logits_av":    [], "labels_av":    [],
        }

    def update(
        self,
        logits_audio: torch.Tensor,
        labels_audio: torch.Tensor,
        logits_video: torch.Tensor,
        labels_video: torch.Tensor,
        logits_av:    torch.Tensor,
        labels_av:    torch.Tensor,
    ) -> None:
        for key, val in [
            ("logits_audio", logits_audio), ("labels_audio", labels_audio),
            ("logits_video", logits_video), ("labels_video", labels_video),
            ("logits_av",    logits_av),    ("labels_av",    labels_av),
        ]:
            self._store[key].append(val.detach().cpu())

    def compute(self) -> Dict[str, float]:
        la_all  = torch.cat(self._store["logits_audio"])
        lv_all  = torch.cat(self._store["logits_video"])
        lav_all = torch.cat(self._store["logits_av"])
        ya_all  = torch.cat(self._store["labels_audio"])
        yv_all  = torch.cat(self._store["labels_video"])
        yav_all = torch.cat(self._store["labels_av"])

        metrics: Dict[str, float] = {}

        # Accuracy
        metrics["acc_audio"] = accuracy(la_all, ya_all).item()
        metrics["acc_video"] = accuracy(lv_all, yv_all).item()
        metrics["acc_av"]    = accuracy(lav_all, yav_all).item()

        # AUC
        metrics["auc_audio"] = compute_auc(la_all, ya_all, num_classes=2)
        metrics["auc_video"] = compute_auc(lv_all, yv_all, num_classes=2)
        metrics["auc_av"]    = compute_auc(lav_all, yav_all, num_classes=4)

        # Per-class accuracy for AV head
        pca = per_class_acc(lav_all, yav_all, num_classes=4)
        class_names = ["RARV", "FAFV", "RAFV", "FARV"]
        for idx, name in enumerate(class_names):
            metrics[f"acc_av_{name}"] = pca[idx]

        # Per-class accuracy for binary heads (real=0, fake=1)
        pca_a = per_class_acc(la_all, ya_all, num_classes=2)
        metrics["acc_audio_real"] = pca_a[0]
        metrics["acc_audio_fake"] = pca_a[1]

        pca_v = per_class_acc(lv_all, yv_all, num_classes=2)
        metrics["acc_video_real"] = pca_v[0]
        metrics["acc_video_fake"] = pca_v[1]

        # Confusion matrix (stored separately, not a float metric)
        self._confusion_matrix = confusion_matrix(lav_all, yav_all, num_classes=4)

        return metrics

    def confusion_matrix(self) -> Optional[np.ndarray]:
        """Return the AV confusion matrix from the last compute() call."""
        return getattr(self, "_confusion_matrix", None)

    def reset(self) -> None:
        for key in self._store:
            self._store[key].clear()
