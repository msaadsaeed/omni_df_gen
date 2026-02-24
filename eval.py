"""
eval.py
-------
Evaluation and inference entry point for OmniDeepfake.

Modes
-----
  test    : Run the model on the test split and print a full metrics table.
  predict : Run inference on a directory of pre-processed clip folders and
            write predictions to a CSV.

Usage
-----
  # Test split evaluation
  python eval.py test \\
      --ckpt     checkpoints/best.ckpt \\
      --root_dir /data/fakeavceleb_processed \\
      --n_frames 16

  # Detailed per-class breakdown
  python eval.py test \\
      --ckpt     checkpoints/best.ckpt \\
      --root_dir /data/fakeavceleb_processed \\
      --detailed

  # Inference on pre-processed clips (each clip folder contains PNGs + a WAV)
  python eval.py predict \\
      --ckpt        checkpoints/best.ckpt \\
      --root_dir    /data/fakeavceleb_processed \\
      --predict_csv predict_list.csv     \\
      --output_csv  predictions.csv      \\
      --modality    av

  predict_list.csv must have at least two columns:
      faces_path, audio_path
  (same format as the metadata CSVs produced by prepare_fakeav_splits.py)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm

from data.datamodule import FakeAVCelebDataModule
from data.dataset import FakeAVCelebDataset, load_frames, load_audio, collate_fn
from lit_module import OmniDeepfakeLitModel
from utils.metrics import EvalAccumulator, accuracy


AV_CLASS_NAMES = ["RARV", "FAFV", "RAFV", "FARV"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_n_frames(value: str):
    if value.lower() == "all":
        return "all"
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"n_frames must be 'all' or a positive integer, got: {value!r}"
        )
    if n < 1:
        raise argparse.ArgumentTypeError(f"n_frames must be >= 1, got: {n}")
    return n


def _device(args) -> torch.device:
    return torch.device(
        "cuda" if (args.gpus > 0 and torch.cuda.is_available()) else "cpu"
    )


# ---------------------------------------------------------------------------
# Test mode  (Lightning Trainer)
# ---------------------------------------------------------------------------

def run_test(args: argparse.Namespace) -> None:
    """Evaluate a checkpoint on the test split using Lightning Trainer."""
    pl.seed_everything(args.seed)

    lit = OmniDeepfakeLitModel.load_from_checkpoint(args.ckpt, strict=True)
    lit.eval()

    dm = FakeAVCelebDataModule(
        root_dir       = args.root_dir,
        test_csv       = args.test_csv,
        n_frames       = args.n_frames,
        max_audio_secs = args.max_audio_secs,
        batch_size     = args.batch_size,
        num_workers    = args.num_workers,
        augment        = False,
    )

    trainer = pl.Trainer(
        accelerator = "gpu" if args.gpus > 0 else "cpu",
        devices     = args.gpus if args.gpus > 0 else "auto",
        precision   = args.precision,
        logger      = False,
    )
    trainer.test(lit, datamodule=dm)


# ---------------------------------------------------------------------------
# Detailed test mode  (manual loop, full per-class table)
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    """Format a metric value for tqdm postfix (nan-safe)."""
    return "n/a" if v != v else f"{v:.4f}"


def _print_results(metrics: dict, cm=None) -> None:
    """Print the full evaluation table including confusion matrix."""
    W = 56
    print("\n" + "=" * W)
    print("  OmniDeepfake  --  Test Evaluation Results")
    print("=" * W)

    groups = [
        ("Audio (binary)", ["acc_audio", "auc_audio",
                            "acc_audio_real", "acc_audio_fake"]),
        ("Video (binary)", ["acc_video", "auc_video",
                            "acc_video_real", "acc_video_fake"]),
        ("AV (4-class)",   ["acc_av", "auc_av",
                            "acc_av_RARV", "acc_av_FAFV",
                            "acc_av_RAFV", "acc_av_FARV"]),
    ]
    for group_name, keys in groups:
        print(f"\n  [{group_name}]")
        for k in keys:
            v   = metrics.get(k, float("nan"))
            bar = ("█" * int(v * 20)) if v == v else ""
            print(f"    {k:<28} {v:.4f}  {bar}")

    # Confusion matrix
    if cm is not None:
        names = ["RARV", "FAFV", "RAFV", "FARV"]
        print(f"\n  [Confusion Matrix  (rows=True, cols=Predicted)]")
        header = "         " + "".join(f"{n:>8}" for n in names)
        print(f"  {header}")
        print(f"  {'  ' + '-'*40}")
        for i, row_name in enumerate(names):
            row_total = cm[i].sum()
            row_str   = "".join(f"{cm[i,j]:>8}" for j in range(4))
            pct       = 100 * cm[i, i] / row_total if row_total > 0 else 0.0
            print(f"  {row_name:>6} |{row_str}   ({pct:.1f}% correct)")

        # Highlight common misclassification
        print(f"\n  [Top misclassifications]")
        errors = []
        for i in range(4):
            for j in range(4):
                if i != j and cm[i, j] > 0:
                    errors.append((cm[i, j], names[i], names[j]))
        for count, true_cls, pred_cls in sorted(errors, reverse=True)[:5]:
            print(f"    {true_cls} predicted as {pred_cls}: {count} samples")

    print("\n" + "=" * W + "\n")


def run_detailed_eval(args: argparse.Namespace) -> None:
    """Full evaluation with tqdm progress bar, live metrics, and confusion matrix."""
    device = _device(args)

    print("\nLoading checkpoint ...")
    lit = OmniDeepfakeLitModel.load_from_checkpoint(args.ckpt, strict=True)
    lit.eval().to(device)

    dm = FakeAVCelebDataModule(
        root_dir       = args.root_dir,
        test_csv       = args.test_csv,
        n_frames       = args.n_frames,
        max_audio_secs = args.max_audio_secs,
        batch_size     = args.batch_size,
        num_workers    = args.num_workers,
        augment        = False,
    )
    dm.setup("test")
    loader    = dm.test_dataloader()
    n_batches = len(loader)
    n_samples = len(dm.test_ds)

    print(f"Evaluating {n_samples:,} samples  ({n_batches} batches)  on {device}\n")

    accumulator = EvalAccumulator()

    # Running totals for live tqdm postfix (lightweight, no AUC)
    n_correct_audio = n_correct_video = n_correct_av = n_seen = 0

    with torch.inference_mode():
        pbar = tqdm(
            loader,
            total     = n_batches,
            desc      = "Evaluating",
            unit      = "batch",
            ncols     = 100,
            dynamic_ncols = True,
        )
        for batch in pbar:
            tensor_batch = {
                k: v.to(device) for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            outputs = lit.model(tensor_batch)

            # Accumulate for final epoch-level metrics
            accumulator.update(
                logits_audio = outputs["logits_audio"],
                labels_audio = tensor_batch["audio_label"],
                logits_video = outputs["logits_video"],
                labels_video = tensor_batch["video_label"],
                logits_av    = outputs["logits_av"],
                labels_av    = tensor_batch["av_label"],
            )

            # Update running totals for live postfix
            bs            = tensor_batch["audio_label"].size(0)
            n_seen       += bs
            n_correct_audio += (outputs["logits_audio"].argmax(-1) == tensor_batch["audio_label"]).sum().item()
            n_correct_video += (outputs["logits_video"].argmax(-1) == tensor_batch["video_label"]).sum().item()
            n_correct_av    += (outputs["logits_av"].argmax(-1)    == tensor_batch["av_label"]).sum().item()

            pbar.set_postfix({
                "acc_a":  f"{n_correct_audio / n_seen:.4f}",
                "acc_v":  f"{n_correct_video / n_seen:.4f}",
                "acc_av": f"{n_correct_av    / n_seen:.4f}",
                "n":      n_seen,
            }, refresh=True)

    # Final epoch-level metrics (AUC, per-class, confusion matrix)
    metrics = accumulator.compute()
    cm      = accumulator.confusion_matrix()
    _print_results(metrics, cm)


# ---------------------------------------------------------------------------
# Predict mode  (raw inference on a list of clip folders)
# ---------------------------------------------------------------------------

def run_predict(args: argparse.Namespace) -> None:
    """
    Inference on clips listed in --predict_csv.

    CSV must have at minimum:
        faces_path   relative to root_dir  (e.g. RealVideo.../id00076/00109)
        audio_path   relative to root_dir  (e.g. RealVideo.../id00076/00109.wav)

    Modality mask via --modality:
        'audio'  -> ma=1, mv=0
        'video'  -> ma=0, mv=1
        'av'     -> ma=1, mv=1  (default)
    """
    device = _device(args)

    lit = OmniDeepfakeLitModel.load_from_checkpoint(args.ckpt, strict=True)
    lit.eval().to(device)

    df = pd.read_csv(args.predict_csv, sep=None, engine="python")
    root = Path(args.root_dir)

    modality = args.modality.lower()
    ma_val   = 0.0 if modality == "video" else 1.0
    mv_val   = 0.0 if modality == "audio" else 1.0

    rows: List[dict] = []

    with torch.inference_mode():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting",
                           unit="clip", ncols=100):
            faces_dir  = root / str(row["faces_path"])
            audio_file = root / str(row["audio_path"])
            stem       = row.get("stem", faces_dir.name)

            try:
                frames   = load_frames(faces_dir,  n_frames=args.n_frames)
                waveform = load_audio(audio_file,  max_secs=args.max_audio_secs)
            except Exception as exc:
                print(f"  SKIP {stem}: {exc}")
                continue

            # Build a single-sample batch
            batch = collate_fn([{
                "audio":       waveform,
                "video":       frames,
                "audio_label": torch.tensor(0),
                "video_label": torch.tensor(0),
                "av_label":    torch.tensor(0),
                "stem":        str(stem),
                "av_class":    "",
            }])
            tensor_batch = {
                k: v.to(device) for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            tensor_batch["ma"] = torch.tensor([ma_val], device=device)
            tensor_batch["mv"] = torch.tensor([mv_val], device=device)

            probs = lit.predict_step(tensor_batch, 0)

            out_row = {"stem": stem}

            if modality in ("audio", "av"):
                pa = probs["prob_audio"][0].cpu().tolist()
                out_row["p_real_audio"] = f"{pa[0]:.4f}"
                out_row["p_fake_audio"] = f"{pa[1]:.4f}"
                out_row["pred_audio"]   = "FakeAudio" if pa[1] > 0.5 else "RealAudio"

            if modality in ("video", "av"):
                pv = probs["prob_video"][0].cpu().tolist()
                out_row["p_real_video"] = f"{pv[0]:.4f}"
                out_row["p_fake_video"] = f"{pv[1]:.4f}"
                out_row["pred_video"]   = "FakeVideo" if pv[1] > 0.5 else "RealVideo"

            if modality == "av":
                pav = probs["prob_av"][0].cpu().tolist()
                for name, p in zip(AV_CLASS_NAMES, pav):
                    out_row[f"p_{name}"] = f"{p:.4f}"
                out_row["pred_av"] = AV_CLASS_NAMES[int(torch.tensor(pav).argmax())]

            rows.append(out_row)
            label = out_row.get("pred_av") or out_row.get("pred_video") or out_row.get("pred_audio")
            print(f"  {stem}: {label}")

    if rows:
        with open(args.output_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPredictions saved to {args.output_csv}  ({len(rows)} clips)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate OmniDeepfake")
    sub = p.add_subparsers(dest="mode", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--ckpt",            type=str,           required=True)
        sp.add_argument("--root_dir",        type=str,           required=True)
        sp.add_argument("--n_frames",        type=_parse_n_frames, default="all",
                        help="'all' or a positive int (e.g. 16)")
        sp.add_argument("--max_audio_secs",  type=float,         default=16.0)
        sp.add_argument("--batch_size",      type=int,           default=16)
        sp.add_argument("--num_workers",     type=int,           default=8)
        sp.add_argument("--gpus",            type=int,           default=1)
        sp.add_argument("--precision",       type=str,           default="16-mixed")
        sp.add_argument("--seed",            type=int,           default=42)

    # test subcommand
    test_p = sub.add_parser("test", help="Evaluate on the test split")
    add_common(test_p)
    test_p.add_argument("--test_csv",  type=str, default="test/metadata.csv")
    test_p.add_argument("--detailed",  action="store_true",
                        help="Show per-class accuracy and AUC breakdown")

    # predict subcommand
    pred_p = sub.add_parser("predict", help="Run inference on a clip list CSV")
    add_common(pred_p)
    pred_p.add_argument("--predict_csv",  type=str, required=True,
                        help="CSV with faces_path and audio_path columns")
    pred_p.add_argument("--output_csv",   type=str, default="predictions.csv")
    pred_p.add_argument("--modality",     type=str, default="av",
                        choices=["audio", "video", "av"])

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "test":
        if getattr(args, "detailed", False):
            run_detailed_eval(args)
        else:
            run_test(args)
    elif args.mode == "predict":
        run_predict(args)

if __name__ == "__main__":
    main()
