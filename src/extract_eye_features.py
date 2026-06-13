#!/usr/bin/env python
"""
Phase 2 — eye-tracking features from raw MIPDB (SMI iView exports).

Re-derives the original Multisource 11-feature eye set directly from
<subject>/Eyetracking/txt/*Samples.txt (one vector per task block), saved per
subject as [n_blocks, 11] so it pools per subject like the EEG windows. Same
definitions as the original pipeline, so the output can be validated against the
old processed_asd_features.csv.

Single subject:
    python src/extract_eye_features.py \
        --input  $SCRATCH/mh/data/mipdb/<subject>/ \
        --output $SCRATCH/mh/data/processed_eye/<subject>_eye.npz

Whole cohort:
    python src/extract_eye_features.py --batch \
        --input      $SCRATCH/mh/data/mipdb/ \
        --output-dir $SCRATCH/mh/data/processed_eye/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EYE_FEATURES = ["mean_pupil_size", "std_pupil_size", "mean_gaze_x", "mean_gaze_y",
                "std_gaze_x", "std_gaze_y", "head_pos_variability",
                "mean_latency", "std_latency", "mean_gaze_vector", "std_gaze_vector"]

MIN_SAMPLES = 100   # QC: drop a block with too few valid eye samples


def load_samples(path):
    """Load an SMI *Samples.txt: skip the ## metadata, header is the 'Time' row,
    keep SMP (sample) rows, coerce to numeric."""
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()
    hdr = next((i for i, l in enumerate(lines)
                if l.startswith("Time\t") or ("Time" in l and "Type" in l)), None)
    if hdr is None:
        raise ValueError("no 'Time' header row found")
    df = pd.read_csv(path, sep="\t", skiprows=hdr, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if "Type" in df.columns:
        df = df[df["Type"].astype(str).str.strip() == "SMP"]
    return df.apply(pd.to_numeric, errors="coerce")


def eye_features(df):
    """The original 11 features (mean/std over samples). NaN if a column absent."""
    def mm(cols):
        return df[cols].mean().mean() if all(c in df.columns for c in cols) else np.nan

    def sm(cols):
        return df[cols].std().mean() if all(c in df.columns for c in cols) else np.nan

    gv = ["L GVEC X", "L GVEC Y", "L GVEC Z", "R GVEC X", "R GVEC Y", "R GVEC Z"]
    f = {
        "mean_pupil_size": mm(["L Dia X [px]", "R Dia X [px]"]),
        "std_pupil_size":  sm(["L Dia X [px]", "R Dia X [px]"]),
        "mean_gaze_x": mm(["L POR X [px]", "R POR X [px]"]),
        "mean_gaze_y": mm(["L POR Y [px]", "R POR Y [px]"]),
        "std_gaze_x":  sm(["L POR X [px]", "R POR X [px]"]),
        "std_gaze_y":  sm(["L POR Y [px]", "R POR Y [px]"]),
        "head_pos_variability": sm(["H POS X [mm]", "H POS Y [mm]", "H POS Z [mm]"]),
        "mean_latency": df["Latency"].mean() if "Latency" in df.columns else np.nan,
        "std_latency":  df["Latency"].std() if "Latency" in df.columns else np.nan,
        "mean_gaze_vector": mm(gv),
        "std_gaze_vector":  sm(gv),
    }
    return np.array([f[k] for k in EYE_FEATURES], dtype=np.float64)


def extract_for_input(input_path, dry_run=False):
    """All eye blocks for one subject -> [n_blocks, 11]. Skips bad/empty blocks."""
    p = Path(input_path)
    files = sorted(p.rglob("*Samples.txt")) if p.is_dir() else [p]
    if not files:
        sys.exit(f"No *Samples.txt under {p}")
    rows, used = [], []
    for f in files:
        try:
            df = load_samples(f)
            if len(df) < MIN_SAMPLES:
                print(f"[eye ] skip {f.name}: only {len(df)} samples")
                continue
            feats = eye_features(df)
            if np.isnan(feats).all():
                print(f"[eye ] skip {f.name}: no usable columns")
                continue
            rows.append(feats)
            used.append(f.name)
        except Exception as e:  # noqa: BLE001
            print(f"[eye ] skip {f.name}: {type(e).__name__}: {e}")
    if not rows:
        raise ValueError("no valid eye blocks")
    arr = np.vstack(rows).astype(np.float32)
    print(f"[eye ] {arr.shape[0]} blocks x {arr.shape[1]} features "
          f"(from {len(files)} files)")
    if dry_run:
        print("[dry-run] OK — no features saved."); return None
    return arr


def main():
    ap = argparse.ArgumentParser(description="Extract MIPDB eye-tracking features.")
    ap.add_argument("--input", required=True, help="Subject dir; with --batch a parent dir.")
    ap.add_argument("--output", help="Output .npz (single subject).")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--output-dir")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.batch:
        run_batch(args); return
    arr = extract_for_input(args.input, dry_run=args.dry_run)
    if args.dry_run:
        return
    if not args.output:
        sys.exit("Provide --output (or --dry-run).")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=arr,
                        feature_names=np.array(EYE_FEATURES))
    print(f"[done] saved {arr.shape} -> {args.output}")


def run_batch(args):
    parent = Path(args.input)
    if not parent.is_dir():
        sys.exit(f"--batch needs a directory; got {parent}")
    if not args.output_dir:
        sys.exit("--batch requires --output-dir.")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    subjects = sorted(p for p in parent.iterdir() if p.is_dir())
    done, skipped, failed = [], [], []
    for i, subj in enumerate(subjects, 1):
        out_path = out_dir / f"{subj.name}_eye.npz"
        if out_path.exists() and not args.overwrite:
            skipped.append(subj.name); continue
        try:
            arr = extract_for_input(subj)
            np.savez_compressed(out_path, features=arr,
                                feature_names=np.array(EYE_FEATURES))
            print(f"[batch {i}/{len(subjects)}] {subj.name}: saved {arr.shape}")
            done.append(subj.name)
        except Exception as e:  # noqa: BLE001
            print(f"[batch {i}/{len(subjects)}] {subj.name}: skip ({e})")
            failed.append(subj.name)
    print(f"\n[batch] done={len(done)} skipped(existing)={len(skipped)} "
          f"no-eye/failed={len(failed)}")


if __name__ == "__main__":
    main()
