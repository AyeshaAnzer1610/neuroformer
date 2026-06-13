#!/usr/bin/env python
"""
Phase 2 — behavioural features from raw MIPDB task logs.

Re-derives the original Multisource 6-feature behaviour set from
<subject>/Behavioral/csv/*.csv (one vector per task block), saved per subject as
[n_blocks, 6] so it pools per subject like the EEG windows. Same definitions as
the original pipeline (RT = CD_RESP - CD_TG; accuracy = trialLR == RespLR), so
the output can be validated against the old behavioural CSV.

Single subject:
    python src/extract_behaviour_features.py \
        --input  $SCRATCH/mh/data/mipdb/<subject>/ \
        --output $SCRATCH/mh/data/processed_behaviour/<subject>_behaviour.npz

Whole cohort:
    python src/extract_behaviour_features.py --batch \
        --input      $SCRATCH/mh/data/mipdb/ \
        --output-dir $SCRATCH/mh/data/processed_behaviour/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BEH_FEATURES = ["mean_rt", "std_rt", "median_rt", "min_rt", "max_rt", "accuracy"]
REQUIRED = ["CD_TG", "CD_RESP", "trialLR", "RespLR"]


def behaviour_features(path):
    """The original 6 features for one task-block CSV; raises if columns absent."""
    df = pd.read_csv(path)
    if not all(c in df.columns for c in REQUIRED):
        raise ValueError(f"missing columns (need {REQUIRED})")
    rt = pd.to_numeric(df["CD_RESP"], errors="coerce") - pd.to_numeric(df["CD_TG"], errors="coerce")
    acc = (df["trialLR"] == df["RespLR"]).astype(float)
    if rt.notna().sum() == 0:
        raise ValueError("no valid trials")
    return np.array([np.nanmean(rt), np.nanstd(rt), np.nanmedian(rt),
                     np.nanmin(rt), np.nanmax(rt), np.nanmean(acc) * 100.0],
                    dtype=np.float64)


def extract_for_input(input_path, dry_run=False):
    """All behaviour blocks for one subject -> [n_blocks, 6]. Skips bad blocks."""
    p = Path(input_path)
    files = sorted(p.rglob("*.csv")) if p.is_dir() else [p]
    # Restrict to the Behavioral/csv area when given a subject dir.
    files = [f for f in files if "Behavioral" in str(f) or p.is_file()]
    if not files:
        raise ValueError(f"no Behavioral/csv/*.csv under {p}")
    rows, n_other = [], 0
    for f in files:
        try:
            rows.append(behaviour_features(f))
        except ValueError:
            # Expected: only the SAIIT_2AFC task has RT/accuracy columns; other
            # paradigms (SurroundSupp/Video/WISC/vis_learn) don't. Quietly count.
            n_other += 1
        except Exception as e:  # noqa: BLE001 - real parse errors only
            print(f"[beh ] skip {f.name}: {type(e).__name__}: {e}")
    if not rows:
        raise ValueError("no valid behaviour blocks (no SAIIT_2AFC task logs)")
    arr = np.vstack(rows).astype(np.float32)
    print(f"[beh ] {arr.shape[0]} blocks x {arr.shape[1]} features "
          f"({len(files)} files, {n_other} non-2AFC skipped)")
    if dry_run:
        print("[dry-run] OK — no features saved."); return None
    return arr


def main():
    ap = argparse.ArgumentParser(description="Extract MIPDB behavioural features.")
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
                        feature_names=np.array(BEH_FEATURES))
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
        out_path = out_dir / f"{subj.name}_behaviour.npz"
        if out_path.exists() and not args.overwrite:
            skipped.append(subj.name); continue
        try:
            arr = extract_for_input(subj)
            np.savez_compressed(out_path, features=arr,
                                feature_names=np.array(BEH_FEATURES))
            print(f"[batch {i}/{len(subjects)}] {subj.name}: saved {arr.shape}")
            done.append(subj.name)
        except Exception as e:  # noqa: BLE001
            print(f"[batch {i}/{len(subjects)}] {subj.name}: skip ({e})")
            failed.append(subj.name)
    print(f"\n[batch] done={len(done)} skipped(existing)={len(skipped)} "
          f"no-behaviour/failed={len(failed)}")


if __name__ == "__main__":
    main()
