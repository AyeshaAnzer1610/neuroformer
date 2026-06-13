#!/usr/bin/env python
"""
Phase 2 validation — re-derived eye/behaviour features vs the original CSVs.

Re-deriving from raw used the *same definitions* as the original Multisource
pipeline, so per-subject values should match closely. This computes, per feature,
the Pearson correlation and median |relative diff| between the freshly extracted
.npz (pooled per subject) and the old CSVs (also pooled per subject), over the
overlapping subjects. High correlation (~1.0) = faithful re-derivation.

    python src/validate_features.py \
        --eye-dir $SCRATCH/mh/data/processed_eye \
        --beh-dir $SCRATCH/mh/data/processed_behaviour
"""
import argparse
import re

import numpy as np
import pandas as pd

from compare_fm_vs_handcrafted import load_windows, pool_subjects
from extract_eye_features import EYE_FEATURES
from extract_behaviour_features import BEH_FEATURES

_SID = re.compile(r"(A\d+)")


def _sid(x):
    m = _SID.match(str(x))
    return m.group(1) if m else None


def old_csv_per_subject(csv_path, id_col):
    df = pd.read_csv(csv_path)
    df["_sid"] = df[id_col].map(_sid)
    feats = [c for c in df.columns if c not in (id_col, "_sid")]
    return df.groupby("_sid")[feats].mean(), feats


def compare(name, npz_dir, suffix, feat_names, old_csv, id_col):
    new = pool_subjects(load_windows(npz_dir, suffix, "features"), "mean")
    old, old_feats = old_csv_per_subject(old_csv, id_col)
    common = sorted(set(new) & set(old.index))
    print(f"\n=== {name}: {len(common)} overlapping subjects "
          f"(new={len(new)}, old={len(old)}) ===")
    if not common:
        print("  no overlap — check IDs"); return
    New = np.vstack([new[s] for s in common])
    # Align old columns to our feature order where names match.
    cols = [f for f in feat_names if f in old.columns]
    Old = old.loc[common, cols].to_numpy(float)
    idx = [feat_names.index(f) for f in cols]
    print(f"  {'feature':<22}{'pearson r':>10}{'median |rel diff|':>20}")
    for k, f in enumerate(cols):
        a, b = New[:, idx[k]], Old[:, k]
        ok = np.isfinite(a) & np.isfinite(b)
        r = np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 2 else np.nan
        denom = np.where(np.abs(b[ok]) > 1e-9, np.abs(b[ok]), np.nan)
        rel = np.nanmedian(np.abs(a[ok] - b[ok]) / denom)
        flag = "" if (np.isnan(r) or r > 0.9) else "  <-- CHECK"
        print(f"  {f:<22}{r:>10.3f}{rel*100:>18.1f}%{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eye-dir", required=True)
    ap.add_argument("--beh-dir", required=True)
    ap.add_argument("--eye-csv", default="data/eye_features.csv")
    ap.add_argument("--beh-csv", default="data/behaviour_features.csv")
    args = ap.parse_args()
    compare("EYE", args.eye_dir, "_eye", EYE_FEATURES, args.eye_csv, "file_name")
    compare("BEHAVIOUR", args.beh_dir, "_behaviour", BEH_FEATURES, args.beh_csv, "file")


if __name__ == "__main__":
    main()
