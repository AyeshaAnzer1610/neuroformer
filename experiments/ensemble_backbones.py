#!/usr/bin/env python
"""
Multi-backbone frozen-readout ENSEMBLE: average per-subject OOF probabilities across
LaBraM / CBraMod / EEGPT frozen readouts and see if the ensemble beats the best single
backbone (the honest ceiling-pusher; typically a small bump).

Inputs: results/frozen_<bb>_preds.csv (cols: repeat, subject, y_true, p) from
scripts/frozen_backbone_sbatch.sh. Reports per-backbone AUC and the ensemble AUC over
the subjects/repeats common to all available backbones.

    python experiments/ensemble_backbones.py
"""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PAT = re.compile(r"frozen_(\w+)_preds\.csv$")
files = {PAT.search(Path(f).name).group(1): f
         for f in glob.glob("results/frozen_*_preds.csv") if PAT.search(Path(f).name)}
if len(files) < 1:
    raise SystemExit("no results/frozen_<bb>_preds.csv — run frozen_backbone_sbatch.sh first")
print(f"[ensemble] backbones found: {sorted(files)}")

dfs = {bb: pd.read_csv(f).rename(columns={"p": f"p_{bb}"}) for bb, f in files.items()}


def _ci(a):
    a = np.asarray(a, float)
    return 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")


def per_repeat_auc(df, pcol):
    out = []
    for r, g in df.groupby("repeat"):
        if g["y_true"].nunique() > 1:
            out.append(roc_auc_score(g["y_true"], g[pcol]))
    return np.array(out)


# merge all backbones on (repeat, subject) — common rows only
merged = None
for bb, d in dfs.items():
    keep = d[["repeat", "subject", "y_true", f"p_{bb}"]]
    merged = keep if merged is None else merged.merge(keep, on=["repeat", "subject", "y_true"])
print(f"[ensemble] {len(merged)} common (repeat,subject) rows across {len(dfs)} backbones")

print("\nbackbone            ROC-AUC (per-repeat mean +/- 95% CI)")
for bb in sorted(dfs):
    a = per_repeat_auc(merged, f"p_{bb}")
    print(f"  {bb:<16} {a.mean():.3f} +/- {_ci(a):.3f}")

merged["p_ens"] = merged[[f"p_{bb}" for bb in dfs]].mean(axis=1)   # mean-prob ensemble
ens = per_repeat_auc(merged, "p_ens")
print(f"  {'ENSEMBLE (mean)':<16} {ens.mean():.3f} +/- {_ci(ens):.3f}")
best = max(per_repeat_auc(merged, f"p_{bb}").mean() for bb in dfs)
print(f"\n[verdict] ensemble {'>' if ens.mean() > best else '<='} best single backbone "
      f"({ens.mean():.3f} vs {best:.3f})")