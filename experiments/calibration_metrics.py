#!/usr/bin/env python
"""
Calibration metrics on out-of-fold predictions (clinical-screening rigor).

Reads a preds CSV with columns (repeat, subject, y_true, p) as written by
finetune_labram.py --save-preds. Reports, per repeat then averaged:
  - ROC-AUC (sanity)
  - Brier score (lower better)
  - ECE (expected calibration error, 10 bins) and MCE (max)
  - a text reliability table

    python experiments/calibration_metrics.py results/finetune_labram_lora_5x5_preds.csv

NOTE: we currently only saved OOF preds for the LoRA run; to report calibration on the
HEADLINE (frozen) model, re-run finetune_labram --freeze-backbone with --save-preds.
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

path = sys.argv[1] if len(sys.argv) > 1 else "results/finetune_labram_lora_5x5_preds.csv"
rows = list(csv.DictReader(open(path)))
by_rep = defaultdict(list)
for r in rows:
    by_rep[r.get("repeat", "0")].append((float(r["y_true"]), float(r["p"])))


def ece_mce(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = m = 0.0
    table = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if sel.sum() == 0:
            continue
        conf, acc, w = p[sel].mean(), y[sel].mean(), sel.mean()
        gap = abs(acc - conf)
        e += w * gap; m = max(m, gap)
        table.append((lo, hi, int(sel.sum()), conf, acc))
    return e, m, table


aucs, briers, eces, mces = [], [], [], []
for rep, pairs in sorted(by_rep.items()):
    y = np.array([a for a, _ in pairs]); p = np.array([b for _, b in pairs])
    if len(np.unique(y)) < 2:
        continue
    aucs.append(roc_auc_score(y, p)); briers.append(brier_score_loss(y, p))
    e, mm, _ = ece_mce(y, p); eces.append(e); mces.append(mm)


def ms(x):
    x = np.array(x)
    return f"{x.mean():.3f} +/- {1.96*x.std(ddof=1)/len(x)**0.5:.3f}" if len(x) > 1 else f"{x[0]:.3f}"


print(f"[calib] {path}  ({len(by_rep)} repeats, {len(rows)} preds)")
print(f"  ROC-AUC : {ms(aucs)}")
print(f"  Brier   : {ms(briers)}   (lower better; 0.25 = uninformative at base rate 0.5)")
print(f"  ECE     : {ms(eces)}   (lower better; <0.05 well-calibrated)")
print(f"  MCE     : {ms(mces)}")

# pooled reliability table (all repeats)
ally = np.array([float(r["y_true"]) for r in rows]); allp = np.array([float(r["p"]) for r in rows])
_, _, table = ece_mce(ally, allp)
print("\n  reliability (pooled):  bin        n   conf   acc")
for lo, hi, n, conf, acc in table:
    print(f"    [{lo:.1f},{hi:.1f})  {n:5d}  {conf:.3f}  {acc:.3f}")
print("\n(For the paper: report ECE/Brier on the FROZEN headline model — re-run with --save-preds.)")
