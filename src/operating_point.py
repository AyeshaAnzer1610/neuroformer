#!/usr/bin/env python
"""
Clinical operating-point + risk-coverage analysis for the fine-tuned screening model.

Consumes the per-subject out-of-fold predictions dumped by finetune_labram.py
(--save-preds): columns (repeat, subject, y_true, p). Averages the predicted
probability per subject across CV repeats (one leak-free OOF score per subject),
then reports what an AUC alone hides:
  - ROC-AUC (sanity, with bootstrap CI)
  - operating points: balanced-accuracy threshold AND a high-sensitivity (>=0.85)
    screening threshold -> sensitivity / specificity / PPV / NPV
  - risk-coverage: refer the least-confident cases to a clinician; accuracy and AUC
    on the confident, retained subset rise as coverage drops.

Usage:
    python src/operating_point.py --preds results/finetune_labram_frozen_r10_preds.csv \
        --out results/operating_point.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix


def _rates(y, yhat):
    """sensitivity, specificity, PPV, NPV from binary y / yhat."""
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return sens, spec, ppv, npv


def _threshold_balanced(y, p):
    """Threshold maximizing balanced accuracy = (sens+spec)/2."""
    best_t, best_b = 0.5, -1.0
    for t in np.unique(p):
        sens, spec, _, _ = _rates(y, (p >= t).astype(int))
        b = 0.5 * (sens + spec)
        if b > best_b:
            best_b, best_t = b, t
    return best_t


def _threshold_min_sens(y, p, target=0.85):
    """Lowest-specificity-cost threshold achieving sensitivity >= target
    (screening prioritises catching cases)."""
    cand = sorted(np.unique(p))
    chosen = cand[0]
    for t in cand:
        sens, _, _, _ = _rates(y, (p >= t).astype(int))
        if sens >= target:
            chosen = t          # keep raising t while sensitivity still meets target
    return chosen


def _boot_auc(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        vals.append(roc_auc_score(y[b], p[b]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="per-subject OOF preds CSV from --save-preds")
    ap.add_argument("--target-sens", type=float, default=0.85)
    ap.add_argument("--out", default="results/operating_point.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.preds)
    # one leak-free OOF score per subject = mean predicted prob across CV repeats
    g = df.groupby("subject").agg(y_true=("y_true", "first"), p=("p", "mean")).reset_index()
    y = g["y_true"].to_numpy().astype(int)
    p = g["p"].to_numpy().astype(float)
    n, pos = len(y), int(y.sum())
    print(f"[data] {n} subjects, {pos} positive ({pos/n:.1%}); "
          f"prediction = mean OOF prob across {df['repeat'].nunique()} repeats")

    auc = roc_auc_score(y, p)
    lo, hi = _boot_auc(y, p)
    print(f"[auc ] ROC-AUC = {auc:.3f}  (95% CI {lo:.3f}-{hi:.3f})")

    rows = []
    for name, t in [("balanced-accuracy", _threshold_balanced(y, p)),
                    (f"high-sensitivity(>={args.target_sens:.2f})",
                     _threshold_min_sens(y, p, args.target_sens))]:
        sens, spec, ppv, npv = _rates(y, (p >= t).astype(int))
        print(f"[op  ] {name:28s} thr={t:.3f}  sens={sens:.2f} spec={spec:.2f} "
              f"PPV={ppv:.2f} NPV={npv:.2f}")
        rows.append({"metric": f"op:{name}", "threshold": t, "sens": sens,
                     "spec": spec, "ppv": ppv, "npv": npv})

    # Risk-coverage: refer least-confident (|p-0.5| smallest); evaluate on retained.
    t_bal = _threshold_balanced(y, p)
    conf = np.abs(p - 0.5)
    order = np.argsort(-conf)                     # most confident first
    print("[rc  ] coverage  n   acc    AUC   (refer the rest to a clinician)")
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        k = max(int(round(cov * n)), 2)
        keep = order[:k]
        yk, pk = y[keep], p[keep]
        acc = (yk == (pk >= t_bal).astype(int)).mean()
        a = roc_auc_score(yk, pk) if len(np.unique(yk)) > 1 else float("nan")
        print(f"[rc  ]   {cov:4.0%}   {k:3d}  {acc:.3f}  {a:.3f}")
        rows.append({"metric": "risk_coverage", "coverage": cov, "n": k,
                     "accuracy": acc, "auc": a})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
