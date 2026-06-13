#!/usr/bin/env python
"""
Phase 4 — external validation of frozen LaBraM EEG on HBN.

HBN labels are CBCL-derived continuous factors (p_factor = general
psychopathology + attention/internalizing/externalizing), not a categorical DX.
So:
- PRIMARY (clean): within-HBN — predict the p_factor from frozen LaBraM
  embeddings (RestingState), via the same protocol as Phase 1. Report regression
  Pearson r (no threshold) and a median-split AUC (comparable to Phase 1's 0.66).
- SECONDARY (honest transfer): train the MIPDB any-DX EEG classifier, apply to
  HBN; correlate its score with the HBN p_factor + median-split AUC. Flag the
  label mismatch.

    python src/eval_hbn.py \
        --hbn-dir   $SCRATCH/mh/data/processed_hbn \
        --hbn-part  $SCRATCH/mh/data/hbn_r1/participants.tsv \
        --mipdb-dir $SCRATCH/mh/data/processed \
        --mipdb-labels data/labels.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (KFold, RepeatedStratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from compare_fm_vs_handcrafted import load_windows, pool_subjects


def _ci95(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


def load_pooled(npz_dir, pool="meanstd"):
    return pool_subjects(load_windows(npz_dir, "_labram", "embeddings"), pool)


def align(emb, ids, values):
    """emb dict + parallel (ids, values) -> X, y on the intersection."""
    keep = [i for i, s in enumerate(ids) if s in emb]
    subs = [ids[i] for i in keep]
    X = np.vstack([emb[s] for s in subs])
    return X, np.array([values[i] for i in keep], float), subs


def within_hbn(emb, part, target, folds, repeats, seed):
    ids = part["participant_id"].tolist()
    vals = part[target].to_numpy(float)
    ok = np.isfinite(vals)
    ids = [s for s, k in zip(ids, ok) if k]; vals = vals[ok]
    X, y, subs = align(emb, ids, vals)
    print(f"\n=== Within-HBN: predict '{target}' from LaBraM "
          f"(N={len(subs)}) ===")

    # Regression: CV-predicted vs actual p_factor (Pearson r), no threshold.
    reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    cvk = KFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = cross_val_predict(reg, X, y, cv=cvk)
    r, p = pearsonr(pred, y)
    print(f"  regression  Pearson r = {r:.3f} (p={p:.3g})")

    # Binary: median split (high vs low symptom) -> AUC comparable to Phase 1.
    yb = (y > np.median(y)).astype(int)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=5000, class_weight="balanced"))
    cvs = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    from sklearn.model_selection import cross_val_score
    auc = cross_val_score(clf, X, yb, cv=cvs, scoring="roc_auc", n_jobs=-1)
    print(f"  median-split AUC = {auc.mean():.3f} +/- {_ci95(auc):.3f} "
          f"(high p_factor vs low; {int(yb.sum())}/{len(yb)} positive)")
    return {"target": target, "n": len(subs), "reg_r": float(r),
            "auc": float(auc.mean()), "auc_ci95": float(_ci95(auc))}


def transfer(mipdb_emb, mipdb_labels, hbn_emb, part, target="p_factor"):
    """Train MIPDB any-DX classifier, apply to HBN; relate score to HBN target."""
    lab = pd.read_csv(mipdb_labels).set_index("ID")["DX_Status"].to_dict()
    mids = [s for s in mipdb_emb if s in lab]
    Xtr = np.vstack([mipdb_emb[s] for s in mids])
    ytr = np.array([lab[s] > 0 for s in mids], int)

    ids = part["participant_id"].tolist(); vals = part[target].to_numpy(float)
    ok = np.isfinite(vals); ids = [s for s, k in zip(ids, ok) if k]; vals = vals[ok]
    Xte, yte, subs = align(hbn_emb, ids, vals)

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    score = clf.predict_proba(sc.transform(Xte))[:, 1]
    r, p = pearsonr(score, yte)
    yb = (yte > np.median(yte)).astype(int)
    auc = roc_auc_score(yb, score)
    print(f"\n=== Transfer MIPDB(any-DX) -> HBN '{target}' "
          f"(train N={len(mids)}, test N={len(subs)}) ===")
    print(f"  score vs {target}: Pearson r = {r:.3f} (p={p:.3g})")
    print(f"  median-split AUC = {auc:.3f}  [HONEST CAVEAT: label mismatch — "
          f"MIPDB DX vs HBN CBCL factor]")
    return {"target": f"transfer_{target}", "n": len(subs),
            "reg_r": float(r), "auc": float(auc), "auc_ci95": float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hbn-dir", required=True, help="Dir of HBN *_labram.npz")
    ap.add_argument("--hbn-part", required=True, help="HBN participants.tsv")
    ap.add_argument("--mipdb-dir", help="Dir of MIPDB *_labram.npz (for transfer)")
    ap.add_argument("--mipdb-labels", default="data/labels.csv")
    ap.add_argument("--pool", default="meanstd")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/phase4_hbn.csv")
    args = ap.parse_args()

    part = pd.read_csv(args.hbn_part, sep="\t")
    hbn = load_pooled(args.hbn_dir, args.pool)
    rows = []
    for tgt in ["p_factor", "attention", "internalizing", "externalizing"]:
        if tgt in part.columns:
            rows.append(within_hbn(hbn, part, tgt, args.folds, args.repeats, args.seed))

    if args.mipdb_dir:
        mipdb = load_pooled(args.mipdb_dir, args.pool)
        rows.append(transfer(mipdb, args.mipdb_labels, hbn, part, "p_factor"))

    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\n[done] Phase 1 EEG bar (within-MIPDB) = 0.66. Wrote {args.out}")


if __name__ == "__main__":
    main()
