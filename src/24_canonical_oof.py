# -*- coding: utf-8 -*-
"""
STEP 24: CANONICAL OUT-OF-FOLD PREDICTIONS  (single source of truth)
====================================================================
Computes the leakage-free out-of-fold (OOF) probability for every model ONCE and
writes results/per_subject_oof.csv. The benchmark point AUC (18), DeLong (22),
and calibration (22) all READ this file, so the OOF-based numbers are identical
across every table instead of being recomputed independently (which previously
diverged: SVC's Platt calibration and torch CPU ops are not bit-reproducible).

Run this BEFORE 18 and 22 whenever the models or cohort change:
  python src/24_canonical_oof.py            # -> results/per_subject_oof.csv
"""
import argparse
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from model_suite import load_mipdb, build_registry, compute_oof_all, save_oof, CANON_OOF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--k-eeg', type=int, default=128)
    ap.add_argument('--out', default=CANON_OOF)
    a = ap.parse_args()

    b = load_mipdb(k_eeg=a.k_eeg)
    reg = [(n, nd, fp) for (n, nd, fp) in build_registry() if fp is not None]
    print(f"MIPDB locked cohort: N={b.N} ({int(b.y.sum())} pos) | computing canonical OOF")
    oof = compute_oof_all(b, reg, cfg={'epochs': a.epochs})
    for name in oof:
        print(f"  {name:<28} OOF AUC = {roc_auc_score(b.y, oof[name]):.4f}")
    path = save_oof(b, oof, a.out)
    print(f"\n[saved] {path}  (canonical OOF; 18 and 22 read from here)")


if __name__ == '__main__':
    main()
