# -*- coding: utf-8 -*-
"""
STEP 18: PER-MODEL 95% CONFIDENCE INTERVALS  (MIPDB, locked N=92)
=================================================================
Two leakage-safe CIs per model, so the manuscript can report proper intervals
(the previous table left NeuroFormer's per-fold CI blank):

  1. per-fold CI : 2.5/97.5 percentile of the RepeatedStratifiedKFold per-fold
                   AUCs -- the same estimator src/6_finalize.py uses.
  2. bootstrap CI: fit ONE out-of-fold prediction vector per model (subject-level,
                   leakage-free), then bootstrap-resample SUBJECTS n_boot times
                   and recompute AUC on the resampled OOF predictions. Resampling
                   subjects (not windows) keeps it leakage-safe.

Run (cluster):
  python src/18_bootstrap_ci.py --n-boot 2000 --out results/table_ci.csv
Quick smoke test:
  python src/18_bootstrap_ci.py --n-boot 200 --only "LogReg (concat)" --epochs 5
"""
import argparse
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from model_suite import load_mipdb, build_registry, cv_perfold_auc, cv_oof


def bootstrap_ci(y, oof, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    ok = ~np.isnan(oof)
    y, oof = y[ok], oof[ok]
    n = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], oof[idx]))
    boots = np.array(boots)
    return float(roc_auc_score(y, oof)), np.percentile(boots, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--repeats', type=int, default=10, help='CV repeats for per-fold CI')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--k-eeg', type=int, default=128)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--out', default='results/table_ci.csv')
    a = ap.parse_args()

    b = load_mipdb(k_eeg=a.k_eeg)
    print(f"MIPDB locked cohort: N={b.N} ({int(b.y.sum())} pos) | n_boot={a.n_boot}")
    reg = [(n, nd, fp) for (n, nd, fp) in build_registry() if fp is not None]
    if a.only:
        reg = [r for r in reg if r[0] in set(a.only)]
    cfg = {'epochs': a.epochs}

    rows = []
    for name, needs, fp in reg:
        try:
            pf = cv_perfold_auc(fp, b, b.y, n_repeats=a.repeats, cfg=cfg)
            oof = cv_oof(fp, b, b.y, cfg=cfg)
            auc_boot, (blo, bhi) = bootstrap_ci(b.y, oof, a.n_boot)
            rows.append(dict(
                model=name, status='ok',
                auc_perfold=round(float(pf.mean()), 4),
                ci_perfold_lo=round(float(np.percentile(pf, 2.5)), 4),
                ci_perfold_hi=round(float(np.percentile(pf, 97.5)), 4),
                auc_oof=round(auc_boot, 4),
                ci_boot_lo=round(float(blo), 4),
                ci_boot_hi=round(float(bhi), 4)))
            print(f"  {name:<28} perfold={pf.mean():.3f} [{np.percentile(pf,2.5):.3f},"
                  f"{np.percentile(pf,97.5):.3f}]  oof={auc_boot:.3f} [{blo:.3f},{bhi:.3f}]")
        except Exception as e:
            rows.append(dict(model=name, status=f'ERROR: {type(e).__name__}: {e}',
                             auc_perfold=None, ci_perfold_lo=None, ci_perfold_hi=None,
                             auc_oof=None, ci_boot_lo=None, ci_boot_hi=None))
            print(f"  {name:<28} ERROR: {type(e).__name__}: {e}")

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\n[saved] {a.out}")


if __name__ == '__main__':
    main()
