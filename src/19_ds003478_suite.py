# -*- coding: utf-8 -*-
"""
STEP 19: FULL MODEL SUITE ON ds003478  (independent replication)
================================================================
Runs the SAME shared model suite as MIPDB on the ds003478 (Cavanagh depression
rest) cohort, extending ds2_extract_and_benchmark.py from LogReg/SVM/demographics
to the whole suite.

HONEST LIMITATION: ds003478 is EEG-only. Eye-tracking and behaviour do not
exist in that dataset, so every multimodal-fusion model (GMU, Cross-modal
Transformer, Tensor Fusion, Low-rank, NeuroFormer) is reported as **N/A --
requires eye/behaviour modality absent in ds003478**, NOT run degenerately and
NOT fabricated. The EEG-applicable models (LogReg, SVM, RF, MLP on subject-mean
EEG; 1D-CNN/LSTM/TCN on EEG windows; demographics) run for real.

Reuses ds2_features/*.npz already extracted by ds2_extract_and_benchmark.py.

Run:
  python src/19_ds003478_suite.py --n-boot 2000 --out results/ds2_suite.csv
Quick smoke test:
  python src/19_ds003478_suite.py --n-boot 100 --epochs 5
"""
import argparse
import numpy as np, pandas as pd
from model_suite import load_ds003478, build_registry, cv_perfold_auc, cv_oof
from importlib import import_module
_ci = import_module('18_bootstrap_ci') if False else None   # noqa (avoid name clash)


def bootstrap_ci(y, oof, n_boot, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    ok = ~np.isnan(oof); y, oof = y[ok], oof[ok]; n = len(y)
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
    ap.add_argument('--feat-dir', default='data/ds2_features')
    ap.add_argument('--root', default='data/ds003478')
    ap.add_argument('--bdi-cut', type=int, default=14)
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--k-eeg', type=int, default=128)
    ap.add_argument('--out', default='results/ds2_suite.csv')
    a = ap.parse_args()

    b = load_ds003478(k_eeg=a.k_eeg, feat_dir=a.feat_dir, root=a.root, bdi_cut=a.bdi_cut)
    print(f"ds003478 cohort: N={b.N} ({int(b.y.sum())} depressed, BDI>={a.bdi_cut}) | "
          f"eeg={b.eeg.shape[1]} | EEG-only (no eye/beh)")
    cfg = {'epochs': a.epochs}

    rows = []
    for name, needs, fp in build_registry():
        if fp is None:
            rows.append(dict(model=name, status='N/A -- torch unavailable',
                             auc_perfold=None, ci_lo=None, ci_hi=None, auc_oof=None))
            print(f"  {name:<28} N/A (torch unavailable)")
            continue
        if 'eye' in needs and not b.has_eye:
            rows.append(dict(model=name, status='N/A -- requires eye/beh (absent in ds003478)',
                             auc_perfold=None, ci_lo=None, ci_hi=None, auc_oof=None))
            print(f"  {name:<28} N/A (needs eye/beh, absent here)")
            continue
        try:
            pf = cv_perfold_auc(fp, b, b.y, n_repeats=a.repeats, cfg=cfg)
            oof = cv_oof(fp, b, b.y, cfg=cfg)
            auc_boot, (lo, hi) = bootstrap_ci(b.y, oof, a.n_boot)
            rows.append(dict(model=name, status='run',
                             auc_perfold=round(float(pf.mean()), 4),
                             ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4),
                             auc_oof=round(auc_boot, 4)))
            print(f"  {name:<28} auc={pf.mean():.3f}  oof={auc_boot:.3f} [{lo:.3f},{hi:.3f}]")
        except Exception as e:
            rows.append(dict(model=name, status=f'ERROR: {type(e).__name__}: {e}',
                             auc_perfold=None, ci_lo=None, ci_hi=None, auc_oof=None))
            print(f"  {name:<28} ERROR: {type(e).__name__}: {e}")

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\n[saved] {a.out}")


if __name__ == '__main__':
    main()
