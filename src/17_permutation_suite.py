# -*- coding: utf-8 -*-
"""
STEP 17: 14-MODEL PERMUTATION TABLE  (MIPDB, locked N=92)
=========================================================
For every model in the shared suite, test whether its leakage-free subject-level
CV AUC is above chance by permuting the LABELS and recomputing the same CV AUC.

  observed_m = mean per-fold AUC of model m on the true labels
  null_m     = same statistic on n_perm label permutations
  p_m        = (#{null >= observed} + 1) / (n_perm + 1)

Speed: deep models are expensive, so the null uses fewer CV repeats
(--null-repeats, default 2) exactly as src/5_permutation.py does; the observed
statistic uses --obs-repeats (default 10). Use --n-perm to control cost and
--only to restrict to a subset of models for a quick smoke test.

Run (cluster):
  python src/17_permutation_suite.py --n-perm 1000 --out results/table_perm.csv
Quick smoke test:
  python src/17_permutation_suite.py --n-perm 20 --obs-repeats 2 \
      --only "LogReg (concat)" "Demographics (age+sex)" --epochs 5
"""
import argparse
import numpy as np, pandas as pd
from model_suite import load_mipdb, build_registry, cv_perfold_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--obs-repeats', type=int, default=10)
    ap.add_argument('--null-repeats', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=60, help='torch model epochs')
    ap.add_argument('--k-eeg', type=int, default=128)
    ap.add_argument('--only', nargs='*', default=None, help='subset of model names')
    ap.add_argument('--out', default='results/table_perm.csv')
    a = ap.parse_args()

    b = load_mipdb(k_eeg=a.k_eeg)
    print(f"MIPDB locked cohort: N={b.N} ({int(b.y.sum())} pos) | eeg={b.eeg.shape[1]} "
          f"eye={b.eye.shape[1]} | perms={a.n_perm}")
    reg = [(n, need, fp) for (n, need, fp) in build_registry() if fp is not None]
    if a.only:
        reg = [r for r in reg if r[0] in set(a.only)]
    cfg = {'epochs': a.epochs}
    rng = np.random.default_rng(0)

    rows = []
    for name, needs, fp in reg:
        try:
            obs = cv_perfold_auc(fp, b, b.y, n_repeats=a.obs_repeats, cfg=cfg).mean()
            null = np.empty(a.n_perm)
            for i in range(a.n_perm):
                yp = rng.permutation(b.y)
                null[i] = cv_perfold_auc(fp, b, yp, n_repeats=a.null_repeats, cfg=cfg).mean()
            p = (np.sum(null >= obs) + 1) / (a.n_perm + 1)
            rows.append(dict(model=name, status='ok', observed_auc=round(float(obs), 4),
                             null_mean=round(float(null.mean()), 4),
                             null_std=round(float(null.std()), 4),
                             p_value=round(float(p), 4), n_perm=a.n_perm))
            print(f"  {name:<28} obs={obs:.4f}  null={null.mean():.4f}±{null.std():.4f}  p={p:.4f}")
        except Exception as e:
            rows.append(dict(model=name, status=f'ERROR: {type(e).__name__}: {e}',
                             observed_auc=None, null_mean=None, null_std=None,
                             p_value=None, n_perm=a.n_perm))
            print(f"  {name:<28} ERROR: {type(e).__name__}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"\n[saved] {a.out}")


if __name__ == '__main__':
    main()
