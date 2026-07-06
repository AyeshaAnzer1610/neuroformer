# -*- coding: utf-8 -*-
"""
STEP 22: OPTIONAL EXTRAS  (MIPDB, locked N=92)
==============================================
All computed from the SAME leakage-free out-of-fold (OOF) predictions per model
(single stratified 5-fold, subject-level), so every extra is consistent with the
main tables.

  delong      Pairwise DeLong test between every model pair on the shared OOF
              scores (fast DeLong, Sun & Xu 2014) -> AUC diff, z, p.
  calibration Brier score + Expected Calibration Error (ECE, --ece-bins) per model.
  peroof      Per-subject OOF probability for every model, with the true group,
              so you can plot the score distribution by class.

Run:
  python src/22_extras.py --out-delong results/delong_pairwise.csv \
      --out-calib results/calibration_suite.csv --out-oof results/per_subject_oof.csv
Quick smoke test:
  python src/22_extras.py --epochs 5 --only "LogReg (concat)" "Demographics (age+sex)"
"""
import argparse, itertools
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss
from model_suite import load_mipdb, build_registry, cv_oof


# ------------------------------------------------------------ DeLong (Sun & Xu 2014)
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def _fast_delong(preds, m):
    n = preds.shape[1] - m; k = preds.shape[0]
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(pos[r]); ty[r] = _midrank(neg[r]); tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(cov)


def delong_test(y, p1, p2):
    order = (-y).argsort(); m = int(y.sum())
    preds = np.vstack((p1, p2))[:, order]
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var + 1e-12)
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * stats.norm.sf(abs(z)))


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', nargs='*', default=['delong', 'calibration', 'peroof'])
    ap.add_argument('--ece-bins', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--k-eeg', type=int, default=128)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--out-delong', default='results/delong_pairwise.csv')
    ap.add_argument('--out-calib', default='results/calibration_suite.csv')
    ap.add_argument('--out-oof', default='results/per_subject_oof.csv')
    a = ap.parse_args()

    b = load_mipdb(k_eeg=a.k_eeg)
    print(f"MIPDB locked cohort: N={b.N} ({int(b.y.sum())} pos)")
    reg = [(n, nd, fp) for (n, nd, fp) in build_registry() if fp is not None]
    if a.only:
        reg = [r for r in reg if r[0] in set(a.only)]
    cfg = {'epochs': a.epochs}

    # shared OOF per model (same folds -> comparable for DeLong)
    oof = {}
    for name, needs, fp in reg:
        try:
            oof[name] = cv_oof(fp, b, b.y, cfg=cfg)
            print(f"  OOF {name:<28} auc={roc_auc_score(b.y, oof[name]):.4f}")
        except Exception as e:
            print(f"  OOF {name:<28} ERROR: {type(e).__name__}: {e}")
    y = b.y

    if 'delong' in a.which:
        rows = []
        for m1, m2 in itertools.combinations(oof, 2):
            a1, a2, z, p = delong_test(y, oof[m1], oof[m2])
            rows.append(dict(model_a=m1, model_b=m2, auc_a=round(a1, 4), auc_b=round(a2, 4),
                             auc_diff=round(a1 - a2, 4), z=round(z, 3), p_value=round(p, 4)))
        pd.DataFrame(rows).to_csv(a.out_delong, index=False)
        print(f"[saved] {a.out_delong} ({len(rows)} pairs)")

    if 'calibration' in a.which:
        rows = []
        for name, p in oof.items():
            rows.append(dict(model=name, auc=round(roc_auc_score(y, p), 4),
                             brier=round(brier_score_loss(y, p), 4),
                             ece=round(ece(y, p, a.ece_bins), 4)))
        pd.DataFrame(rows).to_csv(a.out_calib, index=False)
        print(f"[saved] {a.out_calib}")

    if 'peroof' in a.which:
        df = pd.DataFrame({'subject': b.subjects, 'group': y})
        for name, p in oof.items():
            df[name] = np.round(p, 4)
        df.to_csv(a.out_oof, index=False)
        print(f"[saved] {a.out_oof} (per-subject OOF by group)")


if __name__ == '__main__':
    main()
