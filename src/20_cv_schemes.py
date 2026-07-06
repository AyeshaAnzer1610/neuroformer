# -*- coding: utf-8 -*-
"""
STEP 20: ALTERNATIVE CV / EVALUATION SCHEMES  (MIPDB, locked N=92)
==================================================================
For every model in the shared suite, re-estimate AUC under several schemes so the
manuscript can show the headline number is not an artifact of one CV design:

  loso        Leave-one-subject-out (strictest subject-level). N=92 folds, 1 test
              subject each -> AUC computed on the POOLED out-of-fold scores
              (+ subject-bootstrap 95% CI).
  multiseed   Repeated stratified 5-fold over many seeds (--seeds); report
              mean +/- SD and the full min/max spread across all folds x seeds.
  b632plus    .632+ bootstrap AUC (Efron & Tibshirani 1997), a small-N estimator
              combining apparent and out-of-bag bootstrap performance.
  nested      Nested CV (inner grid-search for hyperparameters, outer for eval) so
              tuning cannot leak into the estimate. Only the tunable sklearn models
              (LogReg/SVM/RF/MLP, see model_suite.sk_grids); torch nets excluded.

Also writes the FULL fold distribution (not just means) to a long-form CSV.

Run (cluster):
  python src/20_cv_schemes.py --seeds 20 --n-boot 2000 --b632 200 \
      --out results/table_cv_schemes.csv --dist results/cv_fold_distribution.csv
Quick smoke test:
  python src/20_cv_schemes.py --schemes multiseed nested --seeds 2 --epochs 5 \
      --only "LogReg (concat)"
"""
import argparse
import numpy as np, pandas as pd
from sklearn.model_selection import (StratifiedKFold, LeaveOneOut, GridSearchCV,
                                     RepeatedStratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from itertools import product
from model_suite import (load_mipdb, build_registry, cv_oof_splitter, X_concat,
                         sk_grids, torch_grids)


def pooled_bootstrap_ci(y, oof, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    ok = ~np.isnan(oof); y, oof = y[ok], oof[ok]; n = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], oof[idx]))
    return float(roc_auc_score(y, oof)), np.percentile(boots, [2.5, 97.5])


def multiseed(fp, b, cfg, seeds):
    """All per-fold AUCs across repeated 5-fold over `seeds` seeds."""
    allf = []
    for s in seeds:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=s)
        for i, (tr, te) in enumerate(cv.split(np.zeros(b.N), b.y)):
            if len(set(b.y[tr])) < 2 or len(set(b.y[te])) < 2:
                continue
            allf.append((s, roc_auc_score(b.y[te], fp(b, tr, te, b.y, s * 100 + i, cfg))))
    return allf


def b632plus(fp, b, cfg, B, seed=0):
    """.632+ AUC via per-sample averaged out-of-bag predictions."""
    rng = np.random.default_rng(seed); N = b.N; y = b.y
    app = roc_auc_score(y, fp(b, np.arange(N), np.arange(N), y, seed, cfg))
    err_app = 1 - app
    oob_pred = [[] for _ in range(N)]
    for t in range(B):
        idx = rng.integers(0, N, N)
        oob = np.setdiff1d(np.arange(N), idx)
        if len(oob) < 2 or len(set(y[idx])) < 2 or len(set(y[oob])) < 2:
            continue
        p = fp(b, idx, oob, y, seed + t + 1, cfg)
        for j, o in enumerate(oob):
            oob_pred[o].append(p[j])
    mask = np.array([len(pp) > 0 for pp in oob_pred])
    pm = np.array([np.mean(pp) if pp else np.nan for pp in oob_pred])
    err_boot = 1 - roc_auc_score(y[mask], pm[mask])
    gamma = 0.5                                    # no-information AUC
    denom = gamma - err_app
    r = np.clip((err_boot - err_app) / denom, 0, 1) if denom != 0 else 0.0
    w = 0.632 / (1 - 0.368 * r)
    err632 = (1 - w) * err_app + w * err_boot
    return 1 - err632, app, 1 - err_boot


def nested(est, grid, X, y, outer=5, inner=3, seed=0):
    """Nested CV for a sklearn estimator via GridSearchCV (concat features)."""
    pipe = Pipeline([('sc', StandardScaler()), ('clf', est)])
    pgrid = {'clf__' + k: v for k, v in grid.items()}
    aucs = []
    for tr, te in StratifiedKFold(outer, shuffle=True, random_state=seed).split(X, y):
        gs = GridSearchCV(pipe, pgrid, scoring='roc_auc',
                          cv=StratifiedKFold(inner, shuffle=True, random_state=seed))
        gs.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], gs.predict_proba(X[te])[:, 1]))
    return np.array(aucs)


def nested_generic(fp, b, grid, cfg, outer=5, inner=3, seed=0):
    """Nested CV for ANY model via the fit_predict(cfg) interface: inner loop
    grid-searches cfg overrides (e.g. weight-decay), outer loop evaluates.
    Used for the torch models, which have no sklearn grid."""
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in product(*[grid[k] for k in keys])]
    aucs = []
    for tr, te in StratifiedKFold(outer, shuffle=True, random_state=seed).split(np.zeros(b.N), b.y):
        best, best_score = combos[0], -1.0
        for combo in combos:
            c2 = {**cfg, **combo}; inner_aucs = []
            for itr, iva in StratifiedKFold(inner, shuffle=True, random_state=seed).split(
                    np.zeros(len(tr)), b.y[tr]):
                p = fp(b, tr[itr], tr[iva], b.y, seed, c2)
                inner_aucs.append(roc_auc_score(b.y[tr[iva]], p))
            s = float(np.mean(inner_aucs))
            if s > best_score:
                best_score, best = s, combo
        p = fp(b, tr, te, b.y, seed, {**cfg, **best})
        aucs.append(roc_auc_score(b.y[te], p))
    return np.array(aucs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--schemes', nargs='*',
                    default=['loso', 'multiseed', 'b632plus', 'nested'])
    ap.add_argument('--seeds', type=int, default=20, help='#seeds for multiseed')
    ap.add_argument('--n-boot', type=int, default=2000, help='LOSO pooled CI boots')
    ap.add_argument('--b632', type=int, default=200, help='#bootstraps for .632+')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--k-eeg', type=int, default=128)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--out', default='results/table_cv_schemes.csv')
    ap.add_argument('--dist', default='results/cv_fold_distribution.csv')
    a = ap.parse_args()

    b = load_mipdb(k_eeg=a.k_eeg)
    print(f"MIPDB locked cohort: N={b.N} ({int(b.y.sum())} pos) | schemes={a.schemes}")
    reg = [(n, nd, fp) for (n, nd, fp) in build_registry() if fp is not None]
    if a.only:
        reg = [r for r in reg if r[0] in set(a.only)]
    cfg = {'epochs': a.epochs}
    grids = sk_grids()
    tgrids = torch_grids()
    Xc = X_concat(b)

    rows, dist = [], []
    for name, needs, fp in reg:
        rec = {'model': name}
        try:
            if 'loso' in a.schemes:
                oof = cv_oof_splitter(fp, b, b.y, LeaveOneOut(), cfg=cfg)
                auc, (lo, hi) = pooled_bootstrap_ci(b.y, oof, a.n_boot)
                rec.update(loso_auc=round(auc, 4), loso_ci_lo=round(float(lo), 4),
                           loso_ci_hi=round(float(hi), 4))
            if 'multiseed' in a.schemes:
                allf = multiseed(fp, b, cfg, range(a.seeds))
                v = np.array([x[1] for x in allf])
                rec.update(ms_mean=round(float(v.mean()), 4), ms_sd=round(float(v.std()), 4),
                           ms_min=round(float(v.min()), 4), ms_max=round(float(v.max()), 4),
                           ms_nfold=len(v))
                for s, val in allf:
                    dist.append(dict(model=name, scheme='multiseed', seed=s, auc=round(val, 4)))
            if 'b632plus' in a.schemes:
                a632, app, aboot = b632plus(fp, b, cfg, a.b632)
                rec.update(b632plus_auc=round(a632, 4), apparent_auc=round(app, 4),
                           oob_auc=round(aboot, 4))
            if 'nested' in a.schemes and name in grids:
                est, grid = grids[name]
                nv = nested(est, grid, Xc, b.y)
                rec.update(nested_mean=round(float(nv.mean()), 4),
                           nested_sd=round(float(nv.std()), 4))
                for k, val in enumerate(nv):
                    dist.append(dict(model=name, scheme='nested', seed=k, auc=round(float(val), 4)))
            elif 'nested' in a.schemes and name in tgrids:
                nv = nested_generic(fp, b, tgrids[name], cfg)
                rec.update(nested_mean=round(float(nv.mean()), 4),
                           nested_sd=round(float(nv.std()), 4))
                for k, val in enumerate(nv):
                    dist.append(dict(model=name, scheme='nested', seed=k, auc=round(float(val), 4)))
            elif 'nested' in a.schemes:
                rec.update(nested_mean='N/A (no grid defined)')
            print(f"  {name:<28} " + "  ".join(f"{k}={v}" for k, v in rec.items() if k != 'model'))
        except Exception as e:
            rec['error'] = f'{type(e).__name__}: {e}'
            print(f"  {name:<28} ERROR: {type(e).__name__}: {e}")
        rows.append(rec)

    pd.DataFrame(rows).to_csv(a.out, index=False)
    pd.DataFrame(dist).to_csv(a.dist, index=False)
    print(f"\n[saved] {a.out}\n[saved] {a.dist} (full fold distribution)")


if __name__ == '__main__':
    main()
