# -*- coding: utf-8 -*-
"""
STEP 23: ds003478 CONFOUND + PERMUTATION  (independent replication)
===================================================================
Mirrors the MIPDB confound analysis (17 global permutation + 21 conditioned
permutation + regress-out/confound-feature) on ds003478, so the confound story
is tested on a second cohort.

ds003478 is EEG-only: models needing eye/behaviour are skipped (reported N/A by
19_ds003478_suite.py). Age & sex ARE available (participants.tsv), so the
confound designs run.

  perm       global label-permutation p per EEG-applicable model
  condperm   labels permuted within sex, and within sex x age-tertile strata
  confvar    LogReg on EEG under raw / residualized(age+sex) / confound-feature

Run:
  python src/23_ds003478_confound.py --n-perm 1000 --out results/ds2_confound.csv
Quick smoke test:
  python src/23_ds003478_confound.py --n-perm 20 --epochs 5 --only "LogReg (concat)"
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from model_suite import load_ds003478, build_registry, cv_perfold_auc, X_concat


def strat_perm(y, strata, rng):
    yp = y.copy()
    for s in np.unique(strata):
        m = np.where(strata == s)[0]
        yp[m] = rng.permutation(y[m])
    return yp


def confvariant_auc(X, y, conf, mode, seed=0):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
    a = []
    for tr, te in cv.split(X, y):
        Xtr, Xte = X[tr], X[te]
        if mode == 'residualized':
            lr = LinearRegression().fit(conf[tr], Xtr)
            Xtr = Xtr - lr.predict(conf[tr]); Xte = Xte - lr.predict(conf[te])
        elif mode == 'confound-feat':
            Xtr = np.column_stack([Xtr, conf[tr]]); Xte = np.column_stack([Xte, conf[te]])
        sc = StandardScaler().fit(Xtr)
        c = LogisticRegression(max_iter=2000, class_weight='balanced').fit(sc.transform(Xtr), y[tr])
        a.append(roc_auc_score(y[te], c.predict_proba(sc.transform(Xte))[:, 1]))
    return np.array(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', nargs='*', default=['perm', 'condperm', 'confvar'])
    ap.add_argument('--feat-dir', default='data/ds2_features')
    ap.add_argument('--root', default='data/ds003478')
    ap.add_argument('--bdi-cut', type=int, default=14)
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--obs-repeats', type=int, default=10)
    ap.add_argument('--null-repeats', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--k-eeg', type=int, default=128)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--out', default='results/ds2_confound.csv')
    a = ap.parse_args()

    b = load_ds003478(k_eeg=a.k_eeg, feat_dir=a.feat_dir, root=a.root, bdi_cut=a.bdi_cut)
    print(f"ds003478: N={b.N} ({int(b.y.sum())} depressed) | EEG-only")
    reg = [(n, nd, fp) for (n, nd, fp) in build_registry()
           if fp is not None and 'eye' not in nd]     # EEG-applicable only
    if a.only:
        reg = [r for r in reg if r[0] in set(a.only)]
    cfg = {'epochs': a.epochs}
    rng = np.random.default_rng(0)
    age_tert = np.digitize(b.age, np.quantile(b.age, [1 / 3, 2 / 3]))
    strata = {'sex': b.sex.astype(int), 'sex_x_agetertile': b.sex.astype(int) * 10 + age_tert}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if 'perm' in a.which or 'condperm' in a.which:
        for name, needs, fp in reg:
            obs = cv_perfold_auc(fp, b, b.y, n_repeats=a.obs_repeats, cfg=cfg).mean()
            rec = dict(analysis='permutation', model=name, observed_auc=round(float(obs), 4))
            if 'perm' in a.which:
                null = np.array([cv_perfold_auc(fp, b, rng.permutation(b.y),
                                                n_repeats=a.null_repeats, cfg=cfg).mean()
                                 for _ in range(a.n_perm)])
                rec['p_global'] = round(float((np.sum(null >= obs) + 1) / (a.n_perm + 1)), 4)
            if 'condperm' in a.which:
                for sname, strat in strata.items():
                    null = np.array([cv_perfold_auc(fp, b, strat_perm(b.y, strat, rng),
                                                    n_repeats=a.null_repeats, cfg=cfg).mean()
                                     for _ in range(a.n_perm)])
                    rec[f'p_{sname}'] = round(float((np.sum(null >= obs) + 1) / (a.n_perm + 1)), 4)
            rows.append(rec)
            print(f"  {name:<26} " + "  ".join(f"{k}={v}" for k, v in rec.items()
                                                if k.startswith(('p_', 'observed'))))

    if 'confvar' in a.which:
        X = X_concat(b); conf = np.column_stack([b.age, b.sex])
        for mode in ['raw', 'residualized', 'confound-feat']:
            v = confvariant_auc(X, b.y, conf, mode)
            rows.append(dict(analysis='confvariant', model='LogReg (EEG)',
                             feature_treatment=mode, observed_auc=round(float(v.mean()), 4),
                             ci_lo=round(float(np.percentile(v, 2.5)), 4),
                             ci_hi=round(float(np.percentile(v, 97.5)), 4)))
            print(f"  [confvar] {mode:<14} auc={v.mean():.4f}")

    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n[saved] {out}")


if __name__ == '__main__':
    main()
