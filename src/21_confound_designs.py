# -*- coding: utf-8 -*-
"""
STEP 21: CONFOUND-CONTROL / ROBUSTNESS DESIGNS  (MIPDB, locked N=92)
====================================================================
Complements the global permutation test (17) and the CIs (18) with designs that
attack the "are we just reading age/sex?" concern from several angles.

  matched      Covariate-matched subsampling. For each design, resample a
               matched case/control set --matched-n times and report the AUC
               distribution:
                 sex   : each case matched to a random same-sex control
                 joint : same-sex AND nearest-age control
               (age-only matching was done previously; this adds sex + joint.)
  condperm     Confound-CONDITIONED permutation: labels permuted WITHIN
               demographic strata (sex; and joint sex x age-tertile) instead of
               globally, so the null preserves the confound structure. Per-model p.
  confvariants Regress-out vs include-as-feature. For the concat sklearn models,
               per-fold AUC under three feature treatments:
                 raw            : features as-is
                 residualized   : age+sex regressed out (train-fit), classify residual
                 confound-feat  : age+sex appended as extra features
  honestleaky  Per-model honest (subject-level) vs leaky (window-level) AUC WITH
               95% CIs, on the sklearn concat models -- the leakage demonstration
               with intervals.

Run (cluster):
  python src/21_confound_designs.py --matched-n 200 --n-perm 1000 \
      --outdir results
Quick smoke test:
  python src/21_confound_designs.py --designs condperm confvariants \
      --matched-n 5 --n-perm 20 --epochs 5 --only "LogReg (concat)"
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     StratifiedGroupKFold)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from model_suite import (load_mipdb, build_registry, subset, X_concat,
                         window_concat, cv_perfold_auc, sk_estimators)

CONCAT_SK = ['LogReg (concat)', 'SVM-RBF (concat)', 'RandomForest (concat)', 'MLP early-fusion']


# ---------------------------------------------------------------- matched
def match_idx(b, mode, rng):
    pos = list(np.where(b.y == 1)[0]); neg = list(np.where(b.y == 0)[0])
    used, chosen = set(), list(pos)
    for p in pos:
        cand = [c for c in neg if c not in used and b.sex[c] == b.sex[p]]
        if not cand:
            cand = [c for c in neg if c not in used]
        if not cand:
            continue
        if mode == 'joint':
            cand = sorted(cand, key=lambda c: abs(b.age[c] - b.age[p]))[:3]
        pick = cand[int(rng.integers(0, len(cand)))]
        used.add(pick); chosen.append(pick)
    return np.array(sorted(chosen))


def run_matched(reg, b, cfg, modes, n, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for name, needs, fp in reg:
        for mode in modes:
            aucs = []
            for _ in range(n):
                idx = match_idx(b, mode, rng)
                sb = subset(b, idx)
                try:
                    aucs.append(cv_perfold_auc(fp, sb, sb.y, n_repeats=1, cfg=cfg).mean())
                except Exception:
                    pass
            a = np.array(aucs)
            rows.append(dict(model=name, match=mode, n_resamples=len(a),
                             auc_mean=round(float(a.mean()), 4) if len(a) else None,
                             auc_sd=round(float(a.std()), 4) if len(a) else None,
                             auc_lo=round(float(np.percentile(a, 2.5)), 4) if len(a) else None,
                             auc_hi=round(float(np.percentile(a, 97.5)), 4) if len(a) else None))
            print(f"  [matched:{mode}] {name:<26} {a.mean() if len(a) else float('nan'):.4f} "
                  f"+/- {a.std() if len(a) else float('nan'):.4f} (n={len(a)})")
    return rows


# ---------------------------------------------------------------- cond perm
def strat_perm(y, strata, rng):
    yp = y.copy()
    for s in np.unique(strata):
        m = np.where(strata == s)[0]
        yp[m] = rng.permutation(y[m])
    return yp


def run_condperm(reg, b, cfg, n_perm, obs_rep, null_rep, seed=0):
    rng = np.random.default_rng(seed)
    age_tert = np.digitize(b.age, np.quantile(b.age, [1 / 3, 2 / 3]))
    strata = {'sex': b.sex.astype(int),
              'sex_x_agetertile': b.sex.astype(int) * 10 + age_tert}
    rows = []
    for name, needs, fp in reg:
        obs = cv_perfold_auc(fp, b, b.y, n_repeats=obs_rep, cfg=cfg).mean()
        rec = dict(model=name, observed_auc=round(float(obs), 4))
        for sname, strat in strata.items():
            null = np.array([cv_perfold_auc(fp, b, strat_perm(b.y, strat, rng),
                                            n_repeats=null_rep, cfg=cfg).mean()
                             for _ in range(n_perm)])
            p = (np.sum(null >= obs) + 1) / (n_perm + 1)
            rec[f'p_{sname}'] = round(float(p), 4)
            rec[f'null_{sname}_mean'] = round(float(null.mean()), 4)
        rows.append(rec)
        print(f"  [condperm] {name:<26} obs={obs:.4f}  " +
              "  ".join(f"{k}={v}" for k, v in rec.items() if k.startswith('p_')))
    return rows


# ---------------------------------------------------------------- conf variants
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


def run_confvariants(b, seed=0):
    X = X_concat(b); conf = np.column_stack([b.age, b.sex]); rows = []
    for mode in ['raw', 'residualized', 'confound-feat']:
        a = confvariant_auc(X, b.y, conf, mode, seed)
        rows.append(dict(feature_treatment=mode, model='LogReg (concat)',
                         auc=round(float(a.mean()), 4),
                         ci_lo=round(float(np.percentile(a, 2.5)), 4),
                         ci_hi=round(float(np.percentile(a, 97.5)), 4)))
        print(f"  [confvariant] {mode:<14} auc={a.mean():.4f} "
              f"[{np.percentile(a,2.5):.3f},{np.percentile(a,97.5):.3f}]")
    return rows


# ---------------------------------------------------------------- honest vs leaky
def _sk_perfold(Xw, yw, splitter, est_factory, groups=None):
    a = []
    it = splitter.split(Xw, yw, groups) if groups is not None else splitter.split(Xw, yw)
    for tr, te in it:
        if len(set(yw[tr])) < 2 or len(set(yw[te])) < 2:
            continue
        sc = StandardScaler().fit(Xw[tr])
        c = est_factory().fit(sc.transform(Xw[tr]), yw[tr])
        a.append(roc_auc_score(yw[te], c.predict_proba(sc.transform(Xw[te]))[:, 1]))
    return np.array(a)


def run_honestleaky(b, seed=0):
    """Per-model honest (subject-level) vs leaky (window-level) AUC + CIs, for
    every concat sklearn model."""
    Xw, grp, yw = window_concat(b)
    rows = []
    for name, factory in sk_estimators().items():
        honest = _sk_perfold(Xw, yw, StratifiedGroupKFold(5, shuffle=True, random_state=seed),
                             factory, grp)
        leaky = _sk_perfold(Xw, yw, StratifiedKFold(5, shuffle=True, random_state=seed), factory)
        for label, a in [('honest (subject-level)', honest), ('leaky (window-level)', leaky)]:
            rows.append(dict(model=name, cv=label,
                             auc=round(float(a.mean()), 4),
                             ci_lo=round(float(np.percentile(a, 2.5)), 4),
                             ci_hi=round(float(np.percentile(a, 97.5)), 4)))
            print(f"  [honestleaky] {name:<22} {label:<24} auc={a.mean():.4f} "
                  f"[{np.percentile(a,2.5):.3f},{np.percentile(a,97.5):.3f}]")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--designs', nargs='*',
                    default=['matched', 'condperm', 'confvariants', 'honestleaky'])
    ap.add_argument('--matched-n', type=int, default=200)
    ap.add_argument('--match-modes', nargs='*', default=['sex', 'joint'])
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--obs-repeats', type=int, default=10)
    ap.add_argument('--null-repeats', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--k-eeg', type=int, default=128)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--outdir', default='results')
    a = ap.parse_args()

    b = load_mipdb(k_eeg=a.k_eeg)
    print(f"MIPDB locked cohort: N={b.N} ({int(b.y.sum())} pos) | designs={a.designs}")
    reg = [(n, nd, fp) for (n, nd, fp) in build_registry() if fp is not None]
    if a.only:
        reg = [r for r in reg if r[0] in set(a.only)]
    cfg = {'epochs': a.epochs}
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    if 'matched' in a.designs:
        pd.DataFrame(run_matched(reg, b, cfg, a.match_modes, a.matched_n)).to_csv(
            out / 'confound_matched.csv', index=False)
        print(f"[saved] {out/'confound_matched.csv'}")
    if 'condperm' in a.designs:
        pd.DataFrame(run_condperm(reg, b, cfg, a.n_perm, a.obs_repeats, a.null_repeats)).to_csv(
            out / 'confound_condperm.csv', index=False)
        print(f"[saved] {out/'confound_condperm.csv'}")
    if 'confvariants' in a.designs:
        pd.DataFrame(run_confvariants(b)).to_csv(out / 'confound_variants.csv', index=False)
        print(f"[saved] {out/'confound_variants.csv'}")
    if 'honestleaky' in a.designs:
        pd.DataFrame(run_honestleaky(b)).to_csv(out / 'honest_vs_leaky_ci.csv', index=False)
        print(f"[saved] {out/'honest_vs_leaky_ci.csv'}")


if __name__ == '__main__':
    main()
