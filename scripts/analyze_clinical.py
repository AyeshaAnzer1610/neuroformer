#!/usr/bin/env python
"""Offline reviewer-requested analyses for Paper A (no GPU, no retraining).

Consumes the per-participant out-of-fold (OOF) probabilities saved by
experiments/neuroformer_baselines.py and produces, into results/:

  Tier 1
    1. Operating-point metrics @ Youden-optimal threshold  -> operating_points.csv
       (sensitivity, specificity, PPV, NPV, F1, threshold, prevalence)
    2. TOST equivalence test for "no gain beyond demographics" -> equivalence_tost.csv
    3. Subgroup AUCs by age tertile and sex                 -> subgroup_auc.csv
  Tier 2
    4. Calibration (Brier score, ECE)                       -> calibration.csv
    5. Pairwise NeuroFormer-vs-baseline (DeLong + Holm)     -> pairwise_delong.csv

Inputs (all produced by a single neuroformer_baselines.py run):
    results/oof_nf.csv        ID,y,nf_prob          (NeuroFormer, seed-averaged)
    results/oof_confound.csv  ID,y,agesex_prob      (age+sex baseline)
    results/oof_table2.pkl    {y, subs, oof}        (12 fusion/temporal baselines)
    results/nf_seed_stability.csv  seed,...,incremental_pp   (for TOST)
    data/labels.csv           ID,Age,Sex,DX_Status  (subgroup metadata)

Run: python scripts/analyze_clinical.py   (from repo root)
Missing inputs are skipped with a clear message rather than crashing.
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TOST_MARGIN_PP = 2.0   # equivalence margin for the demographic increment, in AUC percentage points


# ----------------------------------------------------------------------------- helpers
def _per_participant(oof):
    """Collapse a (repeats, N) OOF matrix to one prob per participant (mean over repeats)."""
    a = np.asarray(oof, float)
    return np.nanmean(a, axis=0) if a.ndim == 2 else a


def operating_point(y, p):
    """Youden-optimal threshold + clinical metrics at that threshold."""
    y = np.asarray(y, int)
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    k = int(np.argmax(j))
    t = float(thr[k])
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")   # = recall
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")    # = precision
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    f1 = (2 * ppv * sens / (ppv + sens)) if (ppv + sens) else float("nan")
    return dict(threshold=t, auc=roc_auc_score(y, p), sensitivity=sens, specificity=spec,
                ppv=ppv, npv=npv, f1=f1, youden_j=float(j[k]),
                tp=tp, fp=fp, tn=tn, fn=fn, prevalence=float(y.mean()), n=int(len(y)))


def ece(y, p, n_bins=10):
    """Expected Calibration Error (equal-width bins)."""
    y = np.asarray(y, int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e, n = 0.0, len(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        e += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(e)


# ---- DeLong test for two correlated ROC AUCs (Sun & Xu 2014 fast algorithm) ----
def _compute_midrank(x):
    J = np.argsort(x)
    z = x[J]
    N = len(x)
    t = np.zeros(N, float)
    i = 0
    while i < N:
        j = i
        while j < N and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, float)
    out[J] = t
    return out


def _fast_delong(preds_sorted, m):
    """preds_sorted: (k, n) with the m positives first, then n-m negatives."""
    n = preds_sorted.shape[1] - m
    k = preds_sorted.shape[0]
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    tx = np.empty((k, m), float)
    ty = np.empty((k, n), float)
    tz = np.empty((k, m + n), float)
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    s = sx / m + sy / n
    return aucs, np.atleast_2d(s)


def delong_test(y, p1, p2):
    """Two-sided p-value that AUC(p1) != AUC(p2) on the same labels y."""
    y = np.asarray(y, int)
    order = np.argsort(-y)          # positives (label 1) first
    m = int(y.sum())
    preds = np.vstack([np.asarray(p1, float)[order], np.asarray(p2, float)[order]])
    aucs, s = _fast_delong(preds, m)
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), float("nan")
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    pval = 2.0 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(pval)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    n = len(p)
    adj = np.empty(n, float)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (n - rank) * p[idx])
        adj[idx] = min(run, 1.0)
    return adj


def load_oof(name, prob_col):
    f = RES / name
    if not f.exists():
        print(f"  [skip] {f.name} not found -- run neuroformer_baselines.py to generate it")
        return None
    d = pd.read_csv(f)
    return d["ID"].tolist(), d["y"].to_numpy(int), d[prob_col].to_numpy(float)


# ----------------------------------------------------------------------------- main
def main():
    print(f"[analyze] results dir: {RES}")
    nf = load_oof("oof_nf.csv", "nf_prob")
    cf = load_oof("oof_confound.csv", "agesex_prob")

    # ---------- 1. Operating-point metrics @ Youden ----------
    print("\n=== 1. Operating-point metrics (Youden-optimal threshold) ===")
    rows = []
    for label, d in [("NeuroFormer (3-modality)", nf), ("age+sex", cf)]:
        if d is None:
            continue
        _, y, p = d
        op = operating_point(y, p); op["model"] = label
        rows.append(op)
        print(f"  {label:<26} AUC={op['auc']:.3f}  sens={op['sensitivity']:.2f} "
              f"spec={op['specificity']:.2f} PPV={op['ppv']:.2f} NPV={op['npv']:.2f} "
              f"F1={op['f1']:.2f} @thr={op['threshold']:.3f}")
    if rows:
        cols = ["model", "auc", "threshold", "sensitivity", "specificity", "ppv", "npv",
                "f1", "youden_j", "tp", "fp", "tn", "fn", "prevalence", "n"]
        pd.DataFrame(rows)[cols].to_csv(RES / "operating_points.csv", index=False)
        print(f"  -> {RES / 'operating_points.csv'}")

    # ---------- 2. TOST equivalence for the demographic increment ----------
    print(f"\n=== 2. TOST equivalence (increment within +/-{TOST_MARGIN_PP}pp of 0) ===")
    sf = RES / "nf_seed_stability.csv"
    if sf.exists():
        inc = pd.read_csv(sf)["incremental_pp"].to_numpy(float)
        n = len(inc); mean = inc.mean(); sd = inc.std(ddof=1); se = sd / np.sqrt(n)
        # two one-sided t-tests against -margin and +margin
        t_lower = (mean - (-TOST_MARGIN_PP)) / se   # H0: mean <= -margin
        t_upper = (mean - (TOST_MARGIN_PP)) / se     # H0: mean >= +margin
        p_lower = stats.t.sf(t_lower, df=n - 1)      # want mean > -margin
        p_upper = stats.t.cdf(t_upper, df=n - 1)     # want mean < +margin
        p_tost = max(p_lower, p_upper)
        ci = stats.t.interval(0.90, df=n - 1, loc=mean, scale=se)  # 90% CI <-> TOST at alpha=.05
        equiv = p_tost < 0.05
        print(f"  increment = {mean:+.2f}pp (sd {sd:.2f}, n={n} seeds)  90% CI [{ci[0]:+.2f},{ci[1]:+.2f}]pp")
        print(f"  TOST p = {p_tost:.4f}  -> {'EQUIVALENT to 0 within margin' if equiv else 'NOT shown equivalent'}")
        pd.DataFrame([dict(n_seeds=n, increment_mean_pp=mean, increment_sd_pp=sd,
                           margin_pp=TOST_MARGIN_PP, ci90_lo=ci[0], ci90_hi=ci[1],
                           p_lower=p_lower, p_upper=p_upper, p_tost=p_tost,
                           equivalent=equiv)]).to_csv(RES / "equivalence_tost.csv", index=False)
        print(f"  -> {RES / 'equivalence_tost.csv'}")
    else:
        print("  [skip] nf_seed_stability.csv not found")

    # ---------- 3. Subgroup AUCs (age tertile, sex) ----------
    print("\n=== 3. Subgroup analysis (age tertile, sex) ===")
    meta_f = ROOT / "data" / "labels.csv"
    sub_rows = []
    if meta_f.exists():
        meta = pd.read_csv(meta_f).set_index("ID")
        for label, d in [("NeuroFormer (3-modality)", nf), ("age+sex", cf)]:
            if d is None:
                continue
            ids, y, p = d
            age = meta.loc[ids, "Age"].to_numpy(float)
            sex = meta.loc[ids, "Sex"].to_numpy(int)
            q1, q2 = np.quantile(age, [1 / 3, 2 / 3])
            groups = {
                f"age<={q1:.0f}": age <= q1,
                f"age {q1:.0f}-{q2:.0f}": (age > q1) & (age <= q2),
                f"age>{q2:.0f}": age > q2,
                "sex=1": sex == 1,
                "sex=2": sex == 2,
                "overall": np.ones(len(y), bool),
            }
            for gname, mask in groups.items():
                yy = y[mask]
                if mask.sum() < 5 or yy.min() == yy.max():
                    auc = float("nan")  # undefined (one class or too few)
                else:
                    auc = roc_auc_score(yy, p[mask])
                sub_rows.append(dict(model=label, subgroup=gname, n=int(mask.sum()),
                                     n_pos=int(yy.sum()), auc=auc))
                print(f"  {label:<26} {gname:<12} n={mask.sum():<3} pos={int(yy.sum()):<3} "
                      f"AUC={'  n/a' if np.isnan(auc) else f'{auc:.3f}'}")
        if sub_rows:
            pd.DataFrame(sub_rows).to_csv(RES / "subgroup_auc.csv", index=False)
            print(f"  -> {RES / 'subgroup_auc.csv'}")
    else:
        print("  [skip] data/labels.csv not found")

    # ---------- 4. Calibration (Brier, ECE) ----------
    print("\n=== 4. Calibration (Brier score, ECE) ===")
    cal_rows = []
    for label, d in [("NeuroFormer (3-modality)", nf), ("age+sex", cf)]:
        if d is None:
            continue
        _, y, p = d
        pc = np.clip(p, 0, 1)
        row = dict(model=label, brier=brier_score_loss(y, pc), ece=ece(y, pc),
                   brier_baseline=float(y.mean() * (1 - y.mean())))  # Brier of always-predict-prevalence
        cal_rows.append(row)
        print(f"  {label:<26} Brier={row['brier']:.4f} (no-skill {row['brier_baseline']:.4f})  ECE={row['ece']:.4f}")
    if cal_rows:
        pd.DataFrame(cal_rows).to_csv(RES / "calibration.csv", index=False)
        print(f"  -> {RES / 'calibration.csv'}")

    # ---------- 5. Pairwise NeuroFormer vs each baseline (DeLong + Holm) ----------
    print("\n=== 5. Pairwise NeuroFormer vs baselines (DeLong, Holm-corrected) ===")
    pkl = RES / "oof_table2.pkl"
    if nf is None:
        print("  [skip] need oof_nf.csv for the NeuroFormer arm")
    elif not pkl.exists():
        print("  [skip] oof_table2.pkl not found (pull from Nibi or re-run)")
    else:
        ids_nf, y_nf, p_nf = nf
        store = pickle.loads(pkl.read_bytes())
        y_b = np.asarray(store["y"], int)
        # align NeuroFormer to the pickle's participant order
        idx = {sid: i for i, sid in enumerate(ids_nf)}
        if list(store["subs"]) != list(ids_nf):
            order = [idx[s] for s in store["subs"]]
            p_nf_a, y_a = p_nf[order], y_nf[np.array(order)]
        else:
            p_nf_a, y_a = p_nf, y_nf
        assert np.array_equal(y_a, y_b), "label mismatch between oof_nf.csv and oof_table2.pkl"
        comps = []
        # include the age+sex baseline if available and aligned
        if cf is not None and list(cf[0]) == list(store["subs"]):
            extra = [("age+sex", cf[2])]
        else:
            extra = []
        for name, oof in list(store["oof"].items()) + extra:
            p_b = _per_participant(oof)
            a_nf, a_b, pv = delong_test(y_b, p_nf_a, p_b)
            comps.append(dict(baseline=name, auc_nf=a_nf, auc_baseline=a_b,
                              delta=a_nf - a_b, p_delong=pv))
        pv_all = [c["p_delong"] for c in comps]
        adj = holm(pv_all)
        for c, pa in zip(comps, adj):
            c["p_holm"] = float(pa)
        comps.sort(key=lambda c: c["p_holm"])
        for c in comps:
            print(f"  NF vs {c['baseline']:<26} dAUC={c['delta']:+.3f}  "
                  f"p={c['p_delong']:.3f} (Holm {c['p_holm']:.3f})")
        pd.DataFrame(comps)[["baseline", "auc_nf", "auc_baseline", "delta",
                             "p_delong", "p_holm"]].to_csv(RES / "pairwise_delong.csv", index=False)
        print(f"  -> {RES / 'pairwise_delong.csv'}")

    print("\n[analyze] done.")


if __name__ == "__main__":
    sys.exit(main())
