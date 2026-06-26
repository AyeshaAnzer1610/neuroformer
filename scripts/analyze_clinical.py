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

NeuroFormer is seed-fragile, so oof_nf.csv stores the per-participant OOF prob FOR
EACH of the 8 seeds (cols seed0..seedK). Every NeuroFormer metric is computed
per-seed and reported as mean +/- std over seeds. This keeps the AUC consistent
with the paper's per-seed pooled-OOF headline (~0.62) instead of the seed-ensemble
mean prob (which inflates AUC to ~0.70). age+sex is a single deterministic CV.

Inputs (all produced by a single neuroformer_baselines.py run):
    results/oof_nf.csv        ID,y,nf_prob,seed0..seedK   (NeuroFormer, per seed)
    results/oof_confound.csv  ID,y,agesex_prob            (age+sex baseline)
    results/oof_table2.pkl    {y, subs, oof}              (12 fusion/temporal baselines)
    results/nf_seed_stability.csv  seed,...,incremental_pp (for TOST)
    data/labels.csv           ID,Age,Sex,DX_Status        (subgroup metadata)

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
MIN_CLASS = 5          # a subgroup with < MIN_CLASS in either class is flagged underpowered


# ----------------------------------------------------------------------------- helpers
def _per_participant(oof):
    """Collapse a (repeats, N) OOF matrix to one prob per participant (mean over repeats)."""
    a = np.asarray(oof, float)
    return np.nanmean(a, axis=0) if a.ndim == 2 else a


def operating_point(y, p):
    """Youden-optimal threshold + clinical metrics at that threshold."""
    y = np.asarray(y, int)
    fpr, tpr, thr = roc_curve(y, p)
    k = int(np.argmax(tpr - fpr))
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
                ppv=ppv, npv=npv, f1=f1, prevalence=float(y.mean()), n=int(len(y)))


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


def agg(dicts, keys):
    """mean + ddof=1 std over a list of metric dicts, for each key."""
    out = {}
    for k in keys:
        v = np.array([d[k] for d in dicts], float)
        out[k] = float(np.nanmean(v))
        out[k + "_std"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan")
    return out


# ---- DeLong test for two correlated ROC AUCs (Sun & Xu 2014 fast algorithm) ----
def _compute_midrank(x):
    J = np.argsort(x); z = x[J]; N = len(x); t = np.zeros(N, float); i = 0
    while i < N:
        j = i
        while j < N and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, float); out[J] = t
    return out


def _fast_delong(preds_sorted, m):
    n = preds_sorted.shape[1] - m; k = preds_sorted.shape[0]
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    tx = np.empty((k, m), float); ty = np.empty((k, n), float); tz = np.empty((k, m + n), float)
    for r in range(k):
        tx[r] = _compute_midrank(pos[r]); ty[r] = _compute_midrank(neg[r]); tz[r] = _compute_midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(s)


def delong_test(y, p1, p2):
    """Two-sided p-value that AUC(p1) != AUC(p2) on the same labels y."""
    y = np.asarray(y, int)
    order = np.argsort(-y); m = int(y.sum())
    preds = np.vstack([np.asarray(p1, float)[order], np.asarray(p2, float)[order]])
    aucs, s = _fast_delong(preds, m)
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), float("nan")
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(2.0 * stats.norm.sf(abs(z)))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float); order = np.argsort(p); n = len(p); adj = np.empty(n, float); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (n - rank) * p[idx]); adj[idx] = min(run, 1.0)
    return adj


def load_nf():
    """NeuroFormer OOF -> (ids, y, P) where P is (N, n_seeds); falls back to single col."""
    f = RES / "oof_nf.csv"
    if not f.exists():
        print("  [skip] oof_nf.csv not found -- run neuroformer_baselines.py")
        return None
    d = pd.read_csv(f)
    seed_cols = [c for c in d.columns if c.startswith("seed")]
    if not seed_cols:
        seed_cols = ["nf_prob"]   # backward-compat: only the seed-mean was saved
        print("  [warn] oof_nf.csv has no per-seed columns -> using seed-mean (AUC will read ~0.70, not 0.62)")
    P = d[seed_cols].to_numpy(float)
    return d["ID"].tolist(), d["y"].to_numpy(int), P


def load_confound():
    f = RES / "oof_confound.csv"
    if not f.exists():
        print("  [skip] oof_confound.csv not found -- run neuroformer_baselines.py")
        return None
    d = pd.read_csv(f)
    return d["ID"].tolist(), d["y"].to_numpy(int), d["agesex_prob"].to_numpy(float)


# ----------------------------------------------------------------------------- main
def main():
    print(f"[analyze] results dir: {RES}")
    nf = load_nf()                       # (ids, y, P[N,S])
    cf = load_confound()                 # (ids, y, p[N])
    S = nf[2].shape[1] if nf else 0
    OP_KEYS = ["auc", "threshold", "sensitivity", "specificity", "ppv", "npv", "f1"]

    # ---------- 1. Operating-point metrics @ Youden (NF: mean+/-std over seeds) ----------
    print(f"\n=== 1. Operating-point metrics (Youden) -- NeuroFormer over {S} seeds ===")
    rows = []
    if nf:
        _, y, P = nf
        ops = [operating_point(y, P[:, s]) for s in range(S)]
        r = dict(model="NeuroFormer (3-modality)", n_seeds=S, prevalence=ops[0]["prevalence"], n=ops[0]["n"])
        r.update(agg(ops, OP_KEYS))
        rows.append(r)
        print(f"  NeuroFormer  AUC={r['auc']:.3f}+/-{r['auc_std']:.3f}  sens={r['sensitivity']:.2f} "
              f"spec={r['specificity']:.2f} PPV={r['ppv']:.2f} NPV={r['npv']:.2f} F1={r['f1']:.2f}")
    if cf:
        _, y, p = cf
        op = operating_point(y, p)
        r = dict(model="age+sex", n_seeds=1, prevalence=op["prevalence"], n=op["n"])
        for k in OP_KEYS:
            r[k] = op[k]; r[k + "_std"] = float("nan")
        rows.append(r)
        print(f"  age+sex      AUC={r['auc']:.3f}        sens={r['sensitivity']:.2f} "
              f"spec={r['specificity']:.2f} PPV={r['ppv']:.2f} NPV={r['npv']:.2f} F1={r['f1']:.2f}")
    if rows:
        cols = ["model", "n_seeds", "auc", "auc_std", "threshold", "threshold_std",
                "sensitivity", "sensitivity_std", "specificity", "specificity_std",
                "ppv", "ppv_std", "npv", "npv_std", "f1", "f1_std", "prevalence", "n"]
        pd.DataFrame(rows)[cols].to_csv(RES / "operating_points.csv", index=False)
        print(f"  -> {RES / 'operating_points.csv'}")

    # ---------- 2. TOST equivalence for the demographic increment ----------
    print(f"\n=== 2. TOST equivalence (increment within +/-{TOST_MARGIN_PP}pp of 0) ===")
    sf = RES / "nf_seed_stability.csv"
    if sf.exists():
        inc = pd.read_csv(sf)["incremental_pp"].to_numpy(float)
        n = len(inc); mean = inc.mean(); sd = inc.std(ddof=1); se = sd / np.sqrt(n)
        p_lower = stats.t.sf((mean + TOST_MARGIN_PP) / se, df=n - 1)   # want mean > -margin
        p_upper = stats.t.cdf((mean - TOST_MARGIN_PP) / se, df=n - 1)  # want mean < +margin
        p_tost = max(p_lower, p_upper)
        ci = stats.t.interval(0.90, df=n - 1, loc=mean, scale=se)      # 90% CI <-> TOST at alpha=.05
        equiv = p_tost < 0.05
        print(f"  increment = {mean:+.2f}pp (sd {sd:.2f}, n={n})  90% CI [{ci[0]:+.2f},{ci[1]:+.2f}]pp")
        print(f"  TOST p = {p_tost:.4f}  -> {'EQUIVALENT to 0 within margin' if equiv else 'NOT shown equivalent'}")
        pd.DataFrame([dict(n_seeds=n, increment_mean_pp=mean, increment_sd_pp=sd, margin_pp=TOST_MARGIN_PP,
                           ci90_lo=ci[0], ci90_hi=ci[1], p_lower=p_lower, p_upper=p_upper,
                           p_tost=p_tost, equivalent=equiv)]).to_csv(RES / "equivalence_tost.csv", index=False)
        print(f"  -> {RES / 'equivalence_tost.csv'}")
    else:
        print("  [skip] nf_seed_stability.csv not found")

    # ---------- 3. Subgroup AUCs (age tertile, sex) -- NF mean+/-std over seeds ----------
    print("\n=== 3. Subgroup analysis (age tertile, sex) ===")
    meta_f = ROOT / "data" / "labels.csv"
    sub_rows = []
    if meta_f.exists():
        meta = pd.read_csv(meta_f).set_index("ID")

        def subgroup_auc(ids, y, P, mask):
            yy = y[mask]
            under = (yy.sum() < MIN_CLASS) or ((~yy.astype(bool)).sum() < MIN_CLASS)
            if mask.sum() < MIN_CLASS or yy.min() == yy.max():
                return float("nan"), float("nan"), True
            aucs = [roc_auc_score(yy, P[mask, s]) for s in range(P.shape[1])]
            return float(np.mean(aucs)), (float(np.std(aucs, ddof=1)) if len(aucs) > 1 else float("nan")), under

        for label, d in [("NeuroFormer (3-modality)", nf), ("age+sex", None if cf is None else (cf[0], cf[1], cf[2][:, None]))]:
            if d is None:
                continue
            ids, y, P = d
            age = meta.loc[ids, "Age"].to_numpy(float); sex = meta.loc[ids, "Sex"].to_numpy(int)
            q1, q2 = np.quantile(age, [1 / 3, 2 / 3])
            groups = {f"age<={q1:.0f}": age <= q1, f"age {q1:.0f}-{q2:.0f}": (age > q1) & (age <= q2),
                      f"age>{q2:.0f}": age > q2, "sex=1": sex == 1, "sex=2": sex == 2,
                      "overall": np.ones(len(y), bool)}
            for gname, mask in groups.items():
                a, asd, under = subgroup_auc(ids, y, P, mask)
                sub_rows.append(dict(model=label, subgroup=gname, n=int(mask.sum()),
                                     n_pos=int(y[mask].sum()), auc=a, auc_std=asd, underpowered=under))
                flag = " [underpowered]" if under else ""
                astr = "  n/a" if np.isnan(a) else f"{a:.3f}"
                print(f"  {label:<26} {gname:<12} n={mask.sum():<3} pos={int(y[mask].sum()):<3} AUC={astr}{flag}")
        if sub_rows:
            pd.DataFrame(sub_rows).to_csv(RES / "subgroup_auc.csv", index=False)
            print(f"  -> {RES / 'subgroup_auc.csv'}")
    else:
        print("  [skip] data/labels.csv not found")

    # ---------- 4. Calibration (Brier, ECE) -- NF mean+/-std over seeds ----------
    print("\n=== 4. Calibration (Brier score, ECE) ===")
    cal_rows = []
    if nf:
        _, y, P = nf
        bs = [brier_score_loss(y, np.clip(P[:, s], 0, 1)) for s in range(S)]
        es = [ece(y, np.clip(P[:, s], 0, 1)) for s in range(S)]
        cal_rows.append(dict(model="NeuroFormer (3-modality)", n_seeds=S,
                             brier=float(np.mean(bs)), brier_std=float(np.std(bs, ddof=1)),
                             ece=float(np.mean(es)), ece_std=float(np.std(es, ddof=1)),
                             brier_baseline=float(y.mean() * (1 - y.mean()))))
        r = cal_rows[-1]
        print(f"  NeuroFormer  Brier={r['brier']:.4f}+/-{r['brier_std']:.4f} (no-skill {r['brier_baseline']:.4f})"
              f"  ECE={r['ece']:.4f}+/-{r['ece_std']:.4f}")
    if cf:
        _, y, p = cf; pc = np.clip(p, 0, 1)
        cal_rows.append(dict(model="age+sex", n_seeds=1, brier=brier_score_loss(y, pc), brier_std=float("nan"),
                             ece=ece(y, pc), ece_std=float("nan"), brier_baseline=float(y.mean() * (1 - y.mean()))))
        r = cal_rows[-1]
        print(f"  age+sex      Brier={r['brier']:.4f}        (no-skill {r['brier_baseline']:.4f})  ECE={r['ece']:.4f}")
    if cal_rows:
        pd.DataFrame(cal_rows).to_csv(RES / "calibration.csv", index=False)
        print(f"  -> {RES / 'calibration.csv'}")

    # ---------- 5. Pairwise NeuroFormer vs each baseline (per-seed DeLong + Holm) ----------
    print(f"\n=== 5. Pairwise NeuroFormer vs baselines (DeLong per seed x{S}, Holm) ===")
    pkl = RES / "oof_table2.pkl"
    if nf is None:
        print("  [skip] need oof_nf.csv for the NeuroFormer arm")
    elif not pkl.exists():
        print("  [skip] oof_table2.pkl not found (pull from Nibi or re-run)")
    else:
        ids_nf, y_nf, P = nf
        store = pickle.loads(pkl.read_bytes())
        y_b = np.asarray(store["y"], int)
        if list(store["subs"]) != list(ids_nf):     # align NeuroFormer rows to the pickle order
            idx = {sid: i for i, sid in enumerate(ids_nf)}
            order = [idx[s] for s in store["subs"]]
            P = P[order]; y_nf = y_nf[np.array(order)]
        assert np.array_equal(y_nf, y_b), "label mismatch between oof_nf.csv and oof_table2.pkl"
        extra = [("age+sex", cf[2])] if (cf is not None and list(cf[0]) == list(store["subs"])) else []
        comps = []
        for name, oof in list(store["oof"].items()) + extra:
            p_b = _per_participant(oof)
            per_seed = [delong_test(y_b, P[:, s], p_b) for s in range(S)]
            a_nf = np.mean([t[0] for t in per_seed]); a_b = per_seed[0][1]
            deltas = np.array([t[0] - t[1] for t in per_seed])
            ps = np.array([t[2] for t in per_seed], float)
            comps.append(dict(baseline=name, auc_nf=a_nf, auc_baseline=a_b,
                              delta=float(deltas.mean()), delta_std=float(deltas.std(ddof=1)),
                              p_delong_median=float(np.nanmedian(ps)),
                              frac_seeds_sig=float(np.nanmean(ps < 0.05))))
        adj = holm([c["p_delong_median"] for c in comps])
        for c, pa in zip(comps, adj):
            c["p_holm"] = float(pa)
        comps.sort(key=lambda c: c["p_holm"])
        for c in comps:
            print(f"  NF vs {c['baseline']:<26} dAUC={c['delta']:+.3f}+/-{c['delta_std']:.3f}  "
                  f"p(med)={c['p_delong_median']:.3f} (Holm {c['p_holm']:.3f})")
        pd.DataFrame(comps)[["baseline", "auc_nf", "auc_baseline", "delta", "delta_std",
                             "p_delong_median", "frac_seeds_sig", "p_holm"]].to_csv(
            RES / "pairwise_delong.csv", index=False)
        print(f"  -> {RES / 'pairwise_delong.csv'}")

    print("\n[analyze] done.")


if __name__ == "__main__":
    sys.exit(main())
