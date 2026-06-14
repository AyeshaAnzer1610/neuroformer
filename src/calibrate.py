#!/usr/bin/env python
"""
Contribution #4 — calibrated, referral-aware screening.

On the best leak-free model (frozen LaBraM EEG, the Phase 1 signal), add the
clinical-triage layer reviewers wanted, grounded in conformal-prediction triage
work (Sci. Rep. 2025; EEG conformal 2026):

1. **Temperature scaling** + Expected Calibration Error (ECE) before/after — are
   the predicted probabilities trustworthy?
2. **Selective prediction / risk-coverage** — abstain on low-confidence cases
   ("refer to clinician"); report metric vs the fraction retained (coverage).
3. **Split-conformal prediction sets** at target error alpha — decisive
   (singleton) vs uncertain ({0,1} -> refer) rate, with empirical coverage.

All on leak-free out-of-fold predictions from subject-level CV.

    python src/calibrate.py --eeg-dir $SCRATCH/mh/data/processed --labels data/labels.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from compare_fm_vs_handcrafted import load_windows, pool_subjects


def oof_probs(X, y, folds=5, seed=0):
    """Leak-free out-of-fold P(positive): each subject scored by a model that
    never saw it. One stratified partition (so each subject appears once)."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        # Plain (unweighted) logreg: class_weight='balanced' deliberately
        # distorts probabilities, which defeats a calibration analysis. We
        # calibrate the honest probabilities via temperature scaling below.
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000))
        pipe.fit(X[tr], y[tr])
        p[te] = pipe.predict_proba(X[te])[:, 1]
    return np.clip(p, 1e-6, 1 - 1e-6)


def ece(p, y, n_bins=10):
    """Expected Calibration Error (binary, positive-class prob)."""
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if m.sum() == 0:
            continue
        conf = p[m].mean()
        acc = ((p[m] >= 0.5).astype(int) == y[m]).mean()
        e += (m.sum() / len(p)) * abs(acc - conf)
    return e


def fit_temperature(p, y):
    """1-param temperature on logits, grid-searched to minimize NLL."""
    logit = np.log(p / (1 - p))
    best_T, best_nll = 1.0, np.inf
    for T in np.linspace(0.5, 5.0, 46):
        pt = 1 / (1 + np.exp(-logit / T))
        pt = np.clip(pt, 1e-6, 1 - 1e-6)
        nll = -np.mean(y * np.log(pt) + (1 - y) * np.log(1 - pt))
        if nll < best_nll:
            best_nll, best_T = nll, T
    return best_T


def risk_coverage(p, y):
    """Selective prediction: abstain on least-confident cases. Report
    balanced-accuracy + AUC at decreasing coverage (= retained fraction)."""
    conf = np.abs(p - 0.5)                       # distance from the decision boundary
    order = np.argsort(-conf)                    # most confident first
    rows = []
    for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        k = max(int(round(cov * len(p))), 10)
        idx = order[:k]
        yp = (p[idx] >= 0.5).astype(int)
        bacc = balanced_accuracy_score(y[idx], yp)
        auc = roc_auc_score(y[idx], p[idx]) if len(np.unique(y[idx])) > 1 else float("nan")
        rows.append((cov, 1 - cov, bacc, auc))
    return rows


def conformal(p, y, alpha=0.2, splits=20, seed=0):
    """Split-conformal sets on leak-free scores. Nonconformity = 1 - p(true).
    Report: empirical coverage, % decisive (singleton) sets, accuracy among them,
    % referred ({0,1}). Averaged over random calib/test splits."""
    rng = np.random.RandomState(seed)
    P = np.column_stack([1 - p, p])              # class probs [n,2]
    cov_, dec_, acc_, ref_ = [], [], [], []
    n = len(y)
    for _ in range(splits):
        perm = rng.permutation(n); h = n // 2
        cal, te = perm[:h], perm[h:]
        nc = 1 - P[cal, y[cal]]                   # calib nonconformity
        qhat = np.quantile(nc, min(np.ceil((len(cal) + 1) * (1 - alpha)) / len(cal), 1.0))
        sets = P[te] >= (1 - qhat)                # include class k if p_k >= 1-qhat
        covered = sets[np.arange(len(te)), y[te]].mean()
        single = sets.sum(1) == 1
        both = sets.sum(1) == 2
        if single.sum():
            pred = sets[single].argmax(1)
            acc_.append((pred == y[te][single]).mean())
        cov_.append(covered); dec_.append(single.mean()); ref_.append(both.mean())
    return (np.mean(cov_), np.mean(dec_), np.mean(acc_) if acc_ else float("nan"),
            np.mean(ref_))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-dir", required=True, help="Dir of *_labram.npz")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--pool", default="meanstd")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.2, help="Conformal error rate")
    ap.add_argument("--out", default="results/calibration.csv")
    args = ap.parse_args()

    lab = pd.read_csv(args.labels).set_index("ID")["DX_Status"].to_dict()
    emb = pool_subjects(load_windows(args.eeg_dir, "_labram", "embeddings"), args.pool)
    subs = sorted(set(emb) & set(lab))
    X = np.vstack([emb[s] for s in subs])
    y = np.array([lab[s] > 0 for s in subs]).astype(int)
    print(f"[calib] N={len(subs)}, positives={int(y.sum())}")

    # Average OOF probs over seeds (stable leak-free scores).
    P = np.mean([oof_probs(X, y, args.folds, s) for s in range(args.seeds)], axis=0)
    P = np.clip(P, 1e-6, 1 - 1e-6)

    e_raw = ece(P, y)
    T = fit_temperature(P, y)
    Pt = np.clip(1 / (1 + np.exp(-np.log(P / (1 - P)) / T)), 1e-6, 1 - 1e-6)
    e_cal = ece(Pt, y)
    print(f"\n=== Calibration ===")
    print(f"  ECE raw = {e_raw:.3f} -> temperature T={T:.2f} -> ECE = {e_cal:.3f}")

    print(f"\n=== Selective prediction (refer the least-confident) ===")
    print(f"  {'coverage':>9}{'refer%':>8}{'bal-acc':>9}{'AUC':>7}")
    rc = risk_coverage(Pt, y)
    for cov, ref, bacc, auc in rc:
        print(f"  {cov*100:>7.0f}%{ref*100:>7.0f}%{bacc:>9.3f}{auc:>7.3f}")

    cov, dec, acc, ref = conformal(Pt, y, args.alpha)
    print(f"\n=== Split-conformal (alpha={args.alpha}, target coverage {1-args.alpha:.0%}) ===")
    print(f"  empirical coverage = {cov:.3f}; decisive(singleton) = {dec:.1%} "
          f"(acc {acc:.3f}); referred({{0,1}}) = {ref:.1%}")

    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"ece_raw": e_raw, "temperature": T, "ece_cal": e_cal,
                   "conformal_coverage": cov, "decisive_frac": dec,
                   "decisive_acc": acc, "referred_frac": ref}]).to_csv(args.out, index=False)
    pd.DataFrame(rc, columns=["coverage", "refer_frac", "bal_acc", "auc"]).to_csv(
        args.out.replace(".csv", "_riskcoverage.csv"), index=False)
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
