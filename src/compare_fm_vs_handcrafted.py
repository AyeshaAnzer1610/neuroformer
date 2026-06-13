#!/usr/bin/env python
"""
Phase 1 gate — head-to-head: frozen LaBraM embeddings vs hand-crafted EEG.

Answers reviewer questions R3.8 / R2.1 / R2.4 ("is EEG worth including, and do
foundation-model features beat the old hand-crafted ones?"). Both feature sets
are pooled per subject (mean over windows), aligned to the same subjects and
labels, and scored with IDENTICAL repeated stratified CV splits so the
comparison is paired and fair. We also score their concatenation (fusion).

Task (default): transdiagnostic binary screening — any diagnosis (DX_Status in
{1,2}) vs none (0). DX_Status is {0:101, 1:22, 2:3}, so multiclass is degenerate
(class 2 n=3); binary is the honest screening framing.

Inputs are the .npz caches produced by:
- src/extract_eeg_embeddings.py   --batch  -> <subject>_labram.npz      (key 'embeddings')
- src/extract_handcrafted_eeg.py  --batch  -> <subject>_handcrafted.npz (key 'features')

Usage:
    python src/compare_fm_vs_handcrafted.py \
        --fm-dir $SCRATCH/mh/data/processed \
        --hc-dir $SCRATCH/mh/data/processed_handcrafted \
        --labels data/labels.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------- loading ----------------------------------

def load_windows(npz_dir, suffix, key):
    """Return {subject_id: [n_windows, n_feat]} from each .npz (no pooling yet).

    Pooling is applied later so we can compare aggregation strategies on the
    same cached per-window features. Files are named <subject><suffix>.npz.
    """
    d = Path(npz_dir)
    if not d.is_dir():
        sys.exit(f"Not a directory: {d}")
    out = {}
    for f in sorted(d.glob(f"*{suffix}.npz")):
        subj = f.name[: -len(f"{suffix}.npz")]
        arr = np.load(f)[key]                       # [n_windows, n_feat]
        if arr.ndim != 2 or arr.shape[0] == 0:
            print(f"[load] skip {f.name}: bad shape {arr.shape}")
            continue
        out[subj] = arr.astype(np.float64)
    print(f"[load] {len(out)} subjects from {d} (*{suffix}.npz)")
    return out


def pool_subjects(windows_dict, mode):
    """Aggregate each subject's [n_windows, n_feat] into one vector.

    mean    -> the window-average (loses cross-window variability)
    std     -> per-feature spread across windows
    meanstd -> concat(mean, std): average + variability (2x dims)
    """
    def pool(a):
        if mode == "mean":
            return a.mean(axis=0)
        if mode == "std":
            return a.std(axis=0)
        if mode == "meanstd":
            return np.concatenate([a.mean(axis=0), a.std(axis=0)])
        raise ValueError(f"unknown pool mode: {mode}")
    return {s: pool(a) for s, a in windows_dict.items()}


def build_xy(feat_dicts, labels_df, task="binary"):
    """Align subjects present in ALL feature dicts and in labels -> X dict, y.

    Returns (subjects, y, {name: X}) on the common subject intersection so every
    feature set is evaluated on exactly the same subjects.
    """
    lab = labels_df.set_index("ID")["DX_Status"].to_dict()
    common = set(lab)
    for fd in feat_dicts.values():
        common &= set(fd)
    subjects = sorted(common)
    if not subjects:
        sys.exit("No overlap between labels and feature caches — check IDs/paths.")

    y_raw = np.array([lab[s] for s in subjects])
    if task == "binary":
        y = (y_raw > 0).astype(int)              # any diagnosis vs none
    else:
        y = y_raw.astype(int)

    X = {name: np.vstack([fd[s] for s in subjects]) for name, fd in feat_dicts.items()}
    pos = int(y.sum()) if task == "binary" else None
    print(f"[align] {len(subjects)} common subjects | task={task}"
          + (f" | positives={pos}/{len(y)}" if pos is not None else "")
          + " | dims " + ", ".join(f"{k}={v.shape[1]}" for k, v in X.items()))
    return subjects, y, X


# ------------------------------- scoring ----------------------------------

def evaluate(X_dict, y, folds=5, repeats=10, seed=0, task="binary"):
    """Paired repeated-stratified-CV. Same splits for every feature set."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats,
                                 random_state=seed)
    if task == "binary":
        scoring = {"auc": "roc_auc", "bacc": "balanced_accuracy"}
    else:
        scoring = {"auc": "roc_auc_ovr_weighted", "bacc": "balanced_accuracy"}

    def pipe():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced"),
        )

    results, per_fold_auc = {}, {}
    for name, X in X_dict.items():
        cvres = cross_validate(pipe(), X, y, cv=cv, scoring=scoring, n_jobs=-1)
        per_fold_auc[name] = cvres["test_auc"]
        results[name] = {
            m: (float(np.mean(cvres[f"test_{m}"])),
                float(_ci95(cvres[f"test_{m}"]))) for m in scoring
        }
    return results, per_fold_auc


def _ci95(x):
    """Half-width of a 95% normal CI for the mean."""
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def paired_delta(per_fold_auc, a, b):
    """Mean paired AUC difference (a - b) over identical folds, with 95% CI."""
    da, db = per_fold_auc.get(a), per_fold_auc.get(b)
    if da is None or db is None:
        return None
    diff = da - db
    return float(diff.mean()), float(_ci95(diff))


# --------------------------------- main -----------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 1 head-to-head: LaBraM vs hand-crafted EEG.")
    ap.add_argument("--fm-dir", required=True, help="Dir of *_labram.npz")
    ap.add_argument("--hc-dir", required=True, help="Dir of *_handcrafted.npz")
    ap.add_argument("--labels", default="data/labels.csv", help="CSV with ID,DX_Status")
    ap.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pools", default="mean,meanstd",
                    help="Comma list of window-pooling modes to compare: "
                         "mean, std, meanstd.")
    ap.add_argument("--out", default="results/phase1_comparison.csv")
    args = ap.parse_args()

    labels_df = pd.read_csv(args.labels)
    fm_w = load_windows(args.fm_dir, "_labram", "embeddings")
    hc_w = load_windows(args.hc_dir, "_handcrafted", "features")

    pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    rows = []
    print(f"\n=== Phase 1 head-to-head ({args.folds}x{args.repeats} stratified CV, "
          f"logreg, class-balanced) ===")
    header = (f"{'pool':<9}{'feature set':<13}{'ROC-AUC (95% CI)':<24}"
              f"{'bal-acc (95% CI)':<24}{'dim':>5}")
    for pool in pools:
        fm = pool_subjects(fm_w, pool)
        hc = pool_subjects(hc_w, pool)
        subjects, y, X = build_xy({"handcrafted": hc, "labram": fm}, labels_df, args.task)
        X["fused"] = np.hstack([X["handcrafted"], X["labram"]])
        results, per_fold_auc = evaluate(X, y, args.folds, args.repeats, args.seed, args.task)

        print(f"\n{header}")
        for name in ["handcrafted", "labram", "fused"]:
            auc_m, auc_ci = results[name]["auc"]
            b_m, b_ci = results[name]["bacc"]
            print(f"{pool:<9}{name:<13}{auc_m:.3f} +/- {auc_ci:.3f}{'':<9}"
                  f"{b_m:.3f} +/- {b_ci:.3f}{'':<9}{X[name].shape[1]:>5}")
            rows.append({"pool": pool, "feature_set": name,
                         "n_subjects": len(subjects), "dim": X[name].shape[1],
                         "auc": results[name]["auc"][0], "auc_ci95": results[name]["auc"][1],
                         "bacc": results[name]["bacc"][0], "bacc_ci95": results[name]["bacc"][1]})
        d = paired_delta(per_fold_auc, "labram", "handcrafted")
        if d:
            delta, ci = d
            verdict = ("LaBraM wins" if delta - ci > 0 else
                       "hand-crafted wins" if delta + ci < 0 else
                       "no sig. difference")
            print(f"  paired AUC delta (LaBraM - hand-crafted): "
                  f"{delta:+.3f} +/- {ci:.3f}  ->  {verdict}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
