#!/usr/bin/env python
"""
Phase 2 evaluation — does missing-modality fusion beat EEG-alone (0.66)?

Reports, with identical subject-level repeated stratified CV:
1. Single-modality + naive-concat baselines (logreg) on the complete-case cohort.
2. The masked set-fusion model (modality dropout) on:
   - complete-case (all 3 present) vs the baselines;
   - missing-modality cohort (union; some modalities absent per subject) — the
     contribution: one model, any subset.
3. Simulated-missingness degradation: train normally, drop each modality at test.

    python src/eval_fusion.py \
        --eeg-dir $SCRATCH/mh/data/processed \
        --eye-dir $SCRATCH/mh/data/processed_eye \
        --beh-dir $SCRATCH/mh/data/processed_behaviour
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from assemble_multimodal import load_eeg, load_npz_modality, assemble
from fusion_model import MaskedSetFusion

MODS = ["eeg", "eye", "behaviour"]


def _ci95(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


def _standardize(Xtr, Xte):
    out_tr, out_te = {}, {}
    for m in Xtr:
        sc = StandardScaler().fit(Xtr[m])
        out_tr[m] = sc.transform(Xtr[m]).astype(np.float32)
        out_te[m] = sc.transform(Xte[m]).astype(np.float32)
    return out_tr, out_te


def _to_t(d):
    return {m: torch.from_numpy(v) for m, v in d.items()}


def _train_predict(Xtr, ytr, mtr, Xte, mte, dims, epochs=150, seed=0):
    torch.manual_seed(seed)
    model = MaskedSetFusion(dims)
    pos = max(int(ytr.sum()), 1)
    pw = torch.tensor([(len(ytr) - pos) / pos], dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    xtr, ytr_t = _to_t(Xtr), torch.from_numpy(ytr.astype(np.float32))
    mtr_t = torch.from_numpy(mtr.astype(np.float32))
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(xtr, mtr_t), ytr_t)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(_to_t(Xte),
                                torch.from_numpy(mte.astype(np.float32)))).numpy()
    return p


def cv_fusion(X, y, mask, dims, folds=5, repeats=10, seed=0, drop_test=None):
    """Repeated stratified CV of the fusion model. drop_test = modality index to
    zero out at test time (simulated missingness), or None."""
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    aucs = []
    for tr, te in cv.split(np.zeros(len(y)), y):
        Xtr = {m: X[m][tr] for m in dims}
        Xte = {m: X[m][te] for m in dims}
        Xtr, Xte = _standardize(Xtr, Xte)
        mte = mask[te].copy()
        if drop_test is not None:
            mte[:, drop_test] = False
        p = _train_predict(Xtr, y[tr], mask[tr], Xte, mte, dims, seed=seed)
        if len(np.unique(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], p))
    return np.array(aucs)


def cv_logreg(X2d, y, folds=5, repeats=10, seed=0):
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000, class_weight="balanced"))
    from sklearn.model_selection import cross_val_score
    return cross_val_score(pipe, X2d, y, cv=cv, scoring="roc_auc", n_jobs=-1)


def cv_late_fusion(X, y, mask, dims, folds=5, repeats=10, seed=0):
    """Stacking late fusion (leak-free, small-N friendly).

    Per fold: get each modality's OUT-OF-FOLD logreg probability on the train
    split (cross_val_predict), train a meta-logreg on [per-modality prob +
    availability flag] -> y, apply to test (test probs from full-train fits).
    Only 2*M meta-features, so the combiner is low-variance and learns to weight
    the reliable modality. Absent modalities -> prob 0.5 + availability 0.
    """
    from sklearn.model_selection import cross_val_predict
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    M = len(dims)
    aucs = []
    for tr, te in cv.split(np.zeros(len(y)), y):
        Ztr = np.full((len(tr), M), 0.5); Atr = np.zeros((len(tr), M))
        Zte = np.full((len(te), M), 0.5); Ate = np.zeros((len(te), M))
        for j, m in enumerate(dims):
            has_tr = mask[tr, j]
            if len(np.unique(y[tr][has_tr])) < 2 or has_tr.sum() < 10:
                continue
            base = make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=5000, class_weight="balanced"))
            oof = cross_val_predict(base, X[m][tr][has_tr], y[tr][has_tr], cv=3,
                                    method="predict_proba")[:, 1]
            idx = np.where(has_tr)[0]
            Ztr[idx, j] = oof; Atr[idx, j] = 1.0
            base.fit(X[m][tr][has_tr], y[tr][has_tr])
            has_te = mask[te, j]
            Zte[has_te, j] = base.predict_proba(X[m][te][has_te])[:, 1]
            Ate[has_te, j] = 1.0
        meta = LogisticRegression(max_iter=5000, class_weight="balanced")
        meta.fit(np.hstack([Ztr, Atr]), y[tr])
        p = meta.predict_proba(np.hstack([Zte, Ate]))[:, 1]
        if len(np.unique(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], p))
    return np.array(aucs)


def row(name, aucs):
    print(f"  {name:<26}{np.mean(aucs):.3f} +/- {_ci95(aucs):.3f}")
    return {"setting": name, "auc": float(np.mean(aucs)), "auc_ci95": float(_ci95(aucs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-dir", required=True)
    ap.add_argument("--eye-dir", required=True)
    ap.add_argument("--beh-dir", required=True)
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--pool", default="meanstd", help="EEG window pooling")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/phase2_fusion.csv")
    args = ap.parse_args()

    labels_df = pd.read_csv(args.labels)
    mod = {
        "eeg": load_eeg(args.eeg_dir, pool=args.pool),
        "eye": load_npz_modality(args.eye_dir, "_eye", pool="mean"),
        "behaviour": load_npz_modality(args.beh_dir, "_behaviour", pool="mean"),
    }
    dims = {m: len(next(iter(d.values()))) for m, d in mod.items()}
    rows = []

    # ---- complete-case cohort (all 3 present) ----
    subj_c, yc, Xc, maskc = assemble(mod, labels_df, how="intersection")
    print(f"\n=== Complete-case (N={len(subj_c)}, pos={int(yc.sum())}) — "
          f"{args.folds}x{args.repeats} CV ===")
    print("  baselines (logreg):")
    for m in MODS:
        rows.append(row(f"{m}-alone", cv_logreg(Xc[m], yc, args.folds, args.repeats, args.seed)))
    concat = np.hstack([Xc[m] for m in MODS])
    rows.append(row("concat (naive)", cv_logreg(concat, yc, args.folds, args.repeats, args.seed)))
    print("  fusion models:")
    rows.append(row("fusion-attn (all present)",
                    cv_fusion(Xc, yc, maskc, dims, args.folds, args.repeats, args.seed)))
    rows.append(row("late-fusion (all present)",
                    cv_late_fusion(Xc, yc, maskc, dims, args.folds, args.repeats, args.seed)))

    # ---- missing-modality cohort (union; real availability) ----
    subj_u, yu, Xu, masku = assemble(mod, labels_df, how="union")
    cover = ", ".join(f"{m}={int(masku[:, j].sum())}" for j, m in enumerate(MODS))
    print(f"\n=== Missing-modality (N={len(subj_u)}, pos={int(yu.sum())}; {cover}) ===")
    rows.append(row("fusion-attn (missing-aware)",
                    cv_fusion(Xu, yu, masku, dims, args.folds, args.repeats, args.seed)))
    rows.append(row("late-fusion (missing-aware)",
                    cv_late_fusion(Xu, yu, masku, dims, args.folds, args.repeats, args.seed)))

    # ---- simulated-missingness degradation (complete-case, drop one at test) ----
    print("\n=== Degradation: trained on all, drop ONE modality at test ===")
    for j, m in enumerate(MODS):
        rows.append(row(f"drop {m} @test",
                        cv_fusion(Xc, yc, maskc, dims, args.folds, args.repeats,
                                  args.seed, drop_test=j)))

    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\n[done] EEG-alone Phase 1 bar = 0.660. Wrote {args.out}")


if __name__ == "__main__":
    main()
