#!/usr/bin/env python
"""
Faithful 3-modality NeuroFormer (EEG + eye + behaviour) leak-free robustness re-run,
the version closest to the original paper. Strict subject-level repeated stratified CV.

Inputs (complete-case = subjects with all three):
- EEG : data/processed_handcrafted/<sid>_handcrafted.npz   key "features" [n_win, 516]
- eye : data/eye_features.csv                              11 feats x N blocks
- beh : <beh-dir>/<sid>_behaviour.npz                      key "features" [1, 6]
        (rt_mean, rt_std, rt_median, rt_min, rt_max, accuracy; see extract_behaviour.py)

Reports per-modality logreg baselines, naive concat, and the 3-modality NeuroFormer
(src/neuroformer.py) with early stopping, as per-repeat ROC-AUC mean +/- 95% CI.

    python experiments/robustness_3mod.py --beh-dir data/processed_beh --nf-repeats 5
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
SID = re.compile(r"(A\d{8})")


def load_eeg(d="data/processed_handcrafted"):
    out = {}
    for f in sorted(Path(d).glob("*_handcrafted.npz")):
        a = np.nan_to_num(np.load(f)["features"].astype(np.float32))
        if a.ndim == 2 and a.shape[0] > 0:
            out[SID.search(f.name).group(1)] = a
    return out


def load_eye(csv="data/eye_features.csv"):
    df = pd.read_csv(csv)
    feats = [c for c in df.columns if c != "file_name"]
    df["sid"] = df["file_name"].map(lambda s: (SID.search(str(s)).group(1)
                                               if SID.search(str(s)) else None))
    out = {}
    for sid, g in df.dropna(subset=["sid"]).groupby("sid"):
        a = np.nan_to_num(g[feats].to_numpy(np.float32))
        if a.shape[0] > 0:
            out[sid] = a
    return out


def load_beh(d="data/processed_beh"):
    out = {}
    for f in sorted(Path(d).glob("*_behaviour.npz")):
        a = np.nan_to_num(np.load(f)["features"].astype(np.float32))
        if a.ndim == 2 and a.shape[0] > 0:
            out[SID.search(f.name).group(1)] = a
    return out


def _ci(a):
    a = np.asarray(a, float)
    return 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")


def logreg_rep(X, y, folds, repeats, seed):
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    fold = []
    for tr, te in cv.split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        fold.append(roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else np.nan)
    return np.nanmean(np.array(fold).reshape(repeats, folds), axis=1)


# --------------------------------------------------------- NeuroFormer 3-modality
def _pad(seqs, dim, max_len, torch):
    B = len(seqs)
    x = torch.zeros(B, max_len, dim); m = torch.ones(B, max_len, dtype=torch.bool)
    for i, s in enumerate(seqs):
        if s is None or len(s) == 0:
            x[i, 0] = 0.0; m[i, 0] = False; continue
        n = min(len(s), max_len)
        x[i, :n] = torch.from_numpy(np.ascontiguousarray(s[:n])); m[i, :n] = False
    return x, m


def make_batch3(rows, dims, max_len, torch):
    out = {}
    for mod in ("eeg", "eye", "beh"):
        out[mod], out[mod + "_m"] = _pad([r[mod] for r in rows], dims[mod], max_len, torch)
    out["y"] = torch.tensor([r["y"] for r in rows], dtype=torch.long)
    return out


def run_fold_es(tr_rows, te_rows, dims, args, torch, nn, create):
    ytr = np.array([r["y"] for r in tr_rows])
    tri, vai = train_test_split(np.arange(len(tr_rows)), test_size=args.val_frac,
                               stratify=ytr, random_state=args.seed)
    tr2 = [tr_rows[i] for i in tri]; va = [tr_rows[i] for i in vai]
    model = create({"eeg_input_dim": dims["eeg"], "eyetracking_input_dim": dims["eye"],
                    "behavioral_input_dim": dims["beh"], "d_model": 128,
                    "n_encoder_layers": 2, "n_fusion_layers": 2, "n_heads": 8,
                    "dropout": 0.3, "n_classes": 2, "use_auxiliary_loss": False})
    pos = sum(r["y"] for r in tr2); neg = len(tr2) - pos
    lossf = nn.CrossEntropyLoss(weight=torch.tensor([1.0, neg / max(pos, 1)]))
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    def ev(rows):
        model.eval()
        with torch.no_grad():
            b = make_batch3(rows, dims, args.max_len, torch)
            o = model(b["eeg"], b["eye"], b["beh"], b["eeg_m"], b["eye_m"], b["beh_m"])
            p = torch.softmax(o["logits"], -1)[:, 1].numpy()
        yy = np.array([r["y"] for r in rows])
        return roc_auc_score(yy, p) if len(np.unique(yy)) > 1 else np.nan

    best, best_state, bad = -1.0, None, 0
    for _ in range(args.max_epochs):
        model.train(); np.random.shuffle(tr2)
        for i in range(0, len(tr2), args.batch):
            b = make_batch3(tr2[i:i + args.batch], dims, args.max_len, torch)
            opt.zero_grad()
            o = model(b["eeg"], b["eye"], b["beh"], b["eeg_m"], b["eye_m"], b["beh_m"])
            loss = lossf(o["logits"], b["y"])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        v = ev(va); v = 0.0 if np.isnan(v) else v
        if v > best + 1e-4:
            best, bad = v, 0
            best_state = {k: x.detach().clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return ev(te_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beh-dir", default="data/processed_beh")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--nf-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--skip-nf", action="store_true")
    ap.add_argument("--out", default="results/robustness_3mod.csv")
    args = ap.parse_args()
    np.random.seed(args.seed)

    eeg, eye, beh = load_eeg(), load_eye(), load_beh(args.beh_dir)
    lab = pd.read_csv("data/labels.csv").set_index("ID")["DX_Status"].to_dict()
    subs = sorted(set(lab) & set(eeg) & set(eye) & set(beh))
    y = np.array([int(lab[s] > 0) for s in subs])
    print(f"[load] complete-case (EEG+eye+beh) N={len(subs)} pos={int(y.sum())} "
          f"neg={len(y)-int(y.sum())} | dims eeg={next(iter(eeg.values())).shape[1]} "
          f"eye={next(iter(eye.values())).shape[1]} beh={next(iter(beh.values())).shape[1]}")
    if len(subs) < 20:
        sys.exit("too few complete-case subjects — is data/processed_beh populated?")

    Xe = np.stack([eeg[s].mean(0) for s in subs])
    Xy = np.stack([eye[s].mean(0) for s in subs])
    Xb = np.stack([beh[s].mean(0) for s in subs])
    rows_out = []
    print(f"\n=== logreg baselines, {args.folds}x{args.repeats} leak-free subject-level CV ===")
    for name, X in [("EEG-alone", Xe), ("eye-alone", Xy), ("behaviour-alone", Xb),
                    ("concat (EEG+eye+beh)", np.hstack([Xe, Xy, Xb]))]:
        rep = logreg_rep(X, y, args.folds, args.repeats, args.seed)
        m, c = float(np.nanmean(rep)), _ci(rep)
        rows_out.append({"model": name, "auc": m, "ci95": c, "protocol": f"{args.folds}x{args.repeats}"})
        print(f"  {name:<24} ROC-AUC = {m:.3f} +/- {c:.3f}")

    if not args.skip_nf:
        import torch
        import torch.nn as nn
        from neuroformer import create_neuroformer
        dims = {"eeg": Xe.shape[1], "eye": Xy.shape[1], "beh": Xb.shape[1]}
        nf = [{"sid": s, "y": int(lab[s] > 0), "eeg": eeg[s], "eye": eye[s], "beh": beh[s]}
              for s in subs]
        cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.nf_repeats,
                                     random_state=args.seed)
        print(f"\n=== NeuroFormer 3-modality, {args.folds}x{args.nf_repeats} leak-free CV, "
              f"early-stop (max {args.max_epochs}ep, patience {args.patience}) ===")
        fa = []
        for k, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y), 1):
            torch.manual_seed(args.seed + k); np.random.seed(args.seed + k)
            auc = run_fold_es([nf[i] for i in tr], [nf[i] for i in te], dims, args,
                              torch, nn, create_neuroformer)
            fa.append(auc); print(f"  [fold {k}/{args.folds*args.nf_repeats}] AUC={auc:.3f}", flush=True)
        rep = np.nanmean(np.array(fa).reshape(args.nf_repeats, args.folds), axis=1)
        m, c = float(np.nanmean(rep)), _ci(rep)
        rows_out.append({"model": "NeuroFormer (3-modality)", "auc": m, "ci95": c,
                         "protocol": f"{args.folds}x{args.nf_repeats}"})
        print(f"\n  NeuroFormer 3-modality ROC-AUC = {m:.3f} +/- {c:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(args.out, index=False)
    print(f"\n[done] wrote {args.out}  (original paper headline was 0.912)")


if __name__ == "__main__":
    main()
