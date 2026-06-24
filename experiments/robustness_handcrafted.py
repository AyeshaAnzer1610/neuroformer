#!/usr/bin/env python
"""
R3.3 robustness re-run of the ORIGINAL-style hand-crafted multimodal NeuroFormer,
under strict leak-free, subject-level, repeated stratified CV.

Feeds the SAME NeuroFormer architecture (src/neuroformer.py) the ORIGINAL inputs:
- EEG : hand-crafted features  data/processed_handcrafted/<sid>_handcrafted.npz
        (key "features", shape [n_windows, 516])  -> per-subject sequence
- eye : data/eye_features.csv  (11 feats x ~N task blocks/subject) -> sequence
(Behaviour omitted: raw RT/accuracy not available locally; it lives in MIPDB raw on
Nibi. EEG+eye are the two modalities carrying the original eye-dominant story.)

Reports, with identical subject-level RepeatedStratifiedKFold (no leakage):
  1. logreg baselines (EEG-alone, eye-alone, naive concat) -- fast honest read
  2. the NeuroFormer transformer (hierarchical cross-modal fusion)
per-repeat ROC-AUC mean +/- 95% CI. This is the faithful answer to R3.3 / R3.8 / R3.9.

    python experiments/robustness_handcrafted.py --repeats 10 --nf-repeats 5
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SID = re.compile(r"(A\d{8})")


def load_eeg(d="data/processed_handcrafted"):
    out = {}
    for f in sorted(Path(d).glob("*_handcrafted.npz")):
        sid = SID.search(f.name).group(1)
        a = np.load(f)["features"].astype(np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        if a.ndim == 2 and a.shape[0] > 0:
            out[sid] = a
    return out


def load_eye(csv="data/eye_features.csv"):
    df = pd.read_csv(csv)
    feats = [c for c in df.columns if c != "file_name"]
    df["sid"] = df["file_name"].map(lambda s: (SID.search(str(s)) or [None])[0]
                                    if SID.search(str(s)) else None)
    out = {}
    for sid, g in df.dropna(subset=["sid"]).groupby("sid"):
        a = np.nan_to_num(g[feats].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if a.shape[0] > 0:
            out[sid] = a
    return out


def _ci(a):
    a = np.asarray(a, float)
    return 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")


def per_repeat_auc_logreg(X, y, folds, repeats, seed):
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    n_per = folds
    fold_aucs = []
    for tr, te in cv.split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        fold_aucs.append(roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else np.nan)
    fold_aucs = np.array(fold_aucs).reshape(repeats, n_per)
    rep = np.nanmean(fold_aucs, axis=1)            # per-repeat mean over folds
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10, help="logreg repeats")
    ap.add_argument("--nf-repeats", type=int, default=5, help="NeuroFormer repeats")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--max-epochs", type=int, default=200, help="NF early-stop cap")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--skip-nf", action="store_true")
    ap.add_argument("--out", default="results/robustness_handcrafted.csv")
    args = ap.parse_args()
    np.random.seed(args.seed)

    eeg, eye = load_eeg(), load_eye()
    lab = pd.read_csv("data/labels.csv").set_index("ID")["DX_Status"].to_dict()
    subs = sorted(set(lab) & set(eeg) & set(eye))      # complete-case (both modalities)
    y = np.array([int(lab[s] > 0) for s in subs])
    print(f"[load] complete-case N={len(subs)}  pos={int(y.sum())} neg={len(y)-int(y.sum())}  "
          f"| EEG dim={next(iter(eeg.values())).shape[1]}  eye dim={next(iter(eye.values())).shape[1]}")

    # per-subject pooled vectors (mean over windows/blocks) for logreg
    Xeeg = np.stack([eeg[s].mean(0) for s in subs])
    Xeye = np.stack([eye[s].mean(0) for s in subs])
    Xcat = np.hstack([Xeeg, Xeye])

    rows = []
    print(f"\n=== logreg baselines, {args.folds}x{args.repeats} leak-free subject-level CV ===")
    for name, X in [("EEG-alone (handcrafted)", Xeeg), ("eye-alone", Xeye),
                    ("concat (EEG+eye)", Xcat)]:
        rep = per_repeat_auc_logreg(X, y, args.folds, args.repeats, args.seed)
        m, c = float(np.nanmean(rep)), _ci(rep)
        rows.append({"model": name, "auc": m, "ci95": c, "protocol": f"{args.folds}x{args.repeats}"})
        print(f"  {name:<26} ROC-AUC = {m:.3f} +/- {c:.3f}")

    if not args.skip_nf:
        import torch
        import torch.nn as nn
        from neuroformer import create_neuroformer
        from train_neuroformer import make_batch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        nf_rows = [{"sid": s, "y": int(lab[s] > 0), "eeg": eeg[s], "eye": eye[s]} for s in subs]
        dims = {"eeg": Xeeg.shape[1], "eye": Xeye.shape[1]}

        class A:  # arg shim for make_batch
            d_model = 128; enc_layers = 2; fus_layers = 2; heads = 8; dropout = 0.3
            max_len = 64; moddrop = 0.0; lr = 1e-4; batch = 16

        def _eval(model, rows):
            model.eval()
            with torch.no_grad():
                b = make_batch(rows, dims, A.max_len, train=False)
                out = model(b["eeg"].to(device), b["eye"].to(device),
                            b["eeg_mask"].to(device), b["eye_mask"].to(device))
                p = torch.softmax(out["logits"], -1)[:, 1].cpu().numpy()
            yy = np.array([r["y"] for r in rows])
            return (roc_auc_score(yy, p) if len(np.unique(yy)) > 1 else np.nan), p

        def run_fold_es(tr_rows, te_rows):
            """Leak-free early stopping: carve a subject-disjoint val set from the
            training fold, restore best-val weights, then score test."""
            ytr = np.array([r["y"] for r in tr_rows])
            from sklearn.model_selection import train_test_split
            idx = np.arange(len(tr_rows))
            tri, vai = train_test_split(idx, test_size=args.val_frac, stratify=ytr,
                                        random_state=args.seed)
            tr2 = [tr_rows[i] for i in tri]; va = [tr_rows[i] for i in vai]
            model = create_neuroformer({
                "eeg_input_dim": dims["eeg"], "eyetracking_input_dim": dims["eye"],
                "d_model": A.d_model, "n_encoder_layers": A.enc_layers,
                "n_fusion_layers": A.fus_layers, "n_heads": A.heads, "dropout": A.dropout,
                "n_classes": 2, "use_auxiliary_loss": False}).to(device)
            pos = sum(r["y"] for r in tr2); neg = len(tr2) - pos
            w = torch.tensor([1.0, neg / max(pos, 1)], device=device)
            lossf = nn.CrossEntropyLoss(weight=w)
            opt = torch.optim.Adam(model.parameters(), lr=A.lr, weight_decay=1e-4)
            best_auc, best_state, bad = -1.0, None, 0
            for ep in range(args.max_epochs):
                model.train(); np.random.shuffle(tr2)
                for i in range(0, len(tr2), A.batch):
                    b = make_batch(tr2[i:i + A.batch], dims, A.max_len, moddrop=0.0, train=True)
                    opt.zero_grad()
                    out = model(b["eeg"].to(device), b["eye"].to(device),
                                b["eeg_mask"].to(device), b["eye_mask"].to(device))
                    loss = lossf(out["logits"], b["y"].to(device))
                    loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                vauc, _ = _eval(model, va)
                if np.isnan(vauc):
                    vauc = 0.0
                if vauc > best_auc + 1e-4:
                    best_auc, bad = vauc, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= args.patience:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)
            return _eval(model, te_rows)[0]

        cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.nf_repeats,
                                     random_state=args.seed)
        print(f"\n=== NeuroFormer (hierarchical fusion), {args.folds}x{args.nf_repeats} "
              f"leak-free CV, early-stop (max {args.max_epochs}ep, patience {args.patience}, "
              f"val {args.val_frac:.0%}, device={device}) ===")
        fold_aucs = []
        for k, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y), 1):
            torch.manual_seed(args.seed + k); np.random.seed(args.seed + k)
            auc = run_fold_es([nf_rows[i] for i in tr], [nf_rows[i] for i in te])
            fold_aucs.append(auc)
            print(f"  [fold {k}/{args.folds*args.nf_repeats}] AUC={auc:.3f}", flush=True)
        fa = np.array(fold_aucs).reshape(args.nf_repeats, args.folds)
        rep = np.nanmean(fa, axis=1)
        m, c = float(np.nanmean(rep)), _ci(rep)
        rows.append({"model": "NeuroFormer (handcrafted EEG+eye)", "auc": m, "ci95": c,
                     "protocol": f"{args.folds}x{args.nf_repeats}"})
        print(f"\n  NeuroFormer ROC-AUC = {m:.3f} +/- {c:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\n[done] wrote {args.out}  (original paper headline was 0.912)")


if __name__ == "__main__":
    main()
