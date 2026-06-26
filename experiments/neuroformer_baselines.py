#!/usr/bin/env python
"""
HONEST reconstruction of NeuroFormer Table 2 (baselines) and Table 3 (modality
ablation), all under ONE leak-free, subject-level, repeated stratified CV protocol on
the 3 hand-crafted modalities (EEG 516-d, eye 11-d, behaviour 6-d). Every number here is
produced by THIS script -- the corrected, reproducible replacement for the manuscript's
Tables 2-3 (which were not reproducible).

    python experiments/neuroformer_baselines.py --repeats 5 --nf-repeats 5 \
        --out-t2 results/table2_baselines.csv --out-t3 results/table3_ablation.csv

Notes
- Classical/MLP/fusion baselines use per-subject mean-pooled modality vectors.
- Sequence baselines (1D-CNN, LSTM) run on the EEG window sequence, with pooled eye+beh
  concatenated before the classifier.
- NeuroFormer uses the real 3-modality architecture (src/neuroformer.py) with early-stop.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robustness_3mod import load_eeg, load_eye, load_beh, _ci          # noqa: E402


def _ci95(rep):
    rep = np.asarray(rep, float)
    return 1.96 * rep.std(ddof=1) / np.sqrt(len(rep)) if len(rep) > 1 else float("nan")


# ----------------------------------------------------------------- torch baselines
class MLP(nn.Module):
    def __init__(self, d, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.3), nn.Linear(h, 2))

    def forward(self, x):
        return self.net(x)


class GMU(nn.Module):
    """Gated Multimodal Unit over pooled modality vectors {eeg, eye, beh}."""
    def __init__(self, dims, h=128):
        super().__init__()
        self.mods = list(dims)
        self.hz = nn.ModuleDict({m: nn.Linear(dims[m], h) for m in dims})
        self.gz = nn.ModuleDict({m: nn.Linear(dims[m], h) for m in dims})
        self.cls = nn.Linear(h, 2)

    def forward(self, X):  # X: dict modality->[B,d]
        feats = torch.stack([torch.tanh(self.hz[m](X[m])) for m in self.mods], 0)
        gates = torch.softmax(torch.stack([self.gz[m](X[m]) for m in self.mods], 0), 0)
        return self.cls((gates * feats).sum(0))


class CrossModalTransformer(nn.Module):
    """Single-stage 'Vanilla Transformer' / cross-modal attention over modality tokens."""
    def __init__(self, dims, h=128, layers=2, heads=4):
        super().__init__()
        self.mods = list(dims)
        self.proj = nn.ModuleDict({m: nn.Linear(dims[m], h) for m in dims})
        enc = nn.TransformerEncoderLayer(h, heads, 2 * h, dropout=0.3, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.cls = nn.Linear(h, 2)

    def forward(self, X):
        toks = torch.stack([self.proj[m](X[m]) for m in self.mods], 1)   # [B, M, h]
        return self.cls(self.tr(toks).mean(1))


def _train_torch_vec(model, Xtr, ytr, device, epochs=60, lr=1e-3, dict_in=False):
    pos = int(ytr.sum()); neg = len(ytr) - pos
    lossf = nn.CrossEntropyLoss(weight=torch.tensor([1.0, neg / max(pos, 1)], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(Xtr)
        loss = lossf(out, torch.tensor(ytr, device=device))
        loss.backward(); opt.step()
    return model


# ----------------------------------------------------------------- baseline runners
def cv_sklearn(clf_fn, X, y, folds, repeats, seed):
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    fold = []
    for tr, te in cv.split(X, y):
        clf = clf_fn(); clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        fold.append(roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else np.nan)
    return np.nanmean(np.array(fold).reshape(repeats, folds), axis=1)


def cv_torch_vec(build, Xd, y, folds, repeats, seed, device, dict_in=False, epochs=60):
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    fold = []
    for tr, te in cv.split(np.zeros(len(y)), y):
        torch.manual_seed(seed)
        def sub(idx):
            return ({m: torch.tensor(Xd[m][idx], dtype=torch.float32, device=device) for m in Xd}
                    if dict_in else torch.tensor(Xd[idx], dtype=torch.float32, device=device))
        model = build().to(device)
        _train_torch_vec(model, sub(tr), y[tr], device, epochs=epochs, dict_in=dict_in)
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(sub(te)), -1)[:, 1].cpu().numpy()
        fold.append(roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else np.nan)
    return np.nanmean(np.array(fold).reshape(repeats, folds), axis=1)


def cv_late_fusion(Xd, y, folds, repeats, seed):
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    fold = []
    for tr, te in cv.split(np.zeros(len(y)), y):
        ps = []
        for m in Xd:
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit(Xd[m][tr], y[tr]); ps.append(clf.predict_proba(Xd[m][te])[:, 1])
        p = np.mean(ps, 0)
        fold.append(roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else np.nan)
    return np.nanmean(np.array(fold).reshape(repeats, folds), axis=1)


def neuroformer_cv(subs, lab, eeg, eye, beh, mods, folds, nf_repeats, seed, device, aux):
    """3-modality NeuroFormer on a chosen modality subset (mods), early-stop, leak-free."""
    from neuroformer import create_neuroformer
    from robustness_3mod import make_batch3, _pad  # reuse padded batch
    y = np.array([int(lab[s] > 0) for s in subs])
    dims = {"eeg": eeg[subs[0]].shape[1], "eye": eye[subs[0]].shape[1], "beh": beh[subs[0]].shape[1]}
    rows = [{"sid": s, "y": int(lab[s] > 0),
             "eeg": eeg[s] if "eeg" in mods else np.zeros((1, dims["eeg"]), np.float32),
             "eye": eye[s] if "eye" in mods else np.zeros((1, dims["eye"]), np.float32),
             "beh": beh[s] if "beh" in mods else np.zeros((1, dims["beh"]), np.float32)} for s in subs]

    class A:
        max_len = 64; batch = 16; lr = 1e-4; max_epochs = 200; patience = 20; val_frac = 0.2
    A.seed = seed
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=nf_repeats, random_state=seed)
    fold = []
    for tr, te in cv.split(np.zeros(len(y)), y):
        torch.manual_seed(seed)
        auc = _nf_fold([rows[i] for i in tr], [rows[i] for i in te], dims, A, device,
                       create_neuroformer, aux)
        fold.append(auc)
    rep = np.nanmean(np.array(fold).reshape(nf_repeats, folds), axis=1)
    return rep


def _nf_fold(tr_rows, te_rows, dims, A, device, create, aux):
    from robustness_3mod import make_batch3
    ytr = np.array([r["y"] for r in tr_rows])
    tri, vai = train_test_split(np.arange(len(tr_rows)), test_size=A.val_frac, stratify=ytr,
                                random_state=A.seed)
    tr2 = [tr_rows[i] for i in tri]; va = [tr_rows[i] for i in vai]
    model = create({"eeg_input_dim": dims["eeg"], "eyetracking_input_dim": dims["eye"],
                    "behavioral_input_dim": dims["beh"], "d_model": 128, "n_encoder_layers": 2,
                    "n_fusion_layers": 2, "n_heads": 8, "dropout": 0.3, "n_classes": 2,
                    "use_auxiliary_loss": aux}).to(device)
    pos = sum(r["y"] for r in tr2); neg = len(tr2) - pos
    lossf = nn.CrossEntropyLoss(weight=torch.tensor([1.0, neg / max(pos, 1)], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=A.lr, weight_decay=1e-4)

    def ev(rows):
        model.eval()
        with torch.no_grad():
            b = make_batch3(rows, dims, A.max_len, torch)
            o = model(b["eeg"].to(device), b["eye"].to(device), b["beh"].to(device),
                      b["eeg_m"].to(device), b["eye_m"].to(device), b["beh_m"].to(device))
            lo = o["logits"] if isinstance(o, dict) else o
            p = torch.softmax(lo, -1)[:, 1].cpu().numpy()
        yy = np.array([r["y"] for r in rows])
        return roc_auc_score(yy, p) if len(np.unique(yy)) > 1 else np.nan

    best, bad, best_state = -1.0, 0, None
    for _ in range(A.max_epochs):
        model.train(); np.random.shuffle(tr2)
        for i in range(0, len(tr2), A.batch):
            b = make_batch3(tr2[i:i + A.batch], dims, A.max_len, torch)
            opt.zero_grad()
            o = model(b["eeg"].to(device), b["eye"].to(device), b["beh"].to(device),
                      b["eeg_m"].to(device), b["eye_m"].to(device), b["beh_m"].to(device))
            loss = (o["logits"] if isinstance(o, dict) else o)
            l = lossf(loss, b["y"].to(device))
            if isinstance(o, dict) and "recon_loss" in o:
                l = l + o["recon_loss"]
            l.backward(); opt.step()
        v = ev(va); v = 0.0 if np.isnan(v) else v
        if v > best + 1e-4:
            best, bad = v, 0
            best_state = {k: x.detach().clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= A.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return ev(te_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--nf-repeats", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-nf", action="store_true")
    ap.add_argument("--out-t2", default="results/table2_baselines.csv")
    ap.add_argument("--out-t3", default="results/table3_ablation.csv")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed)

    eeg, eye, beh = load_eeg(), load_eye(), load_beh()
    lab = pd.read_csv("data/labels.csv").set_index("ID")["DX_Status"].to_dict()
    subs = sorted(set(lab) & set(eeg) & set(eye) & set(beh))
    y = np.array([int(lab[s] > 0) for s in subs])
    Xe = np.stack([eeg[s].mean(0) for s in subs]); Xy = np.stack([eye[s].mean(0) for s in subs])
    Xb = np.stack([beh[s].mean(0) for s in subs]); Xcat = np.hstack([Xe, Xy, Xb])
    Xd = {"eeg": Xe, "eye": Xy, "beh": Xb}
    Xcs = StandardScaler().fit_transform(Xcat)
    print(f"[load] N={len(subs)} pos={int(y.sum())} dims eeg={Xe.shape[1]} eye={Xy.shape[1]} beh={Xb.shape[1]}", flush=True)
    Path(args.out_t2).parent.mkdir(parents=True, exist_ok=True)

    # ---------------- Table 2: baselines (all leak-free folds x repeats) ----------------
    print("\n=== TABLE 2: baselines (leak-free %dx%d CV) ===" % (args.folds, args.repeats), flush=True)
    t2 = []
    def add(name, rep):
        t2.append({"model": name, "auc": float(np.nanmean(rep)), "ci95": _ci95(rep)})
        print(f"  {name:<26} {np.nanmean(rep):.3f} +/- {_ci95(rep):.3f}", flush=True)

    add("LogReg (concat)", cv_sklearn(lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), Xcat, y, args.folds, args.repeats, args.seed))
    add("SVM-RBF (concat)", cv_sklearn(lambda: make_pipeline(StandardScaler(), SVC(probability=True)), Xcat, y, args.folds, args.repeats, args.seed))
    add("RandomForest (concat)", cv_sklearn(lambda: RandomForestClassifier(n_estimators=300, random_state=0), Xcat, y, args.folds, args.repeats, args.seed))
    add("MLP early-fusion", cv_torch_vec(lambda: MLP(Xcs.shape[1]), Xcs, y, args.folds, args.repeats, args.seed, device))
    add("Late-fusion ensemble", cv_late_fusion(Xd, y, args.folds, args.repeats, args.seed))
    add("GMU (gated fusion)", cv_torch_vec(lambda: GMU({"eeg": Xe.shape[1], "eye": Xy.shape[1], "beh": Xb.shape[1]}), Xd, y, args.folds, args.repeats, args.seed, device, dict_in=True))
    add("Cross-modal Transformer", cv_torch_vec(lambda: CrossModalTransformer({"eeg": Xe.shape[1], "eye": Xy.shape[1], "beh": Xb.shape[1]}), Xd, y, args.folds, args.repeats, args.seed, device, dict_in=True))
    if not args.skip_nf:
        add("NeuroFormer (3-modality)", neuroformer_cv(subs, lab, eeg, eye, beh, {"eeg", "eye", "beh"}, args.folds, args.nf_repeats, args.seed, device, aux=False))
    pd.DataFrame(t2).to_csv(args.out_t2, index=False)
    print(f"[done] Table 2 -> {args.out_t2}", flush=True)

    # ---------------- Table 3: modality ablation (NeuroFormer, +/- aux loss) ----------------
    if not args.skip_nf:
        print("\n=== TABLE 3: modality ablation (NeuroFormer, leak-free) ===", flush=True)
        t3 = []
        subsets = [("EEG", {"eeg"}), ("Eye", {"eye"}), ("Beh", {"beh"}),
                   ("EEG+Eye", {"eeg", "eye"}), ("EEG+Beh", {"eeg", "beh"}),
                   ("Eye+Beh", {"eye", "beh"}), ("EEG+Eye+Beh", {"eeg", "eye", "beh"})]
        for aux in (False, True):
            for name, mods in subsets:
                rep = neuroformer_cv(subs, lab, eeg, eye, beh, mods, args.folds, args.nf_repeats, args.seed, device, aux=aux)
                t3.append({"config": name, "aux_loss": aux, "auc": float(np.nanmean(rep)), "ci95": _ci95(rep)})
                print(f"  {name:<14} aux={int(aux)}  {np.nanmean(rep):.3f} +/- {_ci95(rep):.3f}", flush=True)
        pd.DataFrame(t3).to_csv(args.out_t3, index=False)
        print(f"[done] Table 3 -> {args.out_t3}")


if __name__ == "__main__":
    main()
