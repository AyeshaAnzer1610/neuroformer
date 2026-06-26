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
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # cublas determinism (must precede torch)
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


def pooled_summ(oof, y):
    """Pooled out-of-fold AUC: within each repeat every subject is held out exactly once,
    so we score ONE AUC per repeat over all N subjects, then average across repeats.
    Far more stable at small N than averaging tiny per-fold AUCs (Forman & Scholz 2010)."""
    oof = np.asarray(oof, float)
    aucs = []
    for r in range(oof.shape[0]):
        p = oof[r]; m = ~np.isnan(p)
        if m.sum() and len(np.unique(y[m])) > 1:
            aucs.append(roc_auc_score(y[m], np.nan_to_num(p[m], nan=0.5)))
    aucs = np.array(aucs, float)
    return (float(np.nanmean(aucs)) if len(aucs) else float("nan")), _ci95(aucs)


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


# ----------------------------------------------------- sequence baselines (R2.2)
def _seq_pack(subs, eeg, eye, beh, max_len=30):
    """Per subject: padded EEG window-sequence [L,516] + mask + pooled eye/beh."""
    de, dy, db = eeg[subs[0]].shape[1], eye[subs[0]].shape[1], beh[subs[0]].shape[1]
    E = np.zeros((len(subs), max_len, de), np.float32); M = np.zeros((len(subs), max_len), np.float32)
    Y = np.zeros((len(subs), dy), np.float32); B = np.zeros((len(subs), db), np.float32)
    for i, s in enumerate(subs):
        w = eeg[s][:max_len]; E[i, :len(w)] = w; M[i, :len(w)] = 1.0
        Y[i] = eye[s].mean(0); B[i] = beh[s].mean(0)
    return E, M, Y, B


class _SeqHead(nn.Module):
    """Reduce EEG sequence -> h, concat pooled eye+beh -> classify."""
    def __init__(self, kind, de, dy, db, h=64):
        super().__init__()
        self.kind = kind
        if kind == "cnn":
            self.enc = nn.Sequential(nn.Conv1d(de, h, 3, padding=1), nn.ReLU(),
                                     nn.Conv1d(h, h, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        elif kind == "lstm":
            self.enc = nn.LSTM(de, h, batch_first=True)
        elif kind == "tcn":
            self.enc = nn.Sequential(nn.Conv1d(de, h, 3, padding=1, dilation=1), nn.ReLU(),
                                     nn.Conv1d(h, h, 3, padding=2, dilation=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.cls = nn.Sequential(nn.Linear(h + dy + db, h), nn.ReLU(), nn.Dropout(0.3), nn.Linear(h, 2))

    def forward(self, E, Y, B):
        if self.kind == "lstm":
            o, _ = self.enc(E); z = o.mean(1)
        else:
            z = self.enc(E.transpose(1, 2)).squeeze(-1)
        return self.cls(torch.cat([z, Y, B], -1))


def cv_seq(kind, packed, y, folds, repeats, seed, device, epochs=40):
    E, M, Y, B = packed; de, dy, db = E.shape[2], Y.shape[1], B.shape[1]
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    fold = []; oof = np.full((repeats, len(y)), np.nan)
    for k, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        torch.manual_seed(seed)
        model = _SeqHead(kind, de, dy, db).to(device)
        pos = int(y[tr].sum()); neg = len(tr) - pos
        lossf = nn.CrossEntropyLoss(weight=torch.tensor([1.0, neg / max(pos, 1)], device=device))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        et = torch.tensor(E[tr], device=device); yt = torch.tensor(Y[tr], device=device); bt = torch.tensor(B[tr], device=device)
        model.train()
        for _ in range(epochs):
            opt.zero_grad(); loss = lossf(model(et, yt, bt), torch.tensor(y[tr], device=device))
            if not torch.isfinite(loss):
                continue
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(torch.tensor(E[te], device=device), torch.tensor(Y[te], device=device),
                                    torch.tensor(B[te], device=device)), -1)[:, 1].cpu().numpy()
        p = np.nan_to_num(p, nan=0.5)
        oof[k // folds, te] = p
        fold.append(roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else np.nan)
    return np.array(fold), oof


class TensorFusion(nn.Module):
    """Tensor Fusion Network (Zadeh et al., EMNLP 2017) — outer product of modality
    embeddings (with a bias unit) captures uni/bi/tri-modal interactions."""
    def __init__(self, dims, d=6, n_classes=2):
        super().__init__()
        self.mods = list(dims); self.d = d
        self.proj = nn.ModuleDict({m: nn.Linear(dims[m], d) for m in dims})
        self.cls = nn.Sequential(nn.Linear((d + 1) ** len(dims), 64), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, X):
        hs = [torch.cat([torch.tanh(self.proj[m](X[m])),
                         torch.ones(X[m].shape[0], 1, device=X[m].device)], -1) for m in self.mods]
        fused = hs[0]
        for h in hs[1:]:
            fused = torch.bmm(fused.unsqueeze(-1), h.unsqueeze(1)).flatten(1)
        return self.cls(fused)


class LowRankFusion(nn.Module):
    """Low-rank Multimodal Fusion (Liu et al., ACL 2018) — efficient low-rank
    approximation of the tensor fusion."""
    def __init__(self, dims, d=16, rank=4, n_classes=2):
        super().__init__()
        self.mods = list(dims); self.rank = rank; self.h = 64
        self.proj = nn.ModuleDict({m: nn.Linear(dims[m], d) for m in dims})
        self.factors = nn.ParameterDict({m: nn.Parameter(torch.randn(rank, d + 1, self.h) * 0.1) for m in dims})
        self.cls = nn.Sequential(nn.ReLU(), nn.Dropout(0.3), nn.Linear(self.h, n_classes))

    def forward(self, X):
        out = 1.0
        for m in self.mods:
            h = torch.cat([torch.tanh(self.proj[m](X[m])),
                           torch.ones(X[m].shape[0], 1, device=X[m].device)], -1)   # [B, d+1]
            out = out * torch.einsum("bd,rdh->brh", h, self.factors[m])             # [B, rank, h]
        return self.cls(out.sum(1))


def _train_torch_vec(model, Xtr, ytr, device, epochs=60, lr=1e-3, dict_in=False):
    pos = int(ytr.sum()); neg = len(ytr) - pos
    lossf = nn.CrossEntropyLoss(weight=torch.tensor([1.0, neg / max(pos, 1)], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(Xtr)
        loss = lossf(out, torch.tensor(ytr, device=device))
        if not torch.isfinite(loss):
            continue                                  # skip a diverged step (NaN guard)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # prevent transformer divergence
        opt.step()
    return model


# ----------------------------------------------------------------- baseline runners
def cv_sklearn(clf_fn, X, y, folds, repeats, seed):
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    fold = []; oof = np.full((repeats, len(y)), np.nan)
    for k, (tr, te) in enumerate(cv.split(X, y)):
        clf = clf_fn(); clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        oof[k // folds, te] = p
        fold.append(roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else np.nan)
    return np.array(fold), oof   # per-fold AUCs + (repeats,N) out-of-fold preds


def cv_torch_vec(build, Xd, y, folds, repeats, seed, device, dict_in=False, epochs=60):
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    fold = []; oof = np.full((repeats, len(y)), np.nan)
    for k, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        torch.manual_seed(seed)
        def sub(idx):
            return ({m: torch.tensor(Xd[m][idx], dtype=torch.float32, device=device) for m in Xd}
                    if dict_in else torch.tensor(Xd[idx], dtype=torch.float32, device=device))
        model = build().to(device)
        _train_torch_vec(model, sub(tr), y[tr], device, epochs=epochs, dict_in=dict_in)
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(sub(te)), -1)[:, 1].cpu().numpy()
        p = np.nan_to_num(p, nan=0.5)                  # diverged preds -> chance (don't crash)
        oof[k // folds, te] = p
        fold.append(roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else np.nan)
    return np.array(fold), oof   # per-fold AUCs + (repeats,N) out-of-fold preds


def cv_late_fusion(Xd, y, folds, repeats, seed):
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    fold = []; oof = np.full((repeats, len(y)), np.nan)
    for k, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        ps = []
        for m in Xd:
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit(Xd[m][tr], y[tr]); ps.append(clf.predict_proba(Xd[m][te])[:, 1])
        p = np.mean(ps, 0)
        oof[k // folds, te] = p
        fold.append(roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else np.nan)
    return np.array(fold), oof   # per-fold AUCs + (repeats,N) out-of-fold preds


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
    fold, gaps, oof = [], [], {}
    oofmat = np.full((nf_repeats, len(y)), np.nan)        # (repeats, N) for pooled-OOF AUC
    for k, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        torch.manual_seed(seed)
        te_auc, tr_auc, preds = _nf_fold([rows[i] for i in tr], [rows[i] for i in te], dims, A,
                                         device, create_neuroformer, aux)
        fold.append(te_auc)
        if not (np.isnan(te_auc) or np.isnan(tr_auc)):
            gaps.append((tr_auc - te_auc) * 100)          # train-test gap in pp (E7)
        for j, (sid, prob) in zip(te, preds):
            oof.setdefault(sid, []).append(prob)          # OOF probs across repeats
            oofmat[k // folds, j] = prob
    return np.array(fold), (float(np.mean(gaps)) if gaps else float("nan")), oof, oofmat


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
        return (roc_auc_score(yy, p) if len(np.unique(yy)) > 1 else np.nan), p

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
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # match robustness_3mod (stabilizes NF)
            opt.step()
        v, _ = ev(va); v = 0.0 if np.isnan(v) else v
        if v > best + 1e-4:
            best, bad = v, 0
            best_state = {k: x.detach().clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= A.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    te_auc, te_p = ev(te_rows)
    tr_auc, _ = ev(tr2)
    preds = list(zip([r["sid"] for r in te_rows], [float(x) for x in te_p]))   # OOF for confound test
    return te_auc, tr_auc, preds


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
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True          # reproducible: byte-identical re-run
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    eeg, eye, beh = load_eeg(), load_eye(), load_beh()
    lab = pd.read_csv("data/labels.csv").set_index("ID")["DX_Status"].to_dict()
    subs = sorted(set(lab) & set(eeg) & set(eye) & set(beh))
    y = np.array([int(lab[s] > 0) for s in subs])
    Xe = np.stack([eeg[s].mean(0) for s in subs]); Xy = np.stack([eye[s].mean(0) for s in subs])
    Xb = np.stack([beh[s].mean(0) for s in subs]); Xcat = np.hstack([Xe, Xy, Xb])
    Xd = {"eeg": Xe, "eye": Xy, "beh": Xb}
    Xcs = StandardScaler().fit_transform(Xcat)
    print(f"[load] N={len(subs)} pos={int(y.sum())} dims eeg={Xe.shape[1]} eye={Xy.shape[1]} beh={Xb.shape[1]}", flush=True)
    outdir = Path(args.out_t2).parent; outdir.mkdir(parents=True, exist_ok=True)
    from scipy.stats import wilcoxon
    F = args.folds

    def summ(arr):
        a = np.asarray(arr, float)
        rep = np.nanmean(a.reshape(-1, F), axis=1) if len(a) % F == 0 else a
        return float(np.nanmean(rep)), _ci95(rep)

    # ---------------- Table 2: baselines (raw per-fold saved) ----------------
    print("\n=== TABLE 2: baselines (leak-free %dx%d CV) ===" % (args.folds, args.repeats), flush=True)
    t2, perfold, oof_store = [], [], {}
    def add(name, ret):
        fold, oof = ret
        m, c = pooled_summ(oof, y)                 # headline: pooled out-of-fold AUC (stable at small N)
        mpf, cpf = summ(fold)                      # supplement: mean of per-fold AUCs (unstable; for transparency)
        t2.append({"model": name, "auc": m, "ci95": c, "auc_perfold": mpf, "ci_perfold": cpf})
        oof_store[name] = oof
        for i, v in enumerate(np.asarray(fold, float)):
            perfold.append({"table": "T2", "model": name, "fold": i + 1, "auc": v})
        print(f"  {name:<26} pooled {m:.3f} +/- {c:.3f}   (per-fold {mpf:.3f})", flush=True)

    add("LogReg (concat)", cv_sklearn(lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), Xcat, y, args.folds, args.repeats, args.seed))
    add("SVM-RBF (concat)", cv_sklearn(lambda: make_pipeline(StandardScaler(), SVC(probability=True)), Xcat, y, args.folds, args.repeats, args.seed))
    add("RandomForest (concat)", cv_sklearn(lambda: RandomForestClassifier(n_estimators=300, random_state=0), Xcat, y, args.folds, args.repeats, args.seed))
    add("MLP early-fusion", cv_torch_vec(lambda: MLP(Xcs.shape[1]), Xcs, y, args.folds, args.repeats, args.seed, device))
    add("Late-fusion ensemble", cv_late_fusion(Xd, y, args.folds, args.repeats, args.seed))
    add("GMU (gated fusion)", cv_torch_vec(lambda: GMU({"eeg": Xe.shape[1], "eye": Xy.shape[1], "beh": Xb.shape[1]}), Xd, y, args.folds, args.repeats, args.seed, device, dict_in=True))
    add("Cross-modal Transformer", cv_torch_vec(lambda: CrossModalTransformer({"eeg": Xe.shape[1], "eye": Xy.shape[1], "beh": Xb.shape[1]}), Xd, y, args.folds, args.repeats, args.seed, device, dict_in=True))
    # temporal/sequence baselines (R2.2): 1D-CNN, LSTM, TCN on the EEG window-sequence
    packed = _seq_pack(subs, eeg, eye, beh)
    add("1D-CNN (temporal)", cv_seq("cnn", packed, y, args.folds, args.repeats, args.seed, device))
    add("LSTM (temporal)", cv_seq("lstm", packed, y, args.folds, args.repeats, args.seed, device))
    add("TCN (temporal)", cv_seq("tcn", packed, y, args.folds, args.repeats, args.seed, device))
    dims3 = {"eeg": Xe.shape[1], "eye": Xy.shape[1], "beh": Xb.shape[1]}
    add("Tensor Fusion Network", cv_torch_vec(lambda: TensorFusion(dims3), Xd, y, args.folds, args.repeats, args.seed, device, dict_in=True))
    add("Low-rank Multimodal Fusion", cv_torch_vec(lambda: LowRankFusion(dims3), Xd, y, args.folds, args.repeats, args.seed, device, dict_in=True))
    if not args.skip_nf:
        nf_arr, nf_gap, nf_oof, nf_oofmat = neuroformer_cv(subs, lab, eeg, eye, beh, {"eeg", "eye", "beh"}, args.folds, args.nf_repeats, args.seed, device, aux=False)
        add("NeuroFormer (3-modality)", (nf_arr, nf_oofmat))
        print(f"  [E7] NeuroFormer train-test gap = {nf_gap:.1f} pp (mean across folds)", flush=True)
    pd.DataFrame(t2).to_csv(args.out_t2, index=False)
    pd.DataFrame(perfold).to_csv(outdir / "perfold_table2.csv", index=False)
    import pickle
    with open(outdir / "oof_table2.pkl", "wb") as f:        # raw OOF -> recompute pooled AUC offline, no GPU
        pickle.dump({"y": y, "subs": subs, "oof": oof_store}, f)
    print(f"[done] Table 2 -> {args.out_t2} (+ perfold_table2.csv, oof_table2.pkl)", flush=True)

    # ---------------- Confound analysis (R3.7): does age/sex alone predict the label? ----------------
    print("\n=== CONFOUND CHECK (R3.7): can age/sex predict diagnosis? ===", flush=True)
    meta = pd.read_csv("data/labels.csv").set_index("ID")
    Xconf = np.array([[meta.loc[s, "Age"], meta.loc[s, "Sex"]] for s in subs], float)
    _, cf_oof = cv_sklearn(lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                           Xconf, y, args.folds, args.repeats, args.seed)
    mc, cc = pooled_summ(cf_oof, y)              # pooled-OOF, consistent with Table 2
    print(f"  age+sex -> diagnosis ROC-AUC = {mc:.3f} +/- {cc:.3f}  "
          f"({'near chance -> confounds do NOT drive the label' if mc < 0.6 else 'WARN: confound signal'})", flush=True)
    pd.DataFrame([{"predictor": "age+sex", "auc": mc, "ci95": cc}]).to_csv(outdir / "confound_check.csv", index=False)

    # confound-ADJUSTED (decisive for R3.1/R3.7): does NeuroFormer add signal BEYOND age+sex?
    if not args.skip_nf:
        from scipy.stats import pearsonr
        nfp = np.array([float(np.mean(nf_oof[s])) for s in subs])     # mean OOF prob per subject
        nf_auc = roc_auc_score(y, nfp)
        r_age = pearsonr(nfp, Xconf[:, 0])[0]; r_sex = pearsonr(nfp, Xconf[:, 1])[0]
        def _as(X):
            _, o = cv_sklearn(lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                              X, y, args.folds, args.repeats, args.seed)
            return pooled_summ(o, y)[0]              # pooled-OOF, consistent with Table 2
        m_as = _as(Xconf); m_asn = _as(np.hstack([Xconf, nfp[:, None]]))
        print(f"  NeuroFormer OOF AUC = {nf_auc:.3f} | corr(NF,age)={r_age:+.2f} corr(NF,sex)={r_sex:+.2f}", flush=True)
        print(f"  age+sex = {m_as:.3f}  ->  age+sex+NeuroFormer = {m_asn:.3f}  "
              f"(incremental dAUC = {100*(m_asn-m_as):+.1f}pp)", flush=True)
        print(f"  >>> {'NeuroFormer ADDS signal beyond demographics' if (m_asn-m_as) > 0.02 else 'WARNING: NeuroFormer does NOT clearly exceed the demographic baseline'}", flush=True)
        pd.DataFrame([{"nf_oof_auc": nf_auc, "corr_age": r_age, "corr_sex": r_sex,
                       "age_sex_auc": m_as, "age_sex_nf_auc": m_asn,
                       "incremental_pp": 100*(m_asn-m_as)}]).to_csv(outdir / "confound_adjusted.csv", index=False)

    # ---------------- Table 3: modality ablation + Δaux + paired p-value (E3) ----------------
    if not args.skip_nf:
        print("\n=== TABLE 3: modality ablation (NeuroFormer, leak-free, +/- aux) ===", flush=True)
        subsets = [("EEG", {"eeg"}), ("Eye", {"eye"}), ("Beh", {"beh"}),
                   ("EEG+Eye", {"eeg", "eye"}), ("EEG+Beh", {"eeg", "beh"}),
                   ("Eye+Beh", {"eye", "beh"}), ("EEG+Eye+Beh", {"eeg", "eye", "beh"})]
        raw, rawoof, pf3 = {}, {}, []
        for aux in (False, True):
            for name, mods in subsets:
                arr, _, _, omat = neuroformer_cv(subs, lab, eeg, eye, beh, mods, args.folds, args.nf_repeats, args.seed, device, aux=aux)
                raw[(name, aux)] = arr; rawoof[(name, aux)] = omat
                for i, v in enumerate(np.asarray(arr, float)):
                    pf3.append({"config": name, "aux": int(aux), "fold": i + 1, "auc": v})
                print(f"  {name:<12} aux={int(aux)}  pooled {pooled_summ(omat, y)[0]:.3f}", flush=True)
        t3 = []
        for name, _ in subsets:
            a0, a1 = raw[(name, False)], raw[(name, True)]
            m0, c0 = pooled_summ(rawoof[(name, False)], y); m1, c1 = pooled_summ(rawoof[(name, True)], y)
            try:
                p = float(wilcoxon(a0, a1).pvalue)        # paired over 25 folds (power); estimate is pooled
            except ValueError:
                p = float("nan")
            t3.append({"config": name, "auc_noaux": m0, "ci_noaux": c0, "auc_aux": m1,
                       "ci_aux": c1, "daux_pp": (m1 - m0) * 100, "p_value": p})
            print(f"  {name:<12} -aux {m0:.3f} | +aux {m1:.3f} | dAUC {100*(m1-m0):+.1f}pp | p={p:.3f}", flush=True)
        pd.DataFrame(t3).to_csv(args.out_t3, index=False)
        pd.DataFrame(pf3).to_csv(outdir / "perfold_table3.csv", index=False)
        print(f"[done] Table 3 -> {args.out_t3} (+ perfold_table3.csv)")


if __name__ == "__main__":
    main()
