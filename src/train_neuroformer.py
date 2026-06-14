#!/usr/bin/env python
"""
Train the (fixed) NeuroFormer leak-free on foundation-model inputs.

Keeps the original NeuroFormer architecture (src/neuroformer.py) but feeds it:
- EEG  : per-window **LaBraM** embeddings  (sequence [n_windows, 200])
- eye  : per-block features                (sequence [n_blocks, 11])
- beh  : per-block features                (sequence [n_blocks, 6])
Subject-level repeated stratified CV (no leakage), class-balanced loss, optional
auxiliary reconstruction, and **modality dropout** (randomly zero a modality per
sample) so one model serves any subset. Reports CV ROC-AUC.

    python src/train_neuroformer.py \
        --eeg-dir $SCRATCH/mh/data/processed \
        --eye-dir $SCRATCH/mh/data/processed_eye \
        --beh-dir $SCRATCH/mh/data/processed_behaviour \
        --labels data/labels.csv --epochs 60
"""
import argparse
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from neuroformer import create_neuroformer

DIMS = {"eeg": ("_labram", "embeddings"), "eye": ("_eye", "features"),
        "behaviour": ("_behaviour", "features")}


def _load_dir(npz_dir, suffix, key):
    from pathlib import Path
    out = {}
    for f in sorted(Path(npz_dir).glob(f"*{suffix}.npz")):
        sid = f.name[: -len(f"{suffix}.npz")]
        a = np.load(f)[key]
        if a.ndim == 2 and a.shape[0] > 0:
            out[sid] = a.astype(np.float32)
    return out


def load_subjects(eeg_dir, eye_dir, beh_dir, labels_csv):
    lab = pd.read_csv(labels_csv).set_index("ID")["DX_Status"].to_dict()
    eeg = _load_dir(eeg_dir, *DIMS["eeg"])
    eye = _load_dir(eye_dir, *DIMS["eye"])
    beh = _load_dir(beh_dir, *DIMS["behaviour"])
    subs = sorted(set(lab) & (set(eeg) | set(eye) | set(beh)))
    rows = []
    for s in subs:
        rows.append({"sid": s, "y": int(lab[s] > 0),
                     "eeg": eeg.get(s), "eye": eye.get(s), "behaviour": beh.get(s)})
    dims = {"eeg": _dim(eeg), "eye": _dim(eye), "behaviour": _dim(beh)}
    print(f"[load] {len(rows)} subjects | dims {dims} | "
          f"present eeg={sum(r['eeg'] is not None for r in rows)} "
          f"eye={sum(r['eye'] is not None for r in rows)} "
          f"beh={sum(r['behaviour'] is not None for r in rows)}")
    return rows, dims


def _dim(d):
    return int(next(iter(d.values())).shape[1]) if d else 0


def _pad(seqs, dim, max_len):
    """List of [n_i, dim] (or None) -> padded [B, max_len, dim] + mask [B,max_len]
    (True = padded). Missing modality -> zeros + fully-real-length-0 (all-pad)."""
    B = len(seqs)
    x = torch.zeros(B, max_len, dim)
    m = torch.ones(B, max_len, dtype=torch.bool)        # True = pad (ignored)
    for i, s in enumerate(seqs):
        if s is None or len(s) == 0:
            x[i, 0] = 0.0; m[i, 0] = False               # 1 zero token so attn != all-masked
            continue
        n = min(len(s), max_len)
        x[i, :n] = torch.from_numpy(np.ascontiguousarray(s[:n]))
        m[i, :n] = False
    return x, m


def make_batch(rows, dims, max_len, moddrop=0.0, train=False):
    out = {}
    for mod in ["eeg", "eye", "behaviour"]:
        seqs = [r[mod] for r in rows]
        if train and moddrop > 0:                        # modality dropout: zero a present modality
            seqs = [None if (s is not None and np.random.rand() < moddrop) else s
                    for s in seqs]
        x, m = _pad(seqs, dims[mod], max_len)
        out[mod] = x; out[f"{mod}_mask"] = m
    out["y"] = torch.tensor([r["y"] for r in rows], dtype=torch.long)
    return out


def run_fold(tr, te, dims, args, device):
    model = create_neuroformer({
        "eeg_input_dim": dims["eeg"], "eyetracking_input_dim": dims["eye"],
        "behavioral_input_dim": dims["behaviour"], "d_model": args.d_model,
        "n_encoder_layers": args.enc_layers, "n_fusion_layers": args.fus_layers,
        "n_heads": args.heads, "dropout": args.dropout, "n_classes": 2,
        "use_auxiliary_loss": False,
    }).to(device)
    pos = sum(r["y"] for r in tr); neg = len(tr) - pos
    w = torch.tensor([1.0, neg / max(pos, 1)], device=device)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model.train()
    for _ in range(args.epochs):
        np.random.shuffle(tr)
        for i in range(0, len(tr), args.batch):
            b = make_batch(tr[i:i + args.batch], dims, args.max_len,
                           moddrop=args.moddrop, train=True)
            opt.zero_grad()
            out = model(b["eeg"].to(device), b["eye"].to(device), b["behaviour"].to(device),
                        b["eeg_mask"].to(device), b["eye_mask"].to(device), b["behaviour_mask"].to(device))
            loss = lossf(out["logits"], b["y"].to(device))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    model.eval()
    with torch.no_grad():
        b = make_batch(te, dims, args.max_len, train=False)
        out = model(b["eeg"].to(device), b["eye"].to(device), b["behaviour"].to(device),
                    b["eeg_mask"].to(device), b["eye_mask"].to(device), b["behaviour_mask"].to(device))
        p = torch.softmax(out["logits"], -1)[:, 1].cpu().numpy()
    y = np.array([r["y"] for r in te])
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-dir", required=True); ap.add_argument("--eye-dir", required=True)
    ap.add_argument("--beh-dir", required=True); ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--enc-layers", type=int, default=2); ap.add_argument("--fus-layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8); ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--max-len", type=int, default=64); ap.add_argument("--moddrop", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/neuroformer_cv.csv")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, torch.get_num_threads()))

    rows, dims = load_subjects(args.eeg_dir, args.eye_dir, args.beh_dir, args.labels)
    y = np.array([r["y"] for r in rows])
    cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed)
    aucs = []
    for k, (tr_i, te_i) in enumerate(cv.split(np.zeros(len(y)), y), 1):
        print(f"[fold {k}/{args.folds*args.repeats}] training "
              f"({len(tr_i)} train / {len(te_i)} test, {args.epochs} epochs)...",
              flush=True)
        auc = run_fold([rows[i] for i in tr_i], [rows[i] for i in te_i], dims, args, device)
        aucs.append(auc)
        print(f"[fold {k}/{args.folds*args.repeats}] AUC={auc:.3f}", flush=True)
    aucs = np.array([a for a in aucs if not np.isnan(a)])
    ci = 1.96 * aucs.std(ddof=1) / np.sqrt(len(aucs))
    print(f"\n[NeuroFormer] {args.folds}x{args.repeats} CV ROC-AUC = "
          f"{aucs.mean():.3f} +/- {ci:.3f}  (device={device}, EEG-alone bar=0.66)")
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"auc": aucs}).to_csv(args.out, index=False)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
