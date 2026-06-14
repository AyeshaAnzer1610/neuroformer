#!/usr/bin/env python
"""
Fine-tune LaBraM end-to-end for transdiagnostic screening (GPU).

The frozen-embedding pipeline caps at ~0.66; fine-tuning the foundation model is
the main lever for a higher number, and it pays off most with more data (scaled
HBN). Window-level training with subject-level CV (no leakage):
  - per subject -> QC-selected EEG windows (raw), label = subject diagnosis
  - train InterpolatedLaBraM on windows (optionally freeze backbone, train head)
  - evaluate per subject by mean-pooling its window probabilities -> ROC-AUC

Run inside a GPU SLURM job with the CUDA env (scripts/setup_env_gpu.sh):
    python src/finetune_labram.py \
        --input $SCRATCH/mh/data/mipdb --labels data/labels.csv \
        --all-recordings --epochs 8 --freeze-backbone
"""
import argparse
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from extract_eeg_embeddings import load_and_window


def load_all(input_dir, labels_csv, match, all_recordings, window_sec=15.0, max_subj=None):
    """Per-subject QC windows + labels. Returns list of (sid, windows, y), chs_info."""
    lab = pd.read_csv(labels_csv).set_index("ID")["DX_Status"].to_dict()
    from pathlib import Path
    subs = sorted(p for p in Path(input_dir).iterdir() if p.is_dir())
    if max_subj:
        subs = subs[:max_subj]
    data, chs = [], None
    for p in subs:
        sid = p.name
        if sid not in lab:
            continue
        try:
            raw, w = load_and_window(p, match=match, window_sec=window_sec,
                                     all_recordings=all_recordings)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {sid}: {e}"); continue
        if chs is None:
            chs = raw.info["chs"]
        data.append((sid, w.astype(np.float32), int(lab[sid] > 0)))
        print(f"[load] {sid}: {len(w)} windows, y={int(lab[sid] > 0)}", flush=True)
    print(f"[load] {len(data)} subjects with EEG+label")
    return data, chs


def build_model(chs, n_times, freeze_backbone, device):
    from braindecode.models import InterpolatedLaBraM
    m = InterpolatedLaBraM.from_pretrained(
        "braindecode/labram-pretrained", chs_info=chs, n_times=n_times,
        n_outputs=2, patch_size=200, strict=False)
    if freeze_backbone:
        n_train = 0
        for name, par in m.named_parameters():
            keep = any(k in name.lower() for k in ("head", "classif", "final", "fc", "norm"))
            par.requires_grad = keep
            n_train += par.requires_grad
        if n_train == 0:                       # fallback: unfreeze everything
            for par in m.parameters():
                par.requires_grad = True
    return m.to(device)


def run_fold(train, test, chs, args, device):
    n_times = train[0][1].shape[-1]
    model = build_model(chs, n_times, args.freeze_backbone, device)
    # window-level training tensors
    Xtr = np.concatenate([w for _, w, _ in train], 0)
    ytr = np.concatenate([[y] * len(w) for _, w, y in train])
    pos = int(ytr.sum()); neg = len(ytr) - pos
    w_ = torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=w_)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=args.lr, weight_decay=1e-4)
    idx = np.arange(len(Xtr))
    model.train()
    for ep in range(args.epochs):
        np.random.shuffle(idx)
        for i in range(0, len(idx), args.batch):
            b = idx[i:i + args.batch]
            xb = torch.from_numpy(Xtr[b]).to(device)
            yb = torch.from_numpy(ytr[b]).long().to(device)
            opt.zero_grad()
            out = model(xb)
            logits = out if torch.is_tensor(out) else out["logits"]
            loss = lossf(logits, yb)
            loss.backward(); opt.step()
    # subject-level eval: mean window prob
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for _, w, y in test:
            x = torch.from_numpy(w).to(device)
            out = model(x)
            logits = out if torch.is_tensor(out) else out["logits"]
            p = torch.softmax(logits, -1)[:, 1].mean().item()
            ys.append(y); ps.append(p)
    return roc_auc_score(ys, ps) if len(set(ys)) > 1 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--match", default=None); ap.add_argument("--all-recordings", action="store_true")
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-subj", type=int, default=None)
    ap.add_argument("--out", default="results/finetune_labram.csv")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] no GPU — fine-tuning on CPU will be very slow.")

    data, chs = load_all(args.input, args.labels, args.match, args.all_recordings,
                         max_subj=args.max_subj)
    y = np.array([d[2] for d in data])
    cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed)
    aucs = []
    for k, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y), 1):
        print(f"[fold {k}/{args.folds*args.repeats}] fine-tuning "
              f"(freeze_backbone={args.freeze_backbone})...", flush=True)
        auc = run_fold([data[i] for i in tr], [data[i] for i in te], chs, args, device)
        aucs.append(auc); print(f"[fold {k}] AUC={auc:.3f}", flush=True)
    aucs = np.array([a for a in aucs if not np.isnan(a)])
    ci = 1.96 * aucs.std(ddof=1) / np.sqrt(len(aucs)) if len(aucs) > 1 else float("nan")
    print(f"\n[finetune] {args.folds}x{args.repeats} CV ROC-AUC = {aucs.mean():.3f} "
          f"+/- {ci:.3f}  (frozen bar = 0.66)")
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"auc": aucs}).to_csv(args.out, index=False)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
