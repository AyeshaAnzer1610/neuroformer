#!/usr/bin/env python
"""
REAL-DATA test of the CIDA readout head (no synthetic data).

Uses the cached frozen-LaBraM per-window embeddings (data/processed/*_labram.npz,
[n_windows, 200] per subject) + real MIPDB labels (data/labels.csv). Subject-level
repeated stratified CV. Compares, on identical splits:

  (1) baseline   : mean+std pooled embedding -> logreg   (the frozen-readout ~0.66 bar)
  (2) CIDA head  : per-window tokens -> attention-pool + classifier (disentangle OFF,
                   since the mechanism test falsified it). Tests whether the learned
                   pooling/head beats the mean-pool readout on REAL data.

This complements the cluster LoRA job (which fine-tunes the backbone); this one only
adapts the head on frozen features, so it runs locally in a couple of minutes.
"""
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cida import CIDA  # noqa: E402

# Local env has a torch<->NumPy ABI break; bridge via Python lists.
def t_f(a):
    return torch.tensor(np.asarray(a).tolist(), dtype=torch.float32)


SID = re.compile(r"(A\d+)")
MAXW = 40          # cap windows/subject (memory + matches finetune_labram --max-windows)
FOLDS, REPEATS, SEED = 5, 3, 0
torch.manual_seed(SEED); np.random.seed(SEED)

# ---- load real data -------------------------------------------------------
lab = pd.read_csv("data/labels.csv").set_index("ID")["DX_Status"].to_dict()
emb = {}
for f in sorted(glob.glob("data/processed/*_labram.npz")):
    sid = SID.match(Path(f).name).group(1)
    if sid in lab:
        emb[sid] = np.load(f)["embeddings"].astype(np.float32)
subs = sorted(emb)
y = np.array([int(lab[s] > 0) for s in subs])
print(f"[data] {len(subs)} subjects with EEG+label | positives={int(y.sum())} "
      f"| windows/subj median={int(np.median([len(emb[s]) for s in subs]))}")

# ---- (1) baseline: mean+std readout + logreg ------------------------------
Xms = np.array([np.concatenate([emb[s].mean(0), emb[s].std(0)]) for s in subs])
cv = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=SEED)
pipe = make_pipeline(StandardScaler(),
                     LogisticRegression(max_iter=5000, class_weight="balanced"))
base = cross_val_score(pipe, Xms, y, cv=cv, scoring="roc_auc")
print(f"[baseline] mean+std readout + logreg : ROC-AUC {base.mean():.3f} "
      f"+/- {1.96*base.std(ddof=1)/np.sqrt(len(base)):.3f}  (frozen bar ~0.66)")


# ---- (2) CIDA attention-pool head -----------------------------------------
def pad_batch(idx):
    """subjects -> padded tokens [B, MAXW, 1, 200] + window mask [B, MAXW]."""
    B = len(idx)
    X = np.zeros((B, MAXW, 1, 200), np.float32); M = np.zeros((B, MAXW), np.float32)
    for i, s in enumerate(idx):
        w = emb[s][:MAXW]
        X[i, :len(w), 0] = w; M[i, :len(w)] = 1.0
    return t_f(X), t_f(M)


def cida_fold(tr_s, te_s, ytr):
    torch.manual_seed(SEED)
    model = CIDA(n_channels=1, d=200, n_classes=2, d_factor=64)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3,
                           weight_decay=1e-3)
    Xtr, Mtr = pad_batch(tr_s); yt = torch.tensor(ytr.tolist(), dtype=torch.long)
    pos = max(int(ytr.sum()), 1)
    cw = torch.tensor([len(ytr) / (2 * (len(ytr) - pos)), len(ytr) / (2 * pos)],
                      dtype=torch.float32)
    model.train()
    for _ in range(120):
        opt.zero_grad()
        out = model(Xtr, window_mask=Mtr)        # disentangle OFF -> plain CE on logits
        loss = nn.functional.cross_entropy(out["logits"], yt, weight=cw)
        loss.backward(); opt.step()
    model.eval()
    Xte, Mte = pad_batch(te_s)
    with torch.no_grad():
        p = torch.softmax(model(Xte, window_mask=Mte)["logits"], -1)[:, 1].tolist()
    return p


aucs = []
for tr, te in cv.split(np.zeros(len(y)), y):
    tr_s = [subs[i] for i in tr]; te_s = [subs[i] for i in te]
    p = cida_fold(tr_s, te_s, y[tr])
    if len(np.unique(y[te])) > 1:
        aucs.append(roc_auc_score(y[te], p))
aucs = np.array(aucs)
print(f"[CIDA head] attention-pool + classifier: ROC-AUC {aucs.mean():.3f} "
      f"+/- {1.96*aucs.std(ddof=1)/np.sqrt(len(aucs)):.3f}  (disentangle OFF)")
print("\n(both on REAL cached LaBraM embeddings, identical subject-level "
      f"{FOLDS}x{REPEATS} CV; no synthetic data)")
