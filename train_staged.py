#!/usr/bin/env python
"""
Train Stages 1-3 Model with Confidence-Gated Multimodal Fusion

Uses proper architecture with:
- Stage 1: Euclidean alignment + EEGNet tokenizer
- Stage 2: Criss-cross attention
- Stage 3: Hierarchical processing
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from pathlib import Path
import time

sys.path.insert(0, '/home/anzer/mental-health/src')

from staged_model import create_model
import pandas as pd

# ============ LOAD DATA ============
print("\nLoading data...")
labels = pd.read_csv('/home/anzer/mental-health/data/labels.csv', index_col='ID')
mask = labels['DX_Status'] < 2
subs = labels[mask].index.tolist()
y = labels.loc[subs, 'DX_Status'].values

print(f"  Binary: {len(subs)} subjects")
print(f"  Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")

# Load EEG
eeg = {}
for s in subs:
    p = f'/home/anzer/mental-health/data/processed/{s}_eeg.npz'
    if Path(p).exists():
        eeg[s] = np.load(p)['eeg']
    else:
        eeg[s] = np.zeros((1, 516))

# Load eye
eye = pd.read_csv('/home/anzer/mental-health/data/eye_features.csv', index_col=0)

# Aggregate features (mean over window)
Xe = np.array([eeg[s].reshape(-1, 516).mean(0) if s in eeg else np.zeros(516) for s in subs])
Xy = np.array([eye.loc[s].values if s in eye.index else np.zeros(11) for s in subs])
Xb = np.zeros((len(subs), 6))

# Normalize
Xe = (Xe - Xe.mean(0)) / (Xe.std(0) + 1e-6)
Xy = (Xy - Xy.mean(0)) / (Xy.std(0) + 1e-6)

print(f"  EEG: {Xe.shape}, Eye: {Xy.shape}, Beh: {Xb.shape}\n")

# ============ SETUP ============
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\n" + "="*80)
print("Stages 1-3 Model Training")
print("="*80 + "\n")

# ============ CROSS-VALIDATION ============
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
aucs = []
total_start = time.time()

for fold_idx, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
    fold_start = time.time()

    # Create model
    model = create_model(eeg_dim=516, eye_dim=11, beh_dim=6, hidden_dim=64).to(device)

    # Loss with class weights
    pos = (y[tr] == 1).sum()
    neg = len(y[tr]) - pos
    weights = torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # Optimizer with warmup-like behavior
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # Training loop
    model.train()
    for epoch in range(20):
        perm = np.random.permutation(len(y[tr]))

        # Mini-batch training
        batch_size = 4
        for i in range(0, len(perm), batch_size):
            batch_idx = perm[i:i+batch_size]
            actual_idx = tr[batch_idx]

            # Prepare batch
            eeg_b = torch.from_numpy(Xe[actual_idx]).float().to(device)
            eye_b = torch.from_numpy(Xy[actual_idx]).float().to(device)
            beh_b = torch.from_numpy(Xb[actual_idx]).float().to(device)
            y_b = torch.tensor(y[actual_idx], dtype=torch.long, device=device)

            # Forward pass
            optimizer.zero_grad()
            logits = model(eeg_b, eye_b, beh_b)
            loss = loss_fn(logits, y_b)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

    # Evaluation
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(te), 4):
            batch_idx = te[i:i+4]
            eeg_b = torch.from_numpy(Xe[batch_idx]).float().to(device)
            eye_b = torch.from_numpy(Xy[batch_idx]).float().to(device)
            beh_b = torch.from_numpy(Xb[batch_idx]).float().to(device)

            logits = model(eeg_b, eye_b, beh_b)
            preds = torch.softmax(logits, dim=-1)[:, 1]
            probs.extend(preds.cpu().numpy())

    # Compute AUC
    auc = roc_auc_score(y[te], probs[:len(te)])
    aucs.append(auc)

    fold_time = time.time() - fold_start
    print(f"Fold {fold_idx+1:2d}/25: AUC={auc:.4f}, Time={fold_time:.1f}s")

    # Cleanup
    del model
    torch.cuda.empty_cache()

aucs = np.array(aucs)
total_time = time.time() - total_start

# ============ RESULTS ============
print("\n" + "="*80)
print(f"FINAL RESULT: {aucs.mean():.4f} ± {aucs.std():.4f}")
print(f"Total time: {total_time:.1f}s ({total_time/25:.1f}s per fold)")
print("\nComparison with baselines:")
print(f"  Stages 1-3 Model:    {aucs.mean():.4f} ± {aucs.std():.4f}  ← NEW")
print(f"  Low-rank baseline:   0.7070 ± 0.0240")
print(f"  Demographics:        0.7170 ± 0.0220")
print("="*80)

if aucs.mean() > 0.717:
    print("\n✓✓✓ BEATS DEMOGRAPHICS BASELINE! ✓✓✓\n")
elif aucs.mean() > 0.708:
    print("\n✓✓ BEATS LOW-RANK BASELINE! ✓✓\n")
else:
    print(f"\nNeeds {0.708 - aucs.mean():.4f} more AUC points to beat baselines\n")
