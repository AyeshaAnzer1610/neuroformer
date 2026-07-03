#!/usr/bin/env python
"""Fast Advanced Multimodal Training - Optimized for Speed."""

import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from pathlib import Path
import time

sys.path.insert(0, '/home/anzer/mental-health/src')
from advanced_multimodal import create_model
import pandas as pd

# ============ DATA LOADING ============
print("\nLoading data...")
labels = pd.read_csv('/home/anzer/mental-health/data/labels.csv', index_col='ID')
mask = labels['DX_Status'] < 2
subs = labels[mask].index.tolist()
y = labels.loc[subs, 'DX_Status'].values

print(f"  Binary: {len(subs)} subjects")
print(f"  Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}\n")

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

# Aggregate features
Xe = np.array([eeg[s].reshape(-1, 516).mean(0) if s in eeg else np.zeros(516) for s in subs])
Xy = np.array([eye.loc[s].values if s in eye.index else np.zeros(11) for s in subs])
Xb = np.zeros((len(subs), 6))

# Normalize
Xe = (Xe - Xe.mean(axis=0)) / (Xe.std(axis=0) + 1e-6)
Xy = (Xy - Xy.mean(axis=0)) / (Xy.std(axis=0) + 1e-6)

# ============ SETUP ============
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()

batch_size = 8  # Process multiple samples at once
num_epochs = 15  # Reduced from 30 for speed
learning_rate = 1e-2  # Slightly higher LR for faster convergence

# ============ CROSS-VALIDATION ============
print("\n" + "="*80)
print("Advanced Multimodal Model - Fast Training")
print(f"Batch size: {batch_size}, Epochs: {num_epochs}, LR: {learning_rate}")
print("="*80 + "\n")

skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
aucs = []
total_start = time.time()

for fold_idx, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
    fold_start = time.time()

    # Create model
    model = create_model(eeg_dim=516, eye_dim=11, beh_dim=6, hidden_dim=64).to(device)

    # Loss & optimizer
    pos = (y[tr] == 1).sum()
    neg = len(y[tr]) - pos
    weights = torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training
    model.train()
    for epoch in range(num_epochs):
        perm = np.random.permutation(len(y[tr]))

        for i in range(0, len(perm), batch_size):
            batch_idx = perm[i:i+batch_size]
            actual_idx = tr[batch_idx]

            # Batch
            eeg_b = torch.from_numpy(Xe[actual_idx]).float().to(device)
            eye_b = torch.from_numpy(Xy[actual_idx]).float().to(device)
            beh_b = torch.from_numpy(Xb[actual_idx]).float().to(device)
            y_b = torch.tensor(y[actual_idx], dtype=torch.long, device=device)

            # Forward
            optimizer.zero_grad()
            logits, _ = model(eeg_b, eye_b, beh_b)
            loss = loss_fn(logits, y_b)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

    # Evaluation
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(te), batch_size):
            batch_idx = te[i:i+batch_size]
            eeg_b = torch.from_numpy(Xe[batch_idx]).float().to(device)
            eye_b = torch.from_numpy(Xy[batch_idx]).float().to(device)
            beh_b = torch.from_numpy(Xb[batch_idx]).float().to(device)

            logits, _ = model(eeg_b, eye_b, beh_b)
            preds = torch.softmax(logits, -1)[:, 1]
            probs.extend(preds.cpu().numpy())

    auc = roc_auc_score(y[te], probs)
    aucs.append(auc)

    fold_time = time.time() - fold_start
    print(f"Fold {fold_idx+1:2d}/25: AUC={auc:.4f}, Time={fold_time:.1f}s")

    # Clean up
    del model
    torch.cuda.empty_cache()

aucs = np.array(aucs)
total_time = time.time() - total_start

# ============ RESULTS ============
print("\n" + "="*80)
print(f"Final Result: {aucs.mean():.4f} ± {aucs.std():.4f}")
print(f"Total time: {total_time:.1f}s ({total_time/25:.1f}s per fold)")
print("\nComparison:")
print(f"  Advanced Model:      {aucs.mean():.4f} ± {aucs.std():.4f}")
print(f"  Low-rank baseline:   0.7070 ± 0.0240")
print(f"  Demographics:        0.7170 ± 0.0220")
print("="*80)

if aucs.mean() > 0.717:
    print("✓ BEATS DEMOGRAPHICS!")
elif aucs.mean() > 0.708:
    print("✓ BEATS LOW-RANK BASELINE!")
else:
    print(f"  {0.708 - aucs.mean():.4f} AUC points needed to beat baselines")
