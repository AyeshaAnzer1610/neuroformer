#!/usr/bin/env python
"""Test Confidence-Gated and Cross-Modal Refined models on binary classification."""

import sys, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from pathlib import Path

sys.path.insert(0, '/home/anzer/mental-health/src')

from confidence_gated_fusion_simple import ConfidenceGatedFusionSimple, CrossModalRefinedSimple
import pandas as pd

# Load and filter to binary (classes 0 and 1, exclude class 2)
labels = pd.read_csv('/home/anzer/mental-health/data/labels.csv', index_col='ID')
mask = labels['DX_Status'] < 2
subs = labels[mask].index.tolist()
y = labels.loc[subs, 'DX_Status'].values

print(f"Binary classification: {len(subs)} subjects")
print(f"  Class 0: {(y==0).sum()}")
print(f"  Class 1: {(y==1).sum()}")

# Load features
eeg = {}
for s in subs:
    p = f'/home/anzer/mental-health/data/processed/{s}_eeg.npz'
    if Path(p).exists():
        eeg[s] = np.load(p)['eeg']
    else:
        eeg[s] = np.zeros((1, 516))

eye = pd.read_csv('/home/anzer/mental-health/data/eye_features.csv', index_col=0)

Xe = np.array([eeg[s].reshape(-1, 516).mean(0) if s in eeg else np.zeros(516) for s in subs])
Xy = np.array([eye.loc[s].values if s in eye.index else np.zeros(11) for s in subs])
Xb = np.zeros((len(subs), 6))

Xd = {"eeg": Xe, "eye": Xy, "beh": Xb}
dims = {"eeg": 516, "eye": 11, "beh": 6}

def cv_test(model_class, Xd, y, name):
    """Run 5x5 cross-validation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
    aucs = []

    for fold_idx, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        m = model_class(dims).to(device)

        # Class weights for imbalanced data
        pos = (y[tr] == 1).sum()
        neg = len(y[tr]) - pos
        wt = torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float32, device=device)
        lf = nn.CrossEntropyLoss(weight=wt)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)

        # Train for 20 epochs
        for epoch in range(20):
            # Shuffle and iterate through training samples
            for i in np.random.permutation(len(y[tr])):
                opt.zero_grad()
                idx = tr[i]
                xb = {k: torch.from_numpy(Xd[k][idx:idx+1]).float().to(device) for k in Xd}
                lo, _ = m(xb)
                loss = lf(lo, torch.tensor([y[idx]], dtype=torch.long, device=device))
                loss.backward()
                opt.step()

        # Evaluate on test fold
        m.eval()
        probs = []
        with torch.no_grad():
            for i in te:
                xb = {k: torch.from_numpy(Xd[k][i:i+1]).float().to(device) for k in Xd}
                lo, _ = m(xb)
                probs.append(torch.softmax(lo, -1)[0, 1].item())

        auc = roc_auc_score(y[te], probs) if len(np.unique(y[te])) > 1 else np.nan
        aucs.append(auc)
        print(f"  Fold {fold_idx+1}: {auc:.4f}")

    aucs = np.array(aucs)
    print(f"{name:45s}: {aucs.mean():.4f} ± {aucs.std():.4f}\n")
    return aucs

print("\n" + "="*80)
print("Testing new models on 5x5 cross-validation")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("="*80 + "\n")

a1 = cv_test(ConfidenceGatedFusionSimple, Xd, y, "Confidence-Gated Fusion (Simple)")
a2 = cv_test(CrossModalRefinedSimple, Xd, y, "Cross-Modal Refined (Simple)")

print("="*80)
print("Summary:")
print(f"  Confidence-Gated:     {a1.mean():.4f} ± {a1.std():.4f}")
print(f"  Cross-Modal Refined:  {a2.mean():.4f} ± {a2.std():.4f}")
print(f"  Low-rank baseline:    0.7070 ± 0.0240")
print(f"  Demographics (Age+Sex): 0.7170 ± 0.0220")
print("="*80)