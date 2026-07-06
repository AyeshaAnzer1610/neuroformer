# -*- coding: utf-8 -*-
"""
STEP 25: MISSING-MODALITY / GRACEFUL-DEGRADATION EXPERIMENT  (MIPDB, N=92)
=========================================================================
Turns "the architecture is designed to support missing modalities" into an
actual measurement, so the paper can *report* degradation rather than *claim*
robustness.

Mechanism (implemented here, not just intended):
  - a confidence-gated NeuroFormer over EEG (hierarchical encoder) + eye,
  - an availability MASK per sample: absent modalities are excluded from the
    confidence softmax (logit -> -inf) and their features zeroed -- NOT fed as
    zeros through the gate,
  - MODALITY-DROPOUT TRAINING: each training sample randomly drops one modality
    with prob --p-drop, so the model learns to classify from a subset.

Evaluation (subject-level, leakage-free, repeated CV): the SAME trained model is
tested under three availability conditions and AUC is reported for each:
  EEG+eye  (full)   |   EEG-only (eye missing)   |   eye-only (EEG missing)

Behaviour is NOT included: it does not exist for this cohort (X_beh is empty).
The honest read is the shape of the degradation, whatever it is (it may well
fall toward chance -- that is a finding, reported, not hidden).

Run:
  python src/25_missing_modality.py --repeats 10 --epochs 60 --out results/missing_modality.csv
Smoke test:
  python src/25_missing_modality.py --repeats 1 --epochs 5
"""
import argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from model_suite import load_mipdb
from advanced_multimodal import HierarchicalEEGEncoder

MODS = ['eeg', 'eye']


class MissingModalityNeuroFormer(nn.Module):
    """Confidence-gated EEG+eye NeuroFormer with per-sample availability masking."""
    def __init__(self, eeg_dim, eye_dim, h=64):
        super().__init__()
        self.eeg_enc = HierarchicalEEGEncoder(eeg_dim, hidden_dim=h)
        self.eye_proj = nn.Sequential(nn.Linear(eye_dim, h), nn.ReLU(), nn.Dropout(0.2))
        self.conf_eeg = nn.Linear(h, 1)
        self.conf_eye = nn.Linear(h, 1)
        self.fusion = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.2))
        self.classifier = nn.Linear(h, 2)

    def forward(self, eeg, eye, mask):        # mask: (B,2), 1=present
        fe = self.eeg_enc(eeg)
        fy = self.eye_proj(eye)
        logits = torch.cat([self.conf_eeg(fe), self.conf_eye(fy)], dim=1)   # (B,2)
        logits = logits.masked_fill(mask == 0, -1e9)                        # absent -> no weight
        gates = torch.softmax(logits, dim=1)                               # (B,2)
        feats = torch.stack([fe, fy], dim=1) * mask.unsqueeze(-1)          # zero absent feats
        fused = (gates.unsqueeze(-1) * feats).sum(dim=1)                    # (B,h)
        return self.classifier(self.fusion(fused))


def train_mask(n, p_drop):
    """Per-sample training mask: with prob p_drop drop exactly one modality."""
    m = torch.ones(n, 2)
    drop = torch.rand(n) < p_drop
    which = torch.randint(0, 2, (n,))
    m[torch.arange(n)[drop], which[drop]] = 0.0
    return m


def fit_eval(Xe_tr, Xy_tr, y_tr, Xe_te, Xy_te, y_te, epochs, p_drop, seed):
    torch.manual_seed(seed)
    net = MissingModalityNeuroFormer(Xe_tr.shape[1], Xy_tr.shape[1])
    pw = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
    lf = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pw], dtype=torch.float32))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-3)
    et, yt = torch.tensor(Xe_tr, dtype=torch.float32), torch.tensor(Xy_tr, dtype=torch.float32)
    lab = torch.tensor(y_tr, dtype=torch.long)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = net(et, yt, train_mask(len(y_tr), p_drop))
        lf(out, lab).backward()
        opt.step()
    net.eval()
    ete, yte_ = torch.tensor(Xe_te, dtype=torch.float32), torch.tensor(Xy_te, dtype=torch.float32)
    res = {}
    with torch.no_grad():
        for cond, mv in [('EEG+eye', [1, 1]), ('EEG-only', [1, 0]), ('eye-only', [0, 1])]:
            mask = torch.tensor(mv, dtype=torch.float32).repeat(len(y_te), 1)
            p = torch.softmax(net(ete, yte_, mask), dim=1)[:, 1].numpy()
            res[cond] = roc_auc_score(y_te, p)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--p-drop', type=float, default=0.5, help='modality-dropout prob (train)')
    ap.add_argument('--k-eeg', type=int, default=128)
    ap.add_argument('--out', default='results/missing_modality.csv')
    a = ap.parse_args()

    b = load_mipdb(k_eeg=a.k_eeg)
    print(f"MIPDB N={b.N} ({int(b.y.sum())} pos) | modality-dropout p={a.p_drop} | "
          f"repeats={a.repeats} epochs={a.epochs}")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=a.repeats, random_state=0)
    folds = {'EEG+eye': [], 'EEG-only': [], 'eye-only': []}
    for i, (tr, te) in enumerate(cv.split(np.zeros(b.N), b.y)):
        se = StandardScaler().fit(b.eeg[tr]); sy = StandardScaler().fit(b.eye[tr])
        r = fit_eval(se.transform(b.eeg[tr]), sy.transform(b.eye[tr]), b.y[tr],
                     se.transform(b.eeg[te]), sy.transform(b.eye[te]), b.y[te],
                     a.epochs, a.p_drop, seed=i)
        for k in folds:
            folds[k].append(r[k])

    rows = []
    for cond, vals in folds.items():
        v = np.array(vals)
        rows.append(dict(condition=cond, auc=round(float(v.mean()), 4),
                         sd=round(float(v.std()), 4),
                         ci_lo=round(float(np.percentile(v, 2.5)), 4),
                         ci_hi=round(float(np.percentile(v, 97.5)), 4), n_fold=len(v)))
        print(f"  {cond:<10} AUC={v.mean():.4f} +/- {v.std():.4f}  "
              f"[{np.percentile(v,2.5):.3f}, {np.percentile(v,97.5):.3f}]")
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\n[saved] {a.out}")
    print("Interpretation: compare EEG+eye vs EEG-only vs eye-only. Report the shape "
          "of degradation honestly (relies-on-EEG / falls-to-chance / etc.).")


if __name__ == '__main__':
    main()
