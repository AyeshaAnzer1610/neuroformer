#!/usr/bin/env python
"""
Does identity-disentanglement improve MIPDB -> HBN transfer? (real cross-cohort)

The honest negative we already have: a plain MIPDB-trained EEG classifier applied to
HBN collapses to ~0.50 (src/eval_hbn.py). Hypothesis: removing SUBJECT-identity from the
clinical representation makes it less cohort-specific, so it should transfer better.

Protocol (cluster-only — needs HBN embeddings on $SCRATCH):
  - Train a window-level model on ALL MIPDB (frozen-LaBraM embeddings + a small head),
    with the disentangler ON vs OFF (identical architecture). Standardize with MIPDB stats.
  - Apply to HBN: per subject, mean window prob -> score. Evaluate median-split AUC on the
    HBN p_factor (high vs low symptom), exactly like eval_hbn's transfer arm.
  - Compare transfer AUC OFF vs ON (a few adversary strengths), 3 seeds. Also the plain
    logreg transfer as the reference (~0.50).

    python experiments/transfer_hbn.py \
        --mipdb-dir $SCRATCH/mh/data/processed \
        --hbn-dir   $SCRATCH/mh/data/processed_hbn \
        --hbn-part  $SCRATCH/mh/data/hbn_r1/participants.tsv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from compare_fm_vs_handcrafted import load_windows          # noqa: E402
from cida import grad_reverse                                # noqa: E402

MAXW = 40


class Disentangler(nn.Module):
    """Window-level: enc -> z_clin/z_nuis; clinical head on z_clin; subject adversary
    (GRL) on z_clin; nuisance head on z_nuis; orthogonality. (Same design that removed
    real identity at low cost in experiments/disentangle_real.py.)"""

    def __init__(self, d_in=200, d_h=128, d_f=64, n_subj=1):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d_h), nn.GELU(), nn.LayerNorm(d_h))
        mk = lambda o: nn.Sequential(nn.Linear(d_h, d_f), nn.GELU(), nn.Linear(d_f, o))
        self.to_clin = mk(d_f); self.to_nuis = mk(d_f)
        self.clin = nn.Linear(d_f, 2)
        self.adv = nn.Sequential(nn.Linear(d_f, d_f), nn.GELU(), nn.Linear(d_f, n_subj))
        self.nui = nn.Sequential(nn.Linear(d_f, d_f), nn.GELU(), nn.Linear(d_f, n_subj))

    def forward(self, x, grl=0.0):
        h = self.enc(x); zc = self.to_clin(h); zn = self.to_nuis(h)
        return {"logits": self.clin(zc), "zc": zc, "zn": zn,
                "adv": self.adv(grad_reverse(zc, grl)), "nui": self.nui(zn)}


def load_mipdb(d, labels):
    lab = pd.read_csv(labels).set_index("ID")["DX_Status"].to_dict()
    w = load_windows(d, "_labram", "embeddings")
    subs = sorted(s for s in w if s in lab)
    X, y, sidx = [], [], []
    for j, s in enumerate(subs):
        for ww in w[s][:MAXW]:
            X.append(ww); y.append(int(lab[s] > 0)); sidx.append(j)
    return np.asarray(X, np.float32), np.array(y), np.array(sidx), len(subs)


def load_hbn(d, part_tsv, target="p_factor"):
    part = pd.read_csv(part_tsv, sep="\t")
    vals = {r["participant_id"]: r[target] for _, r in part.iterrows()}
    w = load_windows(d, "_labram", "embeddings")
    subs = [s for s in w if s in vals and np.isfinite(vals[s])]
    y = np.array([vals[s] for s in subs], float)
    yb = (y > np.median(y)).astype(int)
    return {s: w[s][:MAXW].astype(np.float32) for s in subs}, subs, yb


def train_eval(Xtr, ytr, sidx, n_subj, hbn_w, hbn_subs, yb, lam_adv, seed, epochs=200):
    torch.manual_seed(seed)
    model = Disentangler(n_subj=n_subj)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr).long(); st = torch.from_numpy(sidx).long()
    pos = max(int(ytr.sum()), 1); cw = torch.tensor(
        [len(ytr) / (2 * (len(ytr) - pos)), len(ytr) / (2 * pos)], dtype=torch.float32)
    model.train()
    for ep in range(epochs):
        grl = min(1.0, ep / 40) if lam_adv > 0 else 0.0
        opt.zero_grad()
        out = model(Xt, grl=grl)
        loss = F.cross_entropy(out["logits"], yt, weight=cw)
        if lam_adv > 0:
            ortho = ((out["zc"] - out["zc"].mean(0)).t() @ (out["zn"] - out["zn"].mean(0))
                     / (len(ytr) - 1)).pow(2).sum()
            loss = loss + lam_adv * F.cross_entropy(out["adv"], st) \
                        + 0.5 * lam_adv * F.cross_entropy(out["nui"], st) + 0.1 * ortho
        loss.backward(); opt.step()
    # transfer: per HBN subject, mean window prob
    model.eval(); scores = []
    with torch.no_grad():
        for s in hbn_subs:
            o = model(torch.from_numpy(hbn_w[s]))
            scores.append(float(torch.softmax(o["logits"], -1)[:, 1].mean()))
    return roc_auc_score(yb, scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mipdb-dir", required=True)
    ap.add_argument("--hbn-dir", required=True)
    ap.add_argument("--hbn-part", required=True)
    ap.add_argument("--mipdb-labels", default="data/labels.csv")
    ap.add_argument("--target", default="p_factor")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/transfer_hbn_disentangle.csv")
    args = ap.parse_args()

    Xtr, ytr, sidx, n_subj = load_mipdb(args.mipdb_dir, args.mipdb_labels)
    hbn_w, hbn_subs, yb = load_hbn(args.hbn_dir, args.hbn_part, args.target)
    print(f"[data] MIPDB train: {n_subj} subjects, {len(Xtr)} windows | "
          f"HBN test: {len(hbn_subs)} subjects, {int(yb.sum())}/{len(yb)} high-{args.target}")

    # standardize with MIPDB stats (applied to both)
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr).astype(np.float32)
    hbn_w = {s: sc.transform(w).astype(np.float32) for s, w in hbn_w.items()}

    # reference: plain logreg transfer (meanstd pooled), the eval_hbn ~0.50 baseline
    Xtr_sub = np.array([Xtr[sidx == j].mean(0) for j in range(n_subj)])
    ytr_sub = np.array([ytr[sidx == j][0] for j in range(n_subj)])
    lr = LogisticRegression(max_iter=5000, class_weight="balanced").fit(Xtr_sub, ytr_sub)
    lr_scores = [lr.predict_proba(hbn_w[s].mean(0, keepdims=True))[0, 1] for s in hbn_subs]
    print(f"\n[ref] plain logreg transfer (mean-pooled): AUC={roc_auc_score(yb, lr_scores):.3f}")

    rows = [{"setting": "logreg-transfer", "auc": float(roc_auc_score(yb, lr_scores))}]
    print("\nsetting                 HBN transfer AUC (median-split, mean over seeds)")
    for lam in [0.0, 0.25, 0.5, 1.0]:
        aucs = [train_eval(Xtr, ytr, sidx, n_subj, hbn_w, hbn_subs, yb, lam, s)
                for s in range(args.seeds)]
        tag = "disentangle OFF" if lam == 0 else f"disentangle lam={lam}"
        print(f"  {tag:<22}{np.mean(aucs):.3f}  (per-seed {[round(a,3) for a in aucs]})")
        rows.append({"setting": tag, "auc": float(np.mean(aucs))})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\n[done] wrote {args.out}  (does disentanglement lift transfer above OFF / ~0.50?)")


if __name__ == "__main__":
    main()
