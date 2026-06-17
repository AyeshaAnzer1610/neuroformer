#!/usr/bin/env python
"""
REAL-DATA disentanglement gate + improved disentangler (no synthetic data).

The frozen LaBraM embedding is the thing that leaked subject identity (cause of the
original 0.91). Here we measure that on REAL cached embeddings and test whether an
IMPROVED disentangler (the fixes from the CIDA design doc: nonlinear factor heads +
a stronger/stabilized multi-layer adversary) removes identity while holding the
clinical signal.

Data: data/processed/*_labram.npz ([n_windows,200]/subject) + data/labels.csv.

Metrics (subject-level split; windows of the SAME subject share identity):
  - clinical AUC : subject-level (mean window prob) on held-out subjects.
  - subject-decodability of z_clin : a probe predicts SUBJECT-ID from z_clin using
    held-out WINDOWS of TRAIN subjects. High = identity still present (bad). Chance
    = 1/n_train_subjects. LOWER (toward chance) = identity removed = better.

Gate: improved disentangler should LOWER subject-decodability vs OFF while keeping
clinical AUC >= the OFF/baseline. Reported honestly either way.
"""
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cida import grad_reverse  # noqa: E402  (reuse the GRL)


def t_f(a):
    return torch.tensor(np.asarray(a).tolist(), dtype=torch.float32)


def t_l(a):
    return torch.tensor(np.asarray(a).tolist(), dtype=torch.long)


SID = re.compile(r"(A\d+)")
MAXW, SEEDS = 40, (0, 1, 2)
torch.manual_seed(0); np.random.seed(0)

# ---- load real data -> window-level arrays --------------------------------
lab = pd.read_csv("data/labels.csv").set_index("ID")["DX_Status"].to_dict()
emb = {}
for f in sorted(glob.glob("data/processed/*_labram.npz")):
    s = SID.match(Path(f).name).group(1)
    if s in lab:
        emb[s] = np.load(f)["embeddings"].astype(np.float32)[:MAXW]
subs = np.array(sorted(emb))
ysub = np.array([int(lab[s] > 0) for s in subs])
print(f"[data] {len(subs)} subjects | positives={int(ysub.sum())} "
      f"| frozen-embedding subject-identity leakage measured below")


class Disentangler(nn.Module):
    """Improved over the falsified linear/GRL version: NONLINEAR factor heads and a
    STRONGER multi-layer adversary (the design-doc fix). z_clin feeds the clinical
    head; GRL(z_clin) feeds the subject adversary; z_nuis absorbs identity."""

    def __init__(self, d_in=200, d_h=128, d_f=64, n_subj=1, nonlinear=True):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d_h), nn.GELU(), nn.LayerNorm(d_h))
        mk = (lambda o: nn.Sequential(nn.Linear(d_h, d_f), nn.GELU(), nn.Linear(d_f, o))) \
            if nonlinear else (lambda o: nn.Linear(d_h, o))
        self.to_clin = mk(d_f); self.to_nuis = mk(d_f)
        self.clin = nn.Linear(d_f, 2)
        self.adv = nn.Sequential(nn.Linear(d_f, d_f), nn.GELU(), nn.Linear(d_f, n_subj))
        self.nui = nn.Sequential(nn.Linear(d_f, d_f), nn.GELU(), nn.Linear(d_f, n_subj))

    def forward(self, x, grl=0.0):
        h = self.enc(x); zc = self.to_clin(h); zn = self.to_nuis(h)
        return {"logits": self.clin(zc), "zc": zc, "zn": zn,
                "adv": self.adv(grad_reverse(zc, grl)), "nui": self.nui(zn)}


def run(lam_adv, seed):
    """lam_adv scales the adversary pressure; 0 = no disentanglement (but same
    nonlinear architecture, for a fair comparison)."""
    disentangle = lam_adv > 0
    torch.manual_seed(seed)
    tr_s, te_s = train_test_split(np.arange(len(subs)), test_size=0.3,
                                  stratify=ysub, random_state=seed)
    # window-level tensors for TRAIN subjects (+ a held-out window split for the probe)
    Xtr, ytr, sidx, is_probe = [], [], [], []
    for j, si in enumerate(tr_s):
        w = emb[subs[si]]; npr = max(1, len(w) // 3)
        for k, ww in enumerate(w):
            Xtr.append(ww); ytr.append(ysub[si]); sidx.append(j)
            is_probe.append(k >= len(w) - npr)         # last third = probe-test windows
    Xtr = t_f(Xtr); ytr_t = t_l(ytr); sidx_t = t_l(sidx); is_probe = np.array(is_probe)
    n_subj = len(tr_s)
    cwn = np.bincount(ytr, minlength=2); cw = t_f([len(ytr) / (2 * max(cwn[0], 1)),
                                                   len(ytr) / (2 * max(cwn[1], 1))])

    model = Disentangler(n_subj=n_subj, nonlinear=True)   # always nonlinear; vary lam_adv
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    EP = 200
    model.train()
    for ep in range(EP):
        grl = min(1.0, ep / 40) if disentangle else 0.0
        opt.zero_grad()
        out = model(Xtr, grl=grl)
        loss = F.cross_entropy(out["logits"], ytr_t, weight=cw)
        if disentangle:
            ortho = ((out["zc"] - out["zc"].mean(0)).t() @ (out["zn"] - out["zn"].mean(0))
                     / (len(ytr) - 1)).pow(2).sum()
            loss = loss + lam_adv * F.cross_entropy(out["adv"], sidx_t) \
                        + 0.5 * lam_adv * F.cross_entropy(out["nui"], sidx_t) + 0.1 * ortho
        loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        zc_all = model(Xtr)["zc"]
    # subject-decodability of z_clin: probe on probe-TRAIN windows -> eval probe-TEST
    zc = np.array(zc_all.tolist()); sidx = np.array(sidx)
    ptr, pte = ~is_probe, is_probe
    probe = LogisticRegression(max_iter=2000, multi_class="auto").fit(zc[ptr], sidx[ptr])
    subj_dec = probe.score(zc[pte], sidx[pte])

    # clinical AUC on held-out SUBJECTS (mean window prob)
    with torch.no_grad():
        ps, ys = [], []
        for si in te_s:
            o = model(t_f(emb[subs[si]]))
            ps.append(float(torch.softmax(o["logits"], -1)[:, 1].mean())); ys.append(ysub[si])
    auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")
    return auc, subj_dec, 1.0 / n_subj


print("\nadversary strength     in-dist clin AUC   subject-decodability(z_clin)   chance")
for lam in [0.0, 0.1, 0.25, 0.5, 1.0]:
    A = np.array([run(lam, s) for s in SEEDS])
    auc, dec, ch = A[:, 0].mean(), A[:, 1].mean(), A[:, 2].mean()
    tag = "OFF" if lam == 0 else f"lam_adv={lam}"
    print(f"  {tag:<22}{auc:>12.3f}{dec:>26.3f}{ch:>12.3f}")
print("\n(subject-decodability: closer to 'chance' = identity removed from z_clin = better.")
print(" GATE: want a row that LOWERS decodability vs OFF while keeping clinical AUC ~0.667.)")
