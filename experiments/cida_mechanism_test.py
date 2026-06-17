#!/usr/bin/env python
"""
Controlled mechanism test for CIDA (synthetic, runs on CPU in seconds).

Purpose: BEFORE writing any claim, check empirically that CIDA does what we say —
on data where we CONTROL the confound. We synthesize "foundation embeddings" in which
subject/site IDENTITY is the DOMINANT signal (mimicking LaBraM's identity-dominance),
with a weaker genuine clinical signal tied to the label. Then we A/B the SAME CIDA
model with disentanglement ON vs OFF (adversary + nuisance + orthogonality losses),
identical capacity and training budget.

We report three things:
  1. In-distribution clinical AUC (test subjects from the SAME sites as train).
  2. Cross-site clinical AUC (test subjects from a HELD-OUT site) — the transfer axis
     where LaBraM collapses (our HBN 0.50).
  3. Site-decodability of z_clin (a probe predicting site from the clinical factor) —
     LOWER = better disentanglement (identity removed from the clinical representation).

Honest expectation: ON should lower site-decodability and hold up better cross-site.
If it does NOT, that is a real negative result and we report it rather than write the claim.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cida import CIDA  # noqa: E402

# This machine has a torch<->NumPy ABI mismatch (torch built for NumPy 1.x, 2.x
# installed), so torch.from_numpy / .numpy() raise. NumPy itself is fine; only the
# C-bridge is broken. Route around it via Python lists. (On the cluster env these
# helpers are still correct, just unnecessary.)
def t_f(a):
    return torch.tensor(np.asarray(a).tolist(), dtype=torch.float32)


def t_l(a):
    return torch.tensor(np.asarray(a).tolist(), dtype=torch.long)


def to_np(t):
    return np.array(t.detach().tolist())


SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# ----- synthetic generator -------------------------------------------------
C, D, W = 8, 32, 6                 # channels, token dim, windows/subject
K_CLIN, K_ID = 3, 16               # latent dims: clinical (small) vs identity (large)
CLIN_SCALE, ID_SCALE, SITE_SCALE, NOISE = 0.6, 2.0, 1.5, 0.5   # identity/site DOMINATE

clin_basis = rng.normal(size=(K_CLIN, C, D))
id_basis = rng.normal(size=(K_ID, C, D))
site_basis = rng.normal(size=(4, C, D))
clin_dir = rng.normal(size=K_CLIN)          # maps label sign -> clinical latent


def make_subject(y, site):
    """One subject's token tensor [W, C, D] + its identity vector."""
    u = rng.normal(size=K_ID)                                   # subject identity
    clin = (2 * y - 1) * clin_dir + 0.3 * rng.normal(size=K_CLIN)
    base = (CLIN_SCALE * np.einsum("k,kcd->cd", clin, clin_basis)
            + ID_SCALE * np.einsum("k,kcd->cd", u, id_basis)
            + SITE_SCALE * site_basis[site])
    win = base[None] + NOISE * rng.normal(size=(W, C, D))       # windows share identity
    return win.astype(np.float32)


def make_cohort(n, sites):
    X, Y, S = [], [], []
    for i in range(n):
        y = int(rng.integers(0, 2)); site = int(rng.choice(sites))
        X.append(make_subject(y, site)); Y.append(y); S.append(site)
    return (np.stack(X), np.array(Y), np.array(S))


# train + in-dist test from sites {0,1}; cross-site test from site {2}
Xtr, ytr, str_ = make_cohort(140, [0, 1])
Xid, yid, sid = make_cohort(50, [0, 1])
Xcs, ycs, scs = make_cohort(50, [2])
print(f"[data] train={len(ytr)} (sites 0,1)  test_in={len(yid)} (sites 0,1)  "
      f"test_cross={len(ycs)} (site 2) | identity/site dominate clinical "
      f"(scales clin={CLIN_SCALE}, id={ID_SCALE}, site={SITE_SCALE})")


def train_eval(disentangle, lam_adv=1.0, grl_max=1.0, grl_ramp=20, epochs=60,
               lam_nui=0.5, lam_ortho=0.1):
    torch.manual_seed(SEED)
    model = CIDA(n_channels=C, d=D, n_classes=2, n_subjects=len(ytr), n_sites=2, d_factor=32)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-3)
    Xt = t_f(Xtr); yt = t_l(ytr)
    subj = torch.arange(len(ytr)); site = t_l(str_)
    # class weights for the (possibly imbalanced) clinical loss
    pos = int(ytr.sum()); cw = torch.tensor([len(ytr) / (2 * (len(ytr) - pos) + 1e-6),
                                             len(ytr) / (2 * pos + 1e-6)], dtype=torch.float32)
    BS = 32
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(ytr))
        grl = min(grl_max, ep / grl_ramp) if disentangle else 0.0
        for i in range(0, len(ytr), BS):
            b = perm[i:i + BS]
            opt.zero_grad()
            out = model(Xt[b], grl_lambda=grl)
            if disentangle:
                loss, _ = model.loss(out, yt[b], subject=subj[b], site=site[b],
                                     lam_adv=lam_adv, lam_nui=lam_nui, lam_ortho=lam_ortho,
                                     class_weight=cw)
            else:
                loss = nn.functional.cross_entropy(out["logits"], yt[b], weight=cw)
            loss.backward(); opt.step()

    model.eval()
    def auc(X, y):
        with torch.no_grad():
            o = model(t_f(X))
            p = to_np(torch.softmax(o["logits"], -1)[:, 1])
        return roc_auc_score(y, p)

    # site-decodability of z_clin: probe site from the clinical factor (lower=better)
    with torch.no_grad():
        z_tr = to_np(model(Xt)["z_clin"])
        z_id = to_np(model(t_f(Xid))["z_clin"])
    probe = LogisticRegression(max_iter=2000, class_weight="balanced").fit(z_tr, str_)
    site_acc = probe.score(z_id, sid)                       # 0.5 = identity fully removed
    return auc(Xid, yid), auc(Xcs, ycs), site_acc


# Sweep: baseline + three disentanglement strengths (reproduces the design-doc table).
RUNS = [
    ("baseline (disentangle OFF)", dict(disentangle=False)),
    ("CIDA weak  (adv1, grl->1)",  dict(disentangle=True, lam_adv=1.0, grl_max=1.0, grl_ramp=20, epochs=60)),
    ("CIDA mod   (adv2, grl->1)",  dict(disentangle=True, lam_adv=2.0, grl_max=1.0, grl_ramp=30, epochs=150)),
    ("CIDA strong(adv5, grl->2)",  dict(disentangle=True, lam_adv=5.0, grl_max=2.0, grl_ramp=15, epochs=150)),
]
print("\nsetting                  in-dist AUC   cross-site AUC   site-decodability(z_clin)")
for name, kw in RUNS:
    a_id, a_cs, sacc = train_eval(**kw)
    print(f"  {name:<25}{a_id:>9.3f}{a_cs:>15.3f}{sacc:>22.3f}")
print("\n(site-decodability: 0.50 = identity fully removed from z_clin; lower is better.")
print(" GATE: promote disentanglement to a claim only when site-decodability -> ~0.5")
print(" AND cross-site AUC >= baseline. As of this run, it does NOT pass.)")
