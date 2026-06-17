#!/usr/bin/env python
"""
CIDA — Connectivity-aware, Identity-Disentangled Adapter for LaBraM.

A new *method* contribution on top of a FROZEN LaBraM backbone (not a new
foundation model). It targets LaBraM's one root weakness — its representation
entangles subject identity + recording domain with the clinical signal, which
explains our leakage (0.91), our full-fine-tune overfit (0.69), and our external
collapse (~0.50). See docs/resubmission/CIDA-Architecture-Design.md.

Four trainable blocks wrap the frozen backbone:
  (1) ConnectivityGraphAdapter — GNN over per-channel LaBraM tokens; injects the
      cross-channel connectivity its per-channel spectral tokenizer omits.
  (2) LoRA — rank-r updates on the backbone's attention (injected at integration
      time into the real LaBraM; LoRALinear here is the building block + a
      drop-in adapter for unit-testing the parameter budget).
  (3) AttentionTemporalPool — set-transformer pooling over a subject's windows
      (replaces mean-pool); emits a pooled vector + an uncertainty scalar.
  (4) DisentanglementHead — splits the representation into z_clin / z_nuis;
      clinical head on z_clin, subject+site adversaries on z_clin via a
      gradient-reversal layer (invariance), nuisance head on z_nuis (absorbs
      identity), and an orthogonality penalty keeping the two factors disjoint.

This module is backbone-agnostic in its *interface* (it consumes LaBraM token
tensors) so it can be unit-tested without the heavy braindecode/CUDA stack; the
real wiring to InterpolatedLaBraM(return_features=True) happens in the training
script (see `forward` docstring + the design doc §3).
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# (2) LoRA building block
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    """Frozen base Linear + trainable rank-`r` low-rank update (B @ A).

    At integration this wraps LaBraM's attention q/k/v/proj layers: the base
    weight is the pretrained (frozen) weight, only A and B train. Here it doubles
    as a standalone adapter layer for the self-test / parameter-budget accounting.
    """

    def __init__(self, in_features, out_features, r=8, alpha=16, base: Optional[nn.Linear] = None):
        super().__init__()
        self.base = base if base is not None else nn.Linear(in_features, out_features)
        for p in self.base.parameters():
            p.requires_grad = False                       # frozen backbone weight
        self.A = nn.Parameter(torch.zeros(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + self.scaling * (x @ self.A.t() @ self.B.t())


# --------------------------------------------------------------------------- #
# (1) Connectivity graph adapter over channel tokens
# --------------------------------------------------------------------------- #
class ConnectivityGraphAdapter(nn.Module):
    """One message-passing layer over electrodes (nodes = channels).

    Input  : per-channel tokens [B, C, d]  (LaBraM channel tokens, patch-pooled).
    Adjacency: a LEARNED, symmetric, softmax-normalised edge matrix [C, C],
    optionally biased by a precomputed connectivity prior (PLV/coherence) passed
    as `prior` [B, C, C] or [C, C]. Output: connectivity-refined tokens [B, C, d].
    """

    def __init__(self, n_channels, d, dropout=0.2):
        super().__init__()
        self.edge = nn.Parameter(torch.zeros(n_channels, n_channels))  # learned adjacency logits
        self.msg = nn.Linear(d, d)
        self.upd = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x, prior: Optional[torch.Tensor] = None):
        logits = self.edge
        if prior is not None:
            logits = logits + prior                       # connectivity prior as an additive bias
        A = torch.softmax(logits, dim=-1)                 # [.,C,C] row-normalised
        agg = A @ self.msg(x)                             # message passing: [B,C,d]
        return x + self.upd(agg)                          # residual update


# --------------------------------------------------------------------------- #
# (3) Attention temporal pooling over a subject's windows
# --------------------------------------------------------------------------- #
class AttentionTemporalPool(nn.Module):
    """Learned-query attention pool over window vectors -> (pooled, uncertainty).

    Input : per-window vectors [B, W, d]  (W = windows for one subject in a batch
            row; pad + mask for ragged W). Uncertainty = attention entropy,
            normalised to [0,1], a cheap dispersion signal for the triage head.
    """

    def __init__(self, d):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d) * 0.02)
        self.proj = nn.Linear(d, d)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        scores = (self.proj(x) * self.query).sum(-1) / (x.shape[-1] ** 0.5)  # [B,W]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=1)                                  # [B,W]
        pooled = (x * attn.unsqueeze(-1)).sum(1)                             # [B,d]
        ent = -(attn.clamp_min(1e-9) * attn.clamp_min(1e-9).log()).sum(1)
        uncertainty = ent / torch.log(torch.tensor(x.shape[1], dtype=x.dtype, device=x.device))
        return pooled, uncertainty


# --------------------------------------------------------------------------- #
# (4) Disentanglement head (the novel core)
# --------------------------------------------------------------------------- #
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x, lambd=1.0):
    """Identity forward; sign-flipped, scaled gradient backward (DANN)."""
    return _GradReverse.apply(x, lambd)


class DisentanglementHead(nn.Module):
    """Split z into z_clin / z_nuis and expose the four training signals.

    - clinical classifier on z_clin                       (keep clinical info)
    - subject + site discriminators on GRL(z_clin)        (push identity OUT)
    - nuisance classifier on z_nuis                        (absorb identity)
    - orthogonality(z_clin, z_nuis)                        (disjoint factors)
    """

    def __init__(self, d, n_classes=2, n_subjects=1, n_sites=1, d_factor=64):
        super().__init__()
        self.to_clin = nn.Linear(d, d_factor)
        self.to_nuis = nn.Linear(d, d_factor)
        self.clinical = nn.Sequential(nn.LayerNorm(d_factor), nn.Linear(d_factor, n_classes))
        self.subj_adv = nn.Linear(d_factor, n_subjects)    # on GRL(z_clin)
        self.site_adv = nn.Linear(d_factor, n_sites)       # on GRL(z_clin)
        self.subj_nui = nn.Linear(d_factor, n_subjects)    # on z_nuis (normal direction)

    def forward(self, z, grl_lambda=1.0):
        z_clin = self.to_clin(z)
        z_nuis = self.to_nuis(z)
        logits = self.clinical(z_clin)
        r = grad_reverse(z_clin, grl_lambda)
        out = {
            "logits": logits,
            "z_clin": z_clin,
            "z_nuis": z_nuis,
            "subj_adv": self.subj_adv(r),          # invariance: encoder fools this
            "site_adv": self.site_adv(r),
            "subj_nui": self.subj_nui(z_nuis),     # nuisance absorbs identity
        }
        return out

    @staticmethod
    def orthogonality_loss(z_clin, z_nuis):
        """Squared Frobenius norm of the cross-covariance — 0 when factors disjoint."""
        a = z_clin - z_clin.mean(0, keepdim=True)
        b = z_nuis - z_nuis.mean(0, keepdim=True)
        cov = a.t() @ b / max(a.shape[0] - 1, 1)
        return (cov ** 2).sum()


# --------------------------------------------------------------------------- #
# Full CIDA model (consumes frozen-LaBraM token tensors)
# --------------------------------------------------------------------------- #
class CIDA(nn.Module):
    """Connectivity-aware, Identity-Disentangled Adapter.

    Trainable wrapper over a FROZEN LaBraM backbone. The backbone is NOT inside
    this module (kept frozen + checkpointed separately, and LoRA-injected at
    integration); CIDA consumes the backbone's per-channel/per-window tokens.
    """

    def __init__(self, n_channels, d, n_classes=2, n_subjects=1, n_sites=1,
                 d_factor=64, lora_r=8):
        super().__init__()
        self.graph = ConnectivityGraphAdapter(n_channels, d)
        self.channel_mix = LoRALinear(d, d, r=lora_r)      # stands in for backbone LoRA in tests
        self.pool = AttentionTemporalPool(d)
        self.head = DisentanglementHead(d, n_classes, n_subjects, n_sites, d_factor)

    def forward(self, tokens, window_mask=None, prior=None, grl_lambda=1.0):
        """tokens: [B, W, C, d] — per (subject-row, window, channel) LaBraM token.

        Real wiring: tokens come from InterpolatedLaBraM(x, return_features=True)
        ("features", reshaped to channels x patches and patch-pooled to [.,C,d]),
        with the frozen backbone's attention LoRA-adapted. Here we accept the
        token tensor directly so the adapter is testable without braindecode.
        """
        B, W, C, d = tokens.shape
        x = tokens.reshape(B * W, C, d)
        x = self.graph(x, prior=prior)                     # (1) connectivity
        x = self.channel_mix(x)                            # (2) LoRA-style channel adaptation
        win_vec = x.mean(dim=1).reshape(B, W, d)           # channel-pool -> per-window vector
        pooled, uncertainty = self.pool(win_vec, window_mask)  # (3) temporal attention pool
        out = self.head(pooled, grl_lambda)                # (4) disentanglement
        out["uncertainty"] = uncertainty
        return out

    def loss(self, out, y, subject=None, site=None,
             lam_adv=1.0, lam_nui=1.0, lam_ortho=0.1, class_weight=None):
        """Combine the CIDA training signals into a scalar (see design doc §4)."""
        L = F.cross_entropy(out["logits"], y, weight=class_weight)
        parts = {"clin": L.item()}
        if subject is not None:
            la = F.cross_entropy(out["subj_adv"], subject)     # GRL already flips sign
            ln = F.cross_entropy(out["subj_nui"], subject)
            L = L + lam_adv * la + lam_nui * ln
            parts["subj_adv"] = la.item(); parts["subj_nui"] = ln.item()
        if site is not None:
            ls = F.cross_entropy(out["site_adv"], site)
            L = L + lam_adv * ls
            parts["site_adv"] = ls.item()
        lo = self.head.orthogonality_loss(out["z_clin"], out["z_nuis"])
        L = L + lam_ortho * lo
        parts["ortho"] = lo.item()
        return L, parts


def trainable_fraction(model, backbone_params):
    """Trainable params as a fraction of (CIDA + frozen backbone) — the PEFT pitch."""
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return train, train / (train + backbone_params)


if __name__ == "__main__":  # self-test: forward + a few optimizer steps + budget
    torch.manual_seed(0)
    B, W, C, d = 8, 6, 19, 200          # 8 subjects, 6 windows each, 19 ch, LaBraM d=200
    n_subjects, n_sites = B, 2
    tokens = torch.randn(B, W, C, d)
    window_mask = torch.ones(B, W); window_mask[0, 3:] = 0          # ragged windows
    y = torch.randint(0, 2, (B,))
    subject = torch.arange(B)                                       # each row a subject
    site = torch.randint(0, n_sites, (B,))

    model = CIDA(n_channels=C, d=d, n_classes=2, n_subjects=n_subjects, n_sites=n_sites)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    model.train()
    for step in range(5):
        opt.zero_grad()
        out = model(tokens, window_mask=window_mask, grl_lambda=min(1.0, 0.2 * (step + 1)))
        loss, parts = model.loss(out, y, subject=subject, site=site)
        loss.backward(); opt.step()
    print(f"forward+backward OK | final loss {loss.item():.3f} | parts={ {k: round(v,3) for k,v in parts.items()} }")

    model.eval()
    with torch.no_grad():
        out = model(tokens, window_mask=window_mask)
    print(f"logits {tuple(out['logits'].shape)}  uncertainty {tuple(out['uncertainty'].shape)} "
          f"in [{out['uncertainty'].min():.2f},{out['uncertainty'].max():.2f}]")

    # PEFT pitch: trainable params vs a frozen LaBraM-Base (~5.8M)
    train, frac = trainable_fraction(model, backbone_params=5_800_000)
    print(f"trainable CIDA params = {train:,}  ->  {frac*100:.1f}% of (CIDA + frozen LaBraM-Base)")
