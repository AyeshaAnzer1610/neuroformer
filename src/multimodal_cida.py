#!/usr/bin/env python
"""
Two-foundation-model screener: the assembled 2-modality architecture.

Ties the pieces together end-to-end and shows the unifying claim in code — ONE
disentanglement adapter (CIDA) serves BOTH identity-dominated foundation-model
representations:

    EEG  --(frozen LaBraM)-->  channel/window tokens --(CIDA)--> z_clin_eeg
    gaze --(frozen OcuFM )-->  channel/window tokens --(CIDA)--> z_clin_eye
                                                  └------> MaskedSetFusion -> logit

- Each branch's CIDA disentangles subject/site identity from the clinical signal and
  returns an identity-invariant clinical vector `z_clin`.
- The two `z_clin` vectors are fused by the existing missing-modality MaskedSetFusion
  (availability mask + modality dropout), so the screener runs on EEG-only, eye-only,
  or both. Behaviour is gone.

Backbones are NOT trained here (frozen LaBraM / proposed-frozen OcuFM); the trainable
parts are the two CIDA adapters + the tiny fusion head. This module consumes the
backbones' token tensors so it unit-tests without braindecode/CUDA.
"""
import torch
import torch.nn as nn

from cida import CIDA
from fusion_model import MaskedSetFusion


class TwoFMScreener(nn.Module):
    """CIDA-on-LaBraM (EEG) + CIDA-on-OcuFM (gaze) -> missing-modality fusion."""

    def __init__(self, eeg_channels, eeg_d, eye_channels, eye_d,
                 n_subjects=1, n_sites=1, d_factor=64, n_classes=2):
        super().__init__()
        self.eeg_cida = CIDA(eeg_channels, eeg_d, n_classes, n_subjects, n_sites, d_factor)
        self.eye_cida = CIDA(eye_channels, eye_d, n_classes, n_subjects, n_sites, d_factor)
        # Fuse the two identity-invariant clinical vectors (z_clin), each d_factor-dim.
        self.fusion = MaskedSetFusion({"eeg": d_factor, "eye": d_factor})

    def forward(self, eeg_tokens, eye_tokens, avail_mask,
                eeg_win_mask=None, eye_win_mask=None, grl_lambda=1.0):
        """eeg_tokens/eye_tokens: [B, W, C, d] per modality (None -> zeros if absent).
        avail_mask: [B, 2] (1=present) for [eeg, eye]. Returns logit + per-branch aux."""
        out_eeg = self.eeg_cida(eeg_tokens, eeg_win_mask, grl_lambda=grl_lambda)
        out_eye = self.eye_cida(eye_tokens, eye_win_mask, grl_lambda=grl_lambda)
        feats = {"eeg": out_eeg["z_clin"], "eye": out_eye["z_clin"]}
        logit = self.fusion(feats, avail_mask)
        return {"logit": logit, "eeg": out_eeg, "eye": out_eye}


if __name__ == "__main__":  # self-test: end-to-end 2-FM forward + backward
    torch.manual_seed(0)
    B = 8
    Ceeg, deeg, Weeg = 19, 200, 6       # LaBraM-side tokens
    Ceye, deye, Weye = 5, 64, 10        # OcuFM-side tokens (matches ocufm.embed_windows)
    eeg_tokens = torch.randn(B, Weeg, Ceeg, deeg)
    eye_tokens = torch.randn(B, Weye, Ceye, deye)
    avail = torch.ones(B, 2); avail[0, 1] = 0; avail[1, 0] = 0   # subj0 EEG-only, subj1 eye-only
    y = torch.randint(0, 2, (B,)).float()

    model = TwoFMScreener(Ceeg, deeg, Ceye, deye, n_subjects=B, n_sites=2)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(5):
        opt.zero_grad()
        out = model(eeg_tokens, eye_tokens, avail)
        loss = lossf(out["logit"], y)
        loss.backward(); opt.step()
    print(f"two-FM forward+backward OK | fusion loss {loss.item():.3f}")

    model.eval()
    with torch.no_grad():
        out = model(eeg_tokens, eye_tokens, avail)
        # EEG-only and eye-only inference via the availability mask
        eeg_only = model(eeg_tokens, eye_tokens, torch.tensor([[1.0, 0.0]] * B))["logit"]
        eye_only = model(eeg_tokens, eye_tokens, torch.tensor([[0.0, 1.0]] * B))["logit"]
    print(f"fused logits {tuple(out['logit'].shape)} | EEG-only {tuple(eeg_only.shape)} "
          f"| eye-only {tuple(eye_only.shape)}  (one model, any subset)")
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params (2x CIDA + fusion head): {train:,}  "
          f"(both LaBraM + OcuFM backbones frozen, not counted)")
