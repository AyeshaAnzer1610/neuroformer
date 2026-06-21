#!/usr/bin/env python
"""
OcuFM — a foundation model for eye-tracking, built on the LaBraM recipe.

Fills the gap that EEG has a foundation model (LaBraM) while gaze has only
hand-crafted features. OcuFM mirrors LaBraM component-for-component:
  - gaze is multi-CHANNEL (x, y, pupil, velocity, acceleration), patched in time;
  - a VQ tokenizer encodes each patch to a discrete code, trained to reconstruct a
    KINEMATIC + SPECTRAL target (the gaze analog of LaBraM's Fourier target);
  - a Transformer is masked-patch pretrained (predict codes of masked patches);
  - the frozen embedding is then adapted downstream with CIDA (src/cida.py) — gaze
    is identity-dominated (eye-movement biometrics), the same reason EEG needs CIDA.

See docs/resubmission/OcuFM-Architecture-Design.md.

STATUS: proposed + runnable. This module is a self-contained skeleton that runs the
masked-patch pretraining step on SYNTHETIC gaze so the design is concrete and unit-
tested. Real pretraining (GazeBase-scale corpus) is the next step; we report no
accuracy gain until it is actually pretrained.
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Canonical gaze channels (the "montage" analog). Tracker-agnostic input layer
# resamples/derives these before patching.
GAZE_CHANNELS = ["x", "y", "pupil", "velocity", "acceleration"]


# --------------------------------------------------------------------------- #
# Kinematic + spectral target  (the gaze analog of LaBraM's Fourier target)
# --------------------------------------------------------------------------- #
def gaze_patch_target(patch):
    """Summarise a gaze patch [.., P] into a small kinematic+spectral target [.., 6].

    Targets the tokenizer is trained to reconstruct: mean, std, mean |Δ| (velocity),
    mean |ΔΔ| (acceleration), and the low/high spectral-band energy of the patch.
    These are cheap, differentiable, and capture saccade/fixation + pupil dynamics.
    """
    mean = patch.mean(-1)
    std = patch.std(-1)
    d1 = patch.diff(dim=-1).abs().mean(-1)
    d2 = patch.diff(dim=-1).diff(dim=-1).abs().mean(-1)
    spec = torch.fft.rfft(patch, dim=-1).abs()                 # [.., P//2+1]
    half = spec.shape[-1] // 2 or 1
    lo = spec[..., :half].mean(-1)
    hi = spec[..., half:].mean(-1)
    return torch.stack([mean, std, d1, d2, lo, hi], dim=-1)    # [.., 6]


# --------------------------------------------------------------------------- #
# Vector-quantizer (codebook + straight-through estimator)  — the "tokenizer"
# --------------------------------------------------------------------------- #
class VectorQuantizer(nn.Module):
    """VQ-VAE codebook. Maps continuous patch encodings to nearest codebook entry
    with a straight-through gradient; returns quantized vectors + codes + VQ loss."""

    def __init__(self, n_codes=512, dim=64, beta=0.25):
        super().__init__()
        self.codebook = nn.Embedding(n_codes, dim)
        self.codebook.weight.data.uniform_(-1.0 / n_codes, 1.0 / n_codes)
        self.beta = beta

    def forward(self, z):                                       # z: [N, dim]
        d = (z.pow(2).sum(1, keepdim=True)
             - 2 * z @ self.codebook.weight.t()
             + self.codebook.weight.pow(2).sum(1))              # [N, n_codes]
        codes = d.argmin(1)                                     # [N]
        zq = self.codebook(codes)                               # [N, dim]
        vq_loss = F.mse_loss(zq.detach(), z) + self.beta * F.mse_loss(zq, z.detach())
        zq = z + (zq - z).detach()                              # straight-through
        return zq, codes, vq_loss


# --------------------------------------------------------------------------- #
# Patch embedding + Transformer encoder
# --------------------------------------------------------------------------- #
class GazePatchEmbed(nn.Module):
    """Per-channel time patches -> tokens, + channel and temporal embeddings."""

    def __init__(self, n_channels, patch_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len, d_model)
        self.chan_emb = nn.Parameter(torch.randn(n_channels, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, 4096, d_model) * 0.02)  # temporal (per patch index)

    def forward(self, x):                                       # x: [B, C, T]
        B, C, T = x.shape
        n_patch = T // self.patch_len
        p = x[:, :, : n_patch * self.patch_len].reshape(B, C, n_patch, self.patch_len)
        tok = self.proj(p)                                      # [B, C, n_patch, d]
        tok = tok + self.chan_emb.view(1, C, 1, -1) + self.pos[:, :n_patch].view(1, 1, n_patch, -1)
        return tok, p                                           # tokens + raw patches (for targets)


class OcuFM(nn.Module):
    """Gaze foundation model: patch-embed -> VQ tokenizer -> masked-patch Transformer.

    Pretraining objective (LaBraM-style): mask a fraction of patches, run the encoder,
    and from masked positions (a) predict the discrete code (cross-entropy to the
    tokenizer's argmin code) and (b) reconstruct the kinematic+spectral target.
    """

    def __init__(self, n_channels=len(GAZE_CHANNELS), patch_len=64, d_model=64,
                 n_layers=4, n_heads=4, n_codes=512, target_dim=6, mask_ratio=0.5):
        super().__init__()
        self.embed = GazePatchEmbed(n_channels, patch_len, d_model)
        self.tok_enc = nn.Sequential(nn.Linear(patch_len, d_model), nn.GELU(),
                                     nn.Linear(d_model, d_model))   # patch -> VQ input
        self.vq = VectorQuantizer(n_codes, d_model)
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.code_head = nn.Linear(d_model, n_codes)            # predict discrete code
        self.recon_head = nn.Linear(d_model, target_dim)        # reconstruct kin+spec target
        self.mask_ratio = mask_ratio

    def tokenize(self, patches):
        """Patches [N, P] -> (quantized z, codes, vq_loss). The frozen 'tokenizer'."""
        return self.vq(self.tok_enc(patches))

    def forward(self, x):
        """x: [B, C, T] raw gaze. Returns pretraining loss + parts (synthetic-ready)."""
        tok, patches = self.embed(x)                            # [B,C,n,d], [B,C,n,P]
        B, C, n, d = tok.shape
        flat = tok.reshape(B, C * n, d)
        raw = patches.reshape(B * C * n, -1)

        # tokenizer codes + targets (teacher signal for masked positions)
        with torch.no_grad():
            _, codes, _ = self.tokenize(raw)                    # [B*C*n]
        _, _, vq_loss = self.tokenize(raw)                      # keep VQ commitment trainable
        target = gaze_patch_target(raw)                         # [B*C*n, 6]

        # random masking of patch tokens
        N = flat.shape[1]
        m = (torch.rand(B, N, device=x.device) < self.mask_ratio)
        m[(~m).all(1), 0] = True                                # ensure >=1 masked row
        h = flat.clone()
        h[m] = self.mask_token
        enc = self.encoder(h)                                   # [B, N, d]

        sel = enc[m]                                            # [n_masked, d]
        codes_b = codes.reshape(B, N)[m]
        target_b = target.reshape(B, N, -1)[m]
        loss_code = F.cross_entropy(self.code_head(sel), codes_b)
        loss_recon = F.mse_loss(self.recon_head(sel), target_b)
        loss = loss_code + loss_recon + vq_loss
        return {"loss": loss, "loss_code": loss_code.item(),
                "loss_recon": loss_recon.item(), "vq_loss": vq_loss.item(),
                "n_masked": int(m.sum())}

    @torch.no_grad()
    def embed_windows(self, x):
        """Frozen inference: gaze [B, C, T] -> ENCODED per-channel tokens [B, n, C, d].
        Runs the pretrained Transformer encoder (no masking) — the learned representation,
        not the raw patch projection. (Bugfix: previously returned un-encoded patches.)"""
        tok, _ = self.embed(x)                                  # [B, C, n, d]
        B, C, n, d = tok.shape
        enc = self.encoder(tok.reshape(B, C * n, d))            # learned repr [B, C*n, d]
        return enc.reshape(B, C, n, d).permute(0, 2, 1, 3).contiguous()  # [B, n, C, d]


if __name__ == "__main__":  # self-test: masked-patch pretraining on synthetic gaze
    torch.manual_seed(0)
    B, C, T = 8, len(GAZE_CHANNELS), 64 * 10                    # 8 subjects, 5 ch, 10 patches
    x = torch.randn(B, C, T)
    model = OcuFM(n_channels=C, patch_len=64, d_model=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(5):
        opt.zero_grad()
        out = model(x)
        out["loss"].backward(); opt.step()
    print(f"pretrain OK | loss {out['loss'].item():.3f} "
          f"(code {out['loss_code']:.3f}, recon {out['loss_recon']:.3f}, "
          f"vq {out['vq_loss']:.3f}) | masked {out['n_masked']} patches")
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"OcuFM params: {params:.2f}M")

    # frozen embeddings → the [B, W, C, d] tensor CIDA consumes (gaze branch)
    emb = model.embed_windows(x)
    print(f"frozen gaze embeddings for CIDA: {tuple(emb.shape)}  (B, windows, channels, d)")
