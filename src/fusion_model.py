#!/usr/bin/env python
"""
Phase 2 — masked set-fusion model for missing-modality screening.

One small model fuses any subset of {EEG, eye, behaviour}:
- each present modality -> linear projection to a shared dim + a learned
  per-modality (availability/identity) embedding -> a token;
- **masked mean pooling** over *present* tokens (absent modalities contribute
  nothing) -> small MLP head -> logit;
- **modality dropout** during training randomly hides present modalities (never
  all of them) so the same weights work for any availability pattern.

Deliberately tiny (shared dim ~32, one hidden layer) for N~100; the contribution
is the missing-modality *mechanism*, not model capacity.
"""
import torch
import torch.nn as nn


class MaskedSetFusion(nn.Module):
    def __init__(self, dims, d=48, hidden=48, p_moddrop=0.2, p_dropout=0.3):
        """dims: {modality_name: input_dim}."""
        super().__init__()
        self.mods = list(dims)
        self.proj = nn.ModuleDict({m: nn.Linear(dims[m], d) for m in self.mods})
        # learned per-modality identity/availability token (added to each modality)
        self.mod_emb = nn.Parameter(torch.randn(len(self.mods), d) * 0.02)
        # learned attention query: lets the model WEIGHT modalities (attend to
        # informative ones, down-weight noise) instead of averaging them — a mean
        # pool forces equal weight and lets near-chance modalities dilute EEG.
        self.query = nn.Parameter(torch.randn(d) * 0.02)
        self.p_moddrop = p_moddrop
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, hidden), nn.ReLU(),
            nn.Dropout(p_dropout), nn.Linear(hidden, 1),
        )

    def forward(self, x, mask):
        """x: {modality: [B, dim]}, mask: [B, M] (1=present). Returns logits [B]."""
        toks = torch.stack(
            [self.proj[m](x[m]) + self.mod_emb[j] for j, m in enumerate(self.mods)],
            dim=1,
        )                                                  # [B, M, d]
        m = mask.float()
        if self.training and self.p_moddrop > 0:
            keep = (torch.rand_like(m) > self.p_moddrop).float()
            dropped = m * keep
            # never drop every present modality for a sample
            empty = dropped.sum(1) == 0
            dropped[empty] = m[empty]
            m = dropped
        # masked attention pooling over present modality tokens
        scores = (toks * self.query).sum(-1) / (toks.shape[-1] ** 0.5)   # [B, M]
        scores = scores.masked_fill(m == 0, float("-inf"))
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)                # [B, M, 1]
        pooled = (toks * attn).sum(1)                                    # [B, d]
        return self.head(pooled).squeeze(-1)               # [B]


if __name__ == "__main__":  # self-test: forward + one optimizer step
    torch.manual_seed(0)
    dims = {"eeg": 400, "eye": 11, "behaviour": 6}
    B = 16
    x = {m: torch.randn(B, d) for m, d in dims.items()}
    mask = (torch.rand(B, 3) > 0.3)
    mask[mask.sum(1) == 0, 0] = True                       # ensure >=1 present
    model = MaskedSetFusion(dims)
    y = torch.randint(0, 2, (B,)).float()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(3):
        opt.zero_grad()
        logit = model(x, mask)
        loss = lossf(logit, y)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(x, mask)
    print(f"forward OK: logits {tuple(out.shape)}; final loss {loss.item():.3f}")
    # eval with a modality fully masked out (simulated missingness)
    m2 = mask.clone(); m2[:, 0] = False
    with torch.no_grad():
        out2 = model(x, m2)
    print(f"missing-EEG forward OK: logits {tuple(out2.shape)}")
