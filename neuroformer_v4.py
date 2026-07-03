"""
NeuroFormer v4 — small-N calibrated (target <50K params)
=========================================================
Root-cause diagnosis of v3 failure:
  - Seed range [0.062, 0.637] within a SINGLE fold = catastrophic variance
  - Even 2-layer 8-head transformers (~400K params) are too large for n=18 positives
  - 8-seed averaging masked seeds that collapsed to 0.06 AUC completely

Root constraint: 18 positive / 80 negative per training fold
  Rule of thumb: max ~10 free parameters per training example for stability
  Total safe: <50K parameters (v3 had 2.1M, original had 1.7M)

Strategy:
  1. Replace ALL transformer encoders with single-layer MLPs
  2. Keep architectural contribution (channel attention, freq mixing)
     but implement as lightweight linear operations
  3. Cross-modal fusion becomes learned weighted sum + 1 MLP layer
  4. Total params: ~25K — matches what LMF baseline (0.708) actually does
  5. Conservative training: lr=2e-4, OneCycleLR, early stopping, label smoothing
"""

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from pathlib import Path
import pandas as pd, time

print("\n" + "="*80)
print("NeuroFormer v4 — small-N calibrated (target <50K params)")
print("="*80 + "\n")

# ── Data loading ─────────────────────────────────────────────────────────────
labels = pd.read_csv('data/labels.csv', index_col='ID')
mask = labels['DX_Status'] < 2
subs = labels[mask].index.tolist()
y = labels.loc[subs, 'DX_Status'].values
print(f"Subjects: {len(subs)}, Pos: {(y==1).sum()}, Neg: {(y==0).sum()}")

eeg = {}
for s in subs:
    p = f'data/processed_handcrafted/{s}_handcrafted.npz'
    eeg[s] = np.load(p)['features'].mean(axis=0) if Path(p).exists() else np.zeros(516)
Xe = np.array([eeg[s] for s in subs])

eye_df = pd.read_csv('data/eye_features.csv', index_col=0)
Xy = np.array([eye_df.loc[s].values if s in eye_df.index else np.zeros(11) for s in subs])

beh = {}
for s in subs:
    p = f'data/processed_beh/{s}_behaviour.npz'
    if Path(p).exists():
        feat = np.load(p)['features']
        beh[s] = feat.squeeze() if feat.ndim > 1 else feat
    else:
        beh[s] = np.zeros(6)
Xb = np.array([beh[s] for s in subs])

print(f"EEG: {Xe.shape} | Eye: {Xy.shape} | Beh: {Xb.shape}")

# Dataset-level normalisation
Xe = (Xe - Xe.mean(0)) / (Xe.std(0) + 1e-6)
Xy = (Xy - Xy.mean(0)) / (Xy.std(0) + 1e-6)
Xb = (Xb - Xb.mean(0)) / (Xb.std(0) + 1e-6)

# Reshape EEG: (N,516) → (N,5,103)
N_CH, FPC = 5, 516 // 5
Xe_r = Xe[:, :N_CH*FPC].reshape(-1, N_CH, FPC)
print(f"EEG reshaped: {Xe_r.shape}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE  (~25K parameters total)
# ═══════════════════════════════════════════════════════════════════════════════

D = 32   # small but sufficient for n=98

class LightChannelAttention(nn.Module):
    """Lightweight channel attention: 5 channels → weighted sum. ~330 params."""
    def __init__(self, d=D, n_ch=5):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.Tanh(),
            nn.Linear(d // 2, 1),
        )

    def forward(self, x):
        # x: (B, n_ch, D)
        scores = self.gate(x).squeeze(-1)          # (B, n_ch)
        weights = torch.softmax(scores, dim=-1)    # (B, n_ch)
        attended = (weights.unsqueeze(-1) * x).sum(dim=1)  # (B, D)
        return attended, weights

class FreqGroupMixer(nn.Module):
    """Frequency-group interaction: learned cross-band mixing. ~128 params."""
    def __init__(self, d=D, n_bands=4):
        super().__init__()
        assert d % n_bands == 0
        self.n_bands = n_bands
        self.band_dim = d // n_bands
        self.mix = nn.Parameter(torch.eye(n_bands) + 0.01 * torch.randn(n_bands, n_bands))
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        # x: (B, D)
        B = x.shape[0]
        bands = x.view(B, self.n_bands, self.band_dim)
        mix = torch.softmax(self.mix, dim=-1)
        mixed = torch.einsum('ij,bjd->bid', mix, bands)
        mixed = mixed.reshape(B, -1)
        return self.norm(x + mixed)

class EEGEncoderV4(nn.Module):
    """Compact EEG encoder: project → channel attention → freq mixing. ~3K params."""
    def __init__(self, n_ch=5, feat_per_ch=103, d=D):
        super().__init__()
        self.proj = nn.Linear(feat_per_ch, d)
        self.norm0 = nn.LayerNorm(d)
        self.ch_attn = LightChannelAttention(d, n_ch)
        self.freq_mix = FreqGroupMixer(d, n_bands=4)
        self.out = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU())

    def forward(self, x):
        # x: (B, 5, 103)
        h = self.norm0(self.proj(x))        # (B, 5, D)
        h, ch_weights = self.ch_attn(h)    # (B, D), (B, 5)
        h = self.freq_mix(h)                # (B, D)
        return self.out(h), ch_weights

class ModalityMLPEncoder(nn.Module):
    """Minimal MLP encoder for eye/behaviour. ~1K params."""
    def __init__(self, in_dim, d=D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d * 2),
            nn.LayerNorm(d * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d * 2, d),
            nn.LayerNorm(d),
        )

    def forward(self, x):
        return self.net(x)

class CrossModalFusionV4(nn.Module):
    """Lightweight fusion: gated weighted sum + refinement MLP. ~600 params."""
    def __init__(self, d=D, n_mod=3):
        super().__init__()
        self.importance = nn.Linear(d, 1, bias=False)
        self.log_temp = nn.Parameter(torch.tensor(0.0))
        self.refine = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d * 2, d),
            nn.LayerNorm(d),
        )

    def forward(self, modalities):
        # modalities: list of (B, D) tensors
        stack = torch.stack(modalities, dim=1)          # (B, n_mod, D)
        scores = self.importance(stack).squeeze(-1)     # (B, n_mod)
        temp = torch.exp(self.log_temp).clamp(0.1, 5.0)
        weights = torch.softmax(scores / temp, dim=-1)  # (B, n_mod)
        fused = (weights.unsqueeze(-1) * stack).sum(1)  # (B, D)
        return self.refine(fused) + fused, weights

class NeuroFormerV4(nn.Module):
    """Full model. ~25K parameters."""
    def __init__(self, n_ch=5, fpc=103, eye_dim=11, beh_dim=6, d=D):
        super().__init__()
        self.eeg_enc = EEGEncoderV4(n_ch, fpc, d)
        self.eye_enc = ModalityMLPEncoder(eye_dim, d)
        self.beh_enc = ModalityMLPEncoder(beh_dim, d)
        self.fusion  = CrossModalFusionV4(d)
        self.head = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d * 2, 2),
        )

    def forward(self, eeg, eye, beh):
        e, ch_w  = self.eeg_enc(eeg)
        y        = self.eye_enc(eye)
        b        = self.beh_enc(beh)
        fused, mod_w = self.fusion([e, y, b])
        return self.head(fused), ch_w, mod_w

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

# Parameter check
_tmp = NeuroFormerV4(N_CH, FPC, Xy.shape[1], Xb.shape[1])
print(f"Model parameters: {_tmp.count_params():,}  (target <50K)")
del _tmp


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

device = "cuda" if torch.cuda.is_available() else "cpu"
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
fold_aucs = []
t0 = time.time()

print("="*80)
print("5x5 repeated subject-level stratified CV, 8 seeds per fold")
print("="*80 + "\n")

N_SEEDS  = 8
EPOCHS   = 80
BS       = 8
LR       = 2e-4
WD       = 1e-3

for fold_idx, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
    seed_aucs = []

    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        model = NeuroFormerV4(N_CH, FPC, Xy.shape[1], Xb.shape[1], d=D).to(device)

        pos_n = int((y[tr] == 1).sum())
        neg_n = int(len(y[tr]) - pos_n)
        cw = torch.tensor([1.0, neg_n / max(pos_n, 1)], dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)

        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=LR * 5,
            total_steps=EPOCHS,
            pct_start=0.2,
            anneal_strategy='cos',
        )

        best_loss, patience = 1e9, 0

        for epoch in range(EPOCHS):
            model.train()
            perm = np.random.permutation(len(tr))
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, len(perm), BS):
                idx = tr[perm[i:i+BS]]
                e_b = torch.from_numpy(Xe_r[idx]).float().to(device)
                y_b = torch.from_numpy(Xy[idx]).float().to(device)
                b_b = torch.from_numpy(Xb[idx]).float().to(device)
                y_t = torch.tensor(y[idx], dtype=torch.long, device=device)

                opt.zero_grad()
                logits, _, _ = model(e_b, y_b, b_b)
                loss = loss_fn(logits, y_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1

            sch.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience = 0
            else:
                patience += 1
            if patience >= 20:
                break

        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(te), BS):
                idx = te[i:i+BS]
                e_b = torch.from_numpy(Xe_r[idx]).float().to(device)
                y_b = torch.from_numpy(Xy[idx]).float().to(device)
                b_b = torch.from_numpy(Xb[idx]).float().to(device)
                logits, _, _ = model(e_b, y_b, b_b)
                probs.extend(torch.softmax(logits, -1)[:, 1].cpu().numpy())

        auc = roc_auc_score(y[te], probs[:len(te)])
        seed_aucs.append(auc)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    mean_auc = float(np.mean(seed_aucs))
    fold_aucs.append(mean_auc)
    sr = f"[{min(seed_aucs):.3f}, {max(seed_aucs):.3f}]"
    print(f"Fold {fold_idx+1:2d}/25: {mean_auc:.4f}  seed range {sr}")

arr = np.array(fold_aucs)
elapsed = time.time() - t0

print(f"\n{'='*80}")
print(f"FINAL RESULT: {arr.mean():.4f} +/- {arr.std():.4f}")
print(f"\nComparison:")
print(f"  NeuroFormer v4:        {arr.mean():.4f} +/- {arr.std():.4f}")
print(f"  Original NeuroFormer:  0.6240 +/- 0.0200")
print(f"  LMF baseline:          0.7070 +/- 0.0240")
print(f"  Demographics:          0.7170 +/- 0.0220")
print(f"\nFold std: {arr.std():.4f}  (target <0.08)")
print(f"Min: {arr.min():.4f}  Max: {arr.max():.4f}")
print(f"Time: {elapsed/60:.1f} min")
print('='*80)

delta = arr.mean() - 0.624
if arr.mean() > 0.717:
    print("\nBEATS demographics baseline!\n")
elif arr.mean() > 0.708:
    print("\nBEATS LMF baseline!\n")
elif arr.mean() > 0.624:
    print(f"\nBEATS original NeuroFormer by +{delta:.3f}\n")
else:
    print(f"\nGap vs original: {-delta:.4f}\n")
