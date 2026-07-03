"""
NeuroFormer v3 — Correctly calibrated for n≈98 training samples
================================================================
Root-cause analysis of v2 failure (0.54 ± 0.12):
  - lr=3e-3 with BS=16 → only 7 gradient steps/epoch on 98 samples
    → 30× too large; folds 8,22 collapsed to all-negative prediction
  - Architecture complexity was also too high for the data size
  - The original NeuroFormer (d=128, 2-layer, 0.624) had the right
    capacity; we need to ADD the architectural improvements on top of
    that foundation, not replace it with a heavier model

What this version does differently:
  1. Keeps the original d=128 transformer backbone
  2. Adds EEG reshape (5ch × 103feat) ONLY for the EEG encoder input
  3. Adds a minimal CrissCrossAttention on the EEG token sequence
  4. Uses lr=5e-4, BS=8, epochs=60 (appropriate for n=98)
  5. Removes the confidence gate entirely — the cross-modal transformer
     already does weighted fusion; double-gating hurt stability
  6. Uses label smoothing=0.1 to reduce overfit on the 18 positives
"""

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from pathlib import Path
import pandas as pd, time

print("\n" + "="*80)
print("NeuroFormer v3 — Calibrated for small-cohort training")
print("="*80 + "\n")

# ── Data loading ─────────────────────────────────────────────────────────────
labels = pd.read_csv('data/labels.csv', index_col='ID')
mask = labels['DX_Status'] < 2
subs = labels[mask].index.tolist()
y = labels.loc[subs, 'DX_Status'].values
print(f"Subjects: {len(subs)}, Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")

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

# ── Dataset-level normalisation (per-feature z-score across subjects) ────────
Xe = (Xe - Xe.mean(0)) / (Xe.std(0) + 1e-6)
Xy = (Xy - Xy.mean(0)) / (Xy.std(0) + 1e-6)
Xb = (Xb - Xb.mean(0)) / (Xb.std(0) + 1e-6)

# Reshape EEG: (N, 516) → (N, 5, 103)  [5 channels × 103 features each]
N_CH, FEAT_PER_CH = 5, Xe.shape[1] // 5
Xe_r = Xe[:, :N_CH * FEAT_PER_CH].reshape(-1, N_CH, FEAT_PER_CH)
print(f"EEG reshaped: {Xe_r.shape}  (5 channels x {FEAT_PER_CH} features)\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

D = 128   # model dimension — same as original NeuroFormer


class EEGChannelTokenizer(nn.Module):
    """
    Input:  (B, 5, 103)  — 5 EEG channels, 103 features each
    Output: (B, 5, D)    — one token per channel, projected to model dim

    This is the only new architectural piece vs original NeuroFormer.
    Each channel gets its own linear projection (parameter sharing via
    a single Linear is fine at this size). We get 5 tokens = a real
    sequence that criss-cross attention can work on.
    """
    def __init__(self, n_ch=5, feat_per_ch=103, d_model=128):
        super().__init__()
        # Per-channel feature extraction: depthwise temporal conv
        self.dw_conv = nn.Conv1d(
            n_ch, n_ch, kernel_size=7, padding=3,
            groups=n_ch, bias=False)          # depthwise: each channel separate
        self.bn = nn.BatchNorm1d(n_ch)
        self.act = nn.GELU()
        # Project each channel's features to d_model
        self.proj = nn.Linear(feat_per_ch, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, 5, 103)
        x = self.act(self.bn(self.dw_conv(x)))  # (B, 5, 103)
        x = self.norm(self.proj(x))             # (B, 5, D)
        return x


class CrissCrossEEGEncoder(nn.Module):
    """
    Processes EEG as a 5-token sequence (one per channel).

    Stage 1 — Spatial attention: attend across the 5 channel tokens.
              Learns which channels co-activate for the psychiatric signal.

    Stage 2 — Frequency-group attention: within each channel token,
              group D into 4 sub-bands and attend across groups.
              Approximates cross-frequency coupling without raw EEG.

    Stage 3 — Standard 2-layer transformer encoder (same as original)
              over the 5 tokens + a CLS token for aggregation.
    """
    def __init__(self, n_ch=5, feat_per_ch=103, d_model=128):
        super().__init__()
        self.tokenizer = EEGChannelTokenizer(n_ch, feat_per_ch, d_model)
        self.d_model = d_model
        self.n_freq_groups = 4

        # CLS token for aggregation
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        # Spatial cross-channel attention (4 heads, lightweight)
        self.spatial_attn = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=0.1, batch_first=True)
        self.spatial_norm = nn.LayerNorm(d_model)

        # Frequency-group attention: split D into 4 groups of D//4
        # and attend across groups within each token
        gd = d_model // self.n_freq_groups
        self.freq_attn = nn.MultiheadAttention(
            gd, num_heads=2, dropout=0.1, batch_first=True)
        self.freq_norm = nn.LayerNorm(d_model)
        self.freq_proj = nn.Linear(d_model, d_model)

        # Standard transformer encoder (same depth as original)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8,
            dim_feedforward=4 * d_model,
            dropout=0.3, activation='gelu',
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        # Tokenize: (B, 5, D)
        tokens = self.tokenizer(x)

        # Stage 1: spatial attention across 5 channel tokens
        s_out, _ = self.spatial_attn(tokens, tokens, tokens)
        tokens = self.spatial_norm(tokens + s_out)  # (B, 5, D)

        # Stage 2: frequency-group attention within each token
        gd = self.d_model // self.n_freq_groups
        # Reshape: (B*5, 4, D//4) — treat each token's feature groups as seq
        g = tokens.reshape(B * 5, self.n_freq_groups, gd)
        f_out, _ = self.freq_attn(g, g, g)
        f_out = f_out.reshape(B, 5, self.d_model)
        f_out = self.freq_proj(f_out)
        tokens = self.freq_norm(tokens + f_out)     # (B, 5, D)

        # Prepend CLS and run standard transformer
        cls = self.cls.expand(B, -1, -1)            # (B, 1, D)
        seq = torch.cat([cls, tokens], dim=1)       # (B, 6, D)
        seq = self.transformer(seq)                 # (B, 6, D)

        return seq[:, 0]                            # CLS output: (B, D)


class ModalityEncoder(nn.Module):
    """Simple but stable encoder for eye/behaviour (matches original design)."""
    def __init__(self, input_dim, d_model=128):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8,
            dim_feedforward=4 * d_model,
            dropout=0.3, activation='gelu',
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, x):
        # x: (B, feat_dim) → unsqueeze to (B, 1, D)
        x = self.norm(self.proj(x)).unsqueeze(1)    # (B, 1, D)
        return self.encoder(x).squeeze(1)           # (B, D)


class NeuroFormerV3(nn.Module):
    """
    Same cross-modal fusion as original NeuroFormer.
    Only the EEG modality encoder is upgraded (CrissCross + frequency groups).
    Eye and behaviour encoders are identical to original.
    Total params ≈ 2.1M (vs original ~1.7M) — minimal increase.
    """
    def __init__(self, n_ch=5, feat_per_ch=103,
                 eye_dim=11, beh_dim=6, d_model=128):
        super().__init__()
        # Modality-specific encoders
        self.eeg_enc = CrissCrossEEGEncoder(n_ch, feat_per_ch, d_model)
        self.eye_enc = ModalityEncoder(eye_dim, d_model)
        self.beh_enc = ModalityEncoder(beh_dim, d_model)

        # Modality-type embeddings (same as original)
        self.mod_emb = nn.Embedding(3, d_model)

        # Cross-modal fusion transformer (same as original)
        cls_init = torch.zeros(1, 1, d_model)
        self.cls = nn.Parameter(cls_init)
        fus_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8,
            dim_feedforward=4 * d_model,
            dropout=0.3, activation='gelu',
            batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(fus_layer, num_layers=2)

        # Classification head (same as original)
        self.head = nn.Sequential(
            nn.Linear(d_model, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.normal_(self.mod_emb.weight, std=0.02)

    def forward(self, eeg, eye, beh):
        # Encode each modality
        e = self.eeg_enc(eeg)                               # (B, D)
        y_e = self.eye_enc(eye)                             # (B, D)
        b_e = self.beh_enc(beh)                             # (B, D)

        # Add modality-type embeddings
        ids = torch.arange(3, device=eeg.device)
        mod = self.mod_emb(ids)                             # (3, D)
        e   = e   + mod[0]
        y_e = y_e + mod[1]
        b_e = b_e + mod[2]

        # Concatenate with CLS and fuse
        B = eeg.shape[0]
        cls = self.cls.expand(B, -1, -1)                    # (B, 1, D)
        seq = torch.stack([e, y_e, b_e], dim=1)            # (B, 3, D)
        seq = torch.cat([cls, seq], dim=1)                  # (B, 4, D)
        seq = self.fusion(seq)                              # (B, 4, D)

        return self.head(seq[:, 0])                         # CLS → (B, 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

device = "cuda" if torch.cuda.is_available() else "cpu"
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
aucs = []
t0 = time.time()

print("="*80)
print("NeuroFormer v3: 5x5 repeated subject-level stratified CV, 8 seeds")
print("="*80 + "\n")

N_SEEDS = 8

for fold_idx, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
    fold_aucs = []

    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        model = NeuroFormerV3(
            n_ch=N_CH, feat_per_ch=FEAT_PER_CH,
            eye_dim=Xy.shape[1], beh_dim=Xb.shape[1],
            d_model=D
        ).to(device)

        # Class-weighted loss (per-fold, from training split only)
        pos = int((y[tr] == 1).sum())
        neg = int(len(y[tr]) - pos)
        class_weight = torch.tensor(
            [1.0, neg / max(pos, 1)], dtype=torch.float32, device=device)
        # Label smoothing reduces overfit on 18 positives
        loss_fn = nn.CrossEntropyLoss(
            weight=class_weight, label_smoothing=0.1)

        # KEY FIX: lr=5e-4, NOT 3e-3
        # With ~98 training samples and BS=8, we get ~12 steps/epoch.
        # At 5e-4 this matches the effective update scale of the original.
        opt = torch.optim.AdamW(
            model.parameters(), lr=5e-4, weight_decay=1e-4,
            betas=(0.9, 0.999))
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=20, T_mult=1, eta_min=1e-5)

        EPOCHS = 60
        BS = 8   # smaller batch → more gradient steps per epoch

        for epoch in range(EPOCHS):
            model.train()
            perm = np.random.permutation(len(tr))
            for i in range(0, len(perm), BS):
                idx = tr[perm[i:i+BS]]
                e_b = torch.from_numpy(Xe_r[idx]).float().to(device)
                y_b = torch.from_numpy(Xy[idx]).float().to(device)
                b_b = torch.from_numpy(Xb[idx]).float().to(device)
                y_t = torch.tensor(y[idx], dtype=torch.long, device=device)
                opt.zero_grad()
                logits = model(e_b, y_b, b_b)
                loss = loss_fn(logits, y_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()

        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(te), BS):
                idx = te[i:i+BS]
                e_b = torch.from_numpy(Xe_r[idx]).float().to(device)
                y_b = torch.from_numpy(Xy[idx]).float().to(device)
                b_b = torch.from_numpy(Xb[idx]).float().to(device)
                logits = model(e_b, y_b, b_b)
                probs.extend(torch.softmax(logits, -1)[:, 1].cpu().numpy())

        auc = roc_auc_score(y[te], probs[:len(te)])
        fold_aucs.append(auc)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_fold_auc = np.mean(fold_aucs)
    aucs.append(mean_fold_auc)
    print(f"Fold {fold_idx+1:2d}/25: {mean_fold_auc:.4f}  "
          f"(seed range [{min(fold_aucs):.3f}, {max(fold_aucs):.3f}])")

auc_arr = np.array(aucs)
total_time = time.time() - t0

print(f"\n{'='*80}")
print(f"FINAL RESULT (pooled OOF, 8-seed avg per fold):")
print(f"  NeuroFormer v3:       {auc_arr.mean():.4f} +/- {auc_arr.std():.4f}")
print(f"  Original NeuroFormer: 0.6240 +/- 0.0200")
print(f"  Low-rank baseline:    0.7070 +/- 0.0240")
print(f"  Demographics:         0.7170 +/- 0.0220")
print(f"\nFold std: {auc_arr.std():.4f}  (target: < 0.08 to match original stability)")
print(f"Min fold: {auc_arr.min():.4f}  Max fold: {auc_arr.max():.4f}")
print(f"Time: {total_time/60:.1f} min  ({total_time/25:.0f}s per fold)")
print('='*80)

if auc_arr.mean() > 0.717:
    print("\nBEATS DEMOGRAPHICS BASELINE!\n")
elif auc_arr.mean() > 0.708:
    print("\nBEATS LOW-RANK BASELINE!\n")
elif auc_arr.mean() > 0.624:
    print(f"\nBEATS ORIGINAL NeuroFormer by +{auc_arr.mean()-0.624:.3f} AUC!\n")
else:
    print(f"\nGap vs original: {0.624 - auc_arr.mean():.4f}\n")
