#!/usr/bin/env python
"""
TAPT — task-/domain-adaptive continued pretraining of LaBraM on UNLABELED clinical EEG.

Motivated by our freeze-don't-fine-tune finding: supervised fine-tuning overfits the
~23 labels, so adapt the *representation* with SELF-SUPERVISION on unlabeled windows
(no labels -> no label overfitting), then freeze + read out. Goal: move the frozen
~0.74 ceiling and/or improve cross-cohort transfer by adapting LaBraM to the
clinical-EEG domain.

Method (low-risk, backbone treated as a black box via return_features): SimCLR-style
contrastive continued pretraining. Two augmented views of each window -> LaBraM
embedding -> projection -> NT-Xent. Backbone + projector train; the adapted backbone
state_dict is saved and later loaded frozen for the downstream CV
(finetune_labram.py --freeze-backbone --backbone-weights).

Stage 1 (this run): MIPDB only (validate the loop end-to-end).
Stage 2 (next): add HBN windows (--extra-dirs) where the real adaptation signal is.

FIRST validate fast on the cluster:
    python src/tapt_pretrain.py --input $SCRATCH/mh/data/mipdb --max-subj 4 --smoke

Then pretrain + save adapted weights:
    python src/tapt_pretrain.py --input $SCRATCH/mh/data/mipdb --all-recordings \
        --epochs 50 --out $SCRATCH/mh/ckpt/labram_tapt_mipdb.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from extract_eeg_embeddings import load_and_window


def collect_windows(input_dir, match, all_recordings, max_subj, max_per_subj):
    """All QC-passing windows across subjects (labels NOT needed). -> [N, C, T], chs."""
    subs = sorted(p for p in Path(input_dir).iterdir() if p.is_dir())
    if max_subj:
        subs = subs[:max_subj]
    wins, chs = [], None
    rng = np.random.default_rng(0)
    for p in subs:
        try:
            raw, w = load_and_window(p, match=match, all_recordings=all_recordings)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {p.name}: {e}"); continue
        if chs is None:
            chs = raw.info["chs"]
        if max_per_subj and len(w) > max_per_subj:
            w = w[rng.choice(len(w), max_per_subj, replace=False)]
        wins.append(w.astype(np.float32))
        print(f"[load] {p.name}: {len(w)} windows", flush=True)
    if not wins:
        sys.exit("no windows collected")
    X = np.concatenate(wins, 0)
    print(f"[data] {len(X)} unlabeled windows, shape {X.shape}")
    return X, chs


def augment(x):
    """Two stochastic views of a window batch [B, C, T] (amplitude scale, Gaussian
    noise, channel dropout, time jitter/roll). Returns a single augmented copy."""
    B, C, T = x.shape
    x = x * (0.8 + 0.4 * torch.rand(B, 1, 1, device=x.device))      # amplitude scale
    x = x + 0.1 * x.std() * torch.randn_like(x)                     # additive noise
    drop = (torch.rand(B, C, 1, device=x.device) > 0.1).float()     # 10% channel dropout
    x = x * drop
    shift = int(0.05 * T)
    if shift:
        x = torch.roll(x, shifts=int(torch.randint(-shift, shift + 1, (1,)).item()), dims=-1)
    return x


def nt_xent(z1, z2, temp=0.2):
    """SimCLR NT-Xent over a batch; positives are (i, i+B)."""
    z = F.normalize(torch.cat([z1, z2], 0), dim=1)                  # [2B, d]
    n = z.shape[0]
    sim = z @ z.t() / temp
    sim.fill_diagonal_(float("-inf"))
    targets = torch.arange(n, device=z.device)
    targets = (targets + n // 2) % n                               # i <-> i+B
    return F.cross_entropy(sim, targets)


class Projector(nn.Module):
    def __init__(self, d, d_out=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d_out))

    def forward(self, x):
        return self.net(x)


def embed(model, x, use_amp):
    """LaBraM window embedding (black-box: cls_token or mean-pooled features)."""
    with torch.autocast(device_type="cuda", enabled=use_amp):
        out = model(x, return_features=True)
    feats = out.get("cls_token")
    if feats is None:
        feats = out["features"]
        if feats.ndim == 3:
            feats = feats.mean(1)
    return feats


def build_backbone(chs, n_times, device):
    from braindecode.models import InterpolatedLaBraM
    m = InterpolatedLaBraM.from_pretrained(
        "braindecode/labram-pretrained", chs_info=chs, n_times=n_times,
        n_outputs=2, patch_size=200, strict=False)
    return m.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--extra-dirs", nargs="*", default=[],
                    help="extra subject-parent dirs to add (e.g. HBN) for Stage 2")
    ap.add_argument("--match", default=None); ap.add_argument("--all-recordings", action="store_true")
    ap.add_argument("--epochs", type=int, default=50); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--max-subj", type=int, default=None)
    ap.add_argument("--max-per-subj", type=int, default=40, help="cap windows/subject (memory)")
    ap.add_argument("--no-amp", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="build + one step, validate env, exit")
    ap.add_argument("--out", default="ckpt/labram_tapt.pt")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] torch {torch.__version__}  cuda={torch.cuda.is_available()}", flush=True)

    X, chs = collect_windows(args.input, args.match, args.all_recordings,
                             args.max_subj, args.max_per_subj)
    for d in args.extra_dirs:
        Xe, _ = collect_windows(d, args.match, args.all_recordings, args.max_subj, args.max_per_subj)
        X = np.concatenate([X, Xe], 0)
    n_times = X.shape[-1]

    model = build_backbone(chs, n_times, device)
    proj = Projector(model.num_features if hasattr(model, "num_features") else embed(
        model, torch.zeros(1, X.shape[1], n_times, device=device), False).shape[-1]).to(device)
    params = list(model.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    use_amp = (device == "cuda") and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def step(xb):
        opt.zero_grad()
        z1 = proj(embed(model, augment(xb), use_amp))
        z2 = proj(embed(model, augment(xb), use_amp))
        loss = nt_xent(z1, z2, args.temp)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        return loss.item()

    if args.smoke:
        xb = torch.from_numpy(X[: min(args.batch, len(X))]).to(device)
        model.train(); proj.train()
        l = step(xb)
        mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
        print(f"[smoke] OK  contrastive loss={l:.3f}  peak GPU mem={mem:.2f} GB  amp={use_amp}")
        print("[smoke] env good — safe to launch full TAPT.")
        return

    model.train(); proj.train()
    idx = np.arange(len(X))
    for ep in range(args.epochs):
        np.random.shuffle(idx); tot = 0.0; nb = 0
        for i in range(0, len(idx), args.batch):
            b = idx[i:i + args.batch]
            if len(b) < 4:
                continue
            tot += step(torch.from_numpy(X[b]).to(device)); nb += 1
        print(f"[tapt] epoch {ep+1}/{args.epochs}  loss={tot/max(nb,1):.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"[done] saved TAPT-adapted backbone -> {args.out}  "
          f"(eval: finetune_labram.py --freeze-backbone --backbone-weights {args.out})")


if __name__ == "__main__":
    main()
