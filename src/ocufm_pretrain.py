#!/usr/bin/env python
"""
Pretrain OcuFM (gaze foundation model) on GazeBase — the real pretraining run.

Uses pymovements to download/load GazeBase (EyeLink 1000 @ 1000 Hz, ~322 subjects),
builds gaze channel windows (x, y, pupil, velocity, acceleration), and runs OcuFM's
masked-patch self-supervised objective (from src/ocufm.py). Saves the pretrained
backbone for downstream gaze tasks + the EEG+gaze fusion framework.

Honest scope: this makes OcuFM a *pretrained, evaluable* gaze FM (vs GazeMTM/GLASS,
which use different recipes). It is NOT expected to help MIPDB screening (eye is at
chance there) — its validation is on gaze benchmarks.

Smoke (few recordings, one step) to validate the pymovements/GazeBase path on Nibi:
    python src/ocufm_pretrain.py --path $PROJECT/mh/data/gazebase --download --smoke
Full:
    python src/ocufm_pretrain.py --path $PROJECT/mh/data/gazebase \
        --epochs 30 --out $PROJECT/mh/ckpt/ocufm_gazebase.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from ocufm import OcuFM, GAZE_CHANNELS


def _derive_channels(x, y, pupil, sfreq):
    """[T] x,y(,pupil) -> [5, T] = x, y, pupil, speed, accel (NaN-filled, z-scored)."""
    def z(a):
        a = np.nan_to_num(a, nan=0.0).astype(np.float32)
        s = a.std() or 1.0
        return (a - a.mean()) / s
    vx = np.gradient(np.nan_to_num(x)); vy = np.gradient(np.nan_to_num(y))
    speed = np.sqrt(vx ** 2 + vy ** 2)
    accel = np.gradient(speed)
    pupil = pupil if pupil is not None else np.zeros_like(x)
    return np.stack([z(x), z(y), z(pupil), z(speed), z(accel)], 0)   # [5, T]


def load_gazebase_windows(path, download, sfreq=250.0, window_sec=4.0,
                          max_rec=None, max_win_per_rec=20):
    """Load GazeBase via pymovements -> windows [N, 5, T]. Cluster-only (needs pymovements
    + the downloaded corpus). The exact column names can vary by pymovements version; we
    look them up defensively and print what we find on first use."""
    import pymovements as pm
    ds = pm.Dataset("GazeBase", path=path)
    if download:
        ds.download()
    ds.load()
    win = int(window_sec * sfreq)
    out, printed = [], False
    recs = ds.gaze[:max_rec] if max_rec else ds.gaze
    for g in recs:
        df = g.frame                                    # polars DataFrame
        cols = {c.lower(): c for c in df.columns}
        if not printed:
            print(f"[gazebase] columns: {list(df.columns)}", flush=True); printed = True
        def col(*names):
            for n in names:
                if n in cols:
                    return np.asarray(df[cols[n]].to_numpy(), float)
            return None
        x = col("x", "x_left", "xpos"); y = col("y", "y_left", "ypos")
        pupil = col("pupil", "pupil_left", "pupil_area", "pa")
        if x is None or y is None:
            continue
        # resample 1000 Hz -> sfreq by simple decimation (GazeBase is 1000 Hz)
        step = max(1, int(round(1000.0 / sfreq)))
        x, y = x[::step], y[::step]
        pupil = pupil[::step] if pupil is not None else None
        ch = _derive_channels(x, y, pupil, sfreq)        # [5, T']
        T = ch.shape[1]; nwin = T // win
        if nwin == 0:
            continue
        w = ch[:, :nwin * win].reshape(5, nwin, win).transpose(1, 0, 2)  # [nwin,5,win]
        if max_win_per_rec and len(w) > max_win_per_rec:
            w = w[np.random.default_rng(0).choice(len(w), max_win_per_rec, replace=False)]
        out.append(w.astype(np.float32))
    if not out:
        sys.exit("no GazeBase windows loaded — check pymovements columns/path")
    X = np.concatenate(out, 0)
    print(f"[gazebase] {len(X)} windows, shape {X.shape}", flush=True)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="GazeBase dir (project space; SCRATCH is full)")
    ap.add_argument("--download", action="store_true", help="download via pymovements first")
    ap.add_argument("--sfreq", type=float, default=250.0); ap.add_argument("--window-sec", type=float, default=4.0)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--max-rec", type=int, default=None)
    ap.add_argument("--max-win-per-rec", type=int, default=20)
    ap.add_argument("--no-amp", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="ckpt/ocufm_gazebase.pt")
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] torch {torch.__version__}  cuda={torch.cuda.is_available()}", flush=True)

    X = load_gazebase_windows(args.path, args.download, args.sfreq, args.window_sec,
                              max_rec=(4 if args.smoke else args.max_rec),
                              max_win_per_rec=args.max_win_per_rec)
    patch_len = int(args.sfreq)                          # 1 s patches
    model = OcuFM(n_channels=X.shape[1], patch_len=patch_len, d_model=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = (device == "cuda") and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def step(xb):
        opt.zero_grad()
        with torch.autocast(device_type="cuda", enabled=use_amp):
            out = model(xb)
        scaler.scale(out["loss"]).backward(); scaler.step(opt); scaler.update()
        return out

    if args.smoke:
        xb = torch.from_numpy(X[: min(args.batch, len(X))]).to(device)
        model.train(); out = step(xb)
        mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
        print(f"[smoke] OK  loss={out['loss'].item():.3f} (code {out['loss_code']:.3f}, "
              f"recon {out['loss_recon']:.3f}, vq {out['vq_loss']:.3f})  peak {mem:.2f} GB")
        print("[smoke] env good — safe to launch full OcuFM pretraining."); return

    idx = np.arange(len(X)); model.train()
    for ep in range(args.epochs):
        np.random.shuffle(idx); tot = nb = 0
        for i in range(0, len(idx), args.batch):
            b = idx[i:i + args.batch]
            if len(b) < 4:
                continue
            tot += step(torch.from_numpy(X[b]).to(device))["loss"].item(); nb += 1
        print(f"[ocufm] epoch {ep+1}/{args.epochs}  loss={tot/max(nb,1):.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"[done] pretrained OcuFM -> {args.out}  ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)")


if __name__ == "__main__":
    main()
