#!/usr/bin/env python
"""
Validate a pretrained OcuFM on a downstream GAZE task (this is what turns OcuFM from a
"proposed" model into a *validated* foundation model).

GazeBase filenames encode subject + task: S_<rrr>_<subj>_S<sess>_<TASK> (TASK in e.g.
RAN/TEX/HSS/FXS/VD1/VD2/BLG). We:
  - parse windows + (subject, task) labels (pure pandas, no pymovements);
  - extract FROZEN OcuFM embeddings (recording = mean over windows);
  - linear-probe for TASK classification (7-class) under SUBJECT-disjoint CV;
  - compare against a hand-crafted gaze-statistics baseline.

If OcuFM's frozen embedding beats the hand-crafted baseline, the gaze FM carries real,
transferable structure — OcuFM's genuine (gaze-benchmark) positive result.

    python src/ocufm_eval.py --path $PROJECT/mh/data/gazebase \
        --ckpt $PROJECT/mh/ckpt/ocufm_gazebase.pt
"""
import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from ocufm import OcuFM
from ocufm_pretrain import _derive_channels

FN = re.compile(r"S_(\d+)_S\d+_([A-Za-z0-9]+)")          # S_<subject>_S<sess>_<TASK> -> (subject, task)


def load_labeled(path, sfreq, window_sec, max_win_per_rec=20, max_rec=None):
    files = sorted(glob.glob(str(Path(path) / "**" / "*.csv"), recursive=True))
    if max_rec:
        files = files[:max_rec]
    win = int(window_sec * sfreq); step = max(1, int(round(1000.0 / sfreq)))
    X, subj, task = [], [], []
    rng = np.random.default_rng(0)
    for f in files:
        m = FN.search(Path(f).stem)
        if not m:
            continue
        s, t = m.group(1), m.group(2)
        try:
            df = pd.read_csv(f)
        except Exception:  # noqa: BLE001
            continue
        cols = {c.lower().strip(): c for c in df.columns}
        def c(*n):
            for k in n:
                if k in cols:
                    return df[cols[k]].to_numpy(float)
            return None
        x, y, pupil = c("x", "xpos"), c("y", "ypos"), c("dp", "pupil", "pa")
        if x is None or y is None:
            continue
        ch = _derive_channels(x[::step], y[::step], pupil[::step] if pupil is not None else None, sfreq)
        T = ch.shape[1]; nwin = T // win
        if nwin == 0:
            continue
        w = ch[:, :nwin * win].reshape(5, nwin, win).transpose(1, 0, 2)
        if max_win_per_rec and len(w) > max_win_per_rec:
            w = w[rng.choice(len(w), max_win_per_rec, replace=False)]
        X.append(w.astype(np.float32)); subj += [s] * len(w); task += [t] * len(w)
    if not X:
        sys.exit("no labeled GazeBase windows — check path/filenames")
    return np.concatenate(X, 0), np.array(subj), np.array(task)


def ocufm_embed(ckpt, Xwin, device):
    """Frozen OcuFM embedding per window: mean over patches+channels -> [N, d]."""
    model = OcuFM(n_channels=Xwin.shape[1], patch_len=Xwin.shape[2] // 4 or 1, d_model=64)
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd, strict=False); model.eval().to(device)
    embs = []
    with torch.no_grad():
        for i in range(0, len(Xwin), 64):
            xb = torch.tensor(Xwin[i:i + 64].tolist(), dtype=torch.float32, device=device)
            tok = model.embed_windows(xb)                 # [B, n, C, d]
            embs.append(tok.mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(embs, 0)


def handcrafted(Xwin):
    """Baseline: per-window channel mean/std/range -> [N, 15]."""
    return np.concatenate([Xwin.mean(-1), Xwin.std(-1), Xwin.ptp(-1)], axis=1)


def probe(feat, y, groups, name):
    feat = feat.reshape(len(feat), -1)
    gkf = GroupKFold(n_splits=5)
    accs = []
    for tr, te in gkf.split(feat, y, groups):
        sc = StandardScaler().fit(feat[tr])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        clf.fit(sc.transform(feat[tr]), y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(sc.transform(feat[te]))))
    print(f"  {name:<22} task-classification balanced-acc = {np.mean(accs):.3f} "
          f"+/- {1.96*np.std(accs)/len(accs)**0.5:.3f}")
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sfreq", type=float, default=250.0); ap.add_argument("--window-sec", type=float, default=4.0)
    ap.add_argument("--max-rec", type=int, default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, subj, task = load_labeled(args.path, args.sfreq, args.window_sec, max_rec=args.max_rec)
    print(f"[eval] {len(X)} windows | {len(set(subj))} subjects | tasks={sorted(set(task))}")
    print("\n=== Downstream gaze task classification (subject-disjoint 5-fold) ===")
    base = probe(handcrafted(X), task, subj, "hand-crafted gaze")
    emb = probe(ocufm_embed(args.ckpt, X, device), task, subj, "OcuFM (frozen)")
    verdict = "OcuFM > hand-crafted (FM learns useful gaze structure)" if emb > base \
        else "OcuFM <= hand-crafted (no benefit -> report honestly)"
    print(f"\n[verdict] {verdict}")


if __name__ == "__main__":
    main()
