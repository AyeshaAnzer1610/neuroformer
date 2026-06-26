#!/usr/bin/env python
"""Investigate whether NeuroFormer's underperformance (pooled-OOF 0.62, train-test gap
9-25pp = overfitting) is a CAPACITY / REGULARIZATION artifact.

We test a SMALL, PRE-DECLARED set of principled configs (lower capacity + stronger
regularization are the textbook fixes for overfitting at N=102) and report the FULL sweep
-- every config, mean +/- std over seeds -- NOT a cherry-picked winner. Whatever wins is
still compared honestly to the age+sex demographic predictor (pooled-OOF 0.717) and to
Low-rank Fusion (0.708). This run cannot, by construction, manufacture a positive claim;
it can only tell us the honest best-case for the architecture.

    python experiments/nf_regularize.py --seeds 4 --out results/nf_regularize.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robustness_3mod import load_eeg, load_eye, load_beh                      # noqa: E402
from neuroformer_baselines import neuroformer_cv, pooled_summ                 # noqa: E402

# Pre-declared configs (name, overrides). d_model must be divisible by n_heads.
CONFIGS = [
    ("baseline d128 L2/2 do0.3 wd1e-4", {}),
    ("reg do0.5 wd1e-3", {"dropout": 0.5, "weight_decay": 1e-3}),
    ("smaller d64 L1/1", {"d_model": 64, "n_encoder_layers": 1, "n_fusion_layers": 1, "n_heads": 8}),
    ("small+reg d64 L1/1 do0.5 wd1e-3", {"d_model": 64, "n_encoder_layers": 1, "n_fusion_layers": 1,
                                         "n_heads": 8, "dropout": 0.5, "weight_decay": 1e-3}),
    ("tiny d32 L1/1 do0.5 wd1e-2", {"d_model": 32, "n_encoder_layers": 1, "n_fusion_layers": 1,
                                    "n_heads": 4, "dropout": 0.5, "weight_decay": 1e-2}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--nf-repeats", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=4); ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default="results/nf_regularize.csv")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed0); torch.manual_seed(args.seed0)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    eeg, eye, beh = load_eeg(), load_eye(), load_beh()
    lab = pd.read_csv("data/labels.csv").set_index("ID")["DX_Status"].to_dict()
    subs = sorted(set(lab) & set(eeg) & set(eye) & set(beh))
    y = np.array([int(lab[s] > 0) for s in subs])
    print(f"[load] N={len(subs)} pos={int(y.sum())} | seeds={args.seeds} | configs={len(CONFIGS)}", flush=True)
    print("REFERENCE (pooled-OOF): age+sex 0.717 | Low-rank Fusion 0.708 | GMU 0.669\n", flush=True)

    rows = []
    for name, cfg in CONFIGS:
        pooled, gaps = [], []
        for sd in range(args.seed0, args.seed0 + args.seeds):
            arr, gap, oof_d, omat = neuroformer_cv(subs, lab, eeg, eye, beh, {"eeg", "eye", "beh"},
                                                   args.folds, args.nf_repeats, sd, device, aux=False, cfg=cfg)
            m, _ = pooled_summ(omat, y)
            pooled.append(m); gaps.append(gap)
            print(f"  [{name}] seed {sd}: pooled-OOF {m:.3f}  (gap {gap:.0f}pp)", flush=True)
        pooled = np.array(pooled)
        beats = "beats age+sex" if (pooled.mean() - pooled.std(ddof=1)) > 0.717 else "does NOT beat age+sex (0.717)"
        rows.append({"config": name, "pooled_auc_mean": float(pooled.mean()),
                     "pooled_auc_std": float(pooled.std(ddof=1)) if len(pooled) > 1 else float("nan"),
                     "gap_pp_mean": float(np.mean(gaps)), "n_seeds": args.seeds, "verdict": beats})
        print(f"  >>> {name}: {pooled.mean():.3f} +/- {pooled.std(ddof=1):.3f}  [{beats}]\n", flush=True)

    df = pd.DataFrame(rows).sort_values("pooled_auc_mean", ascending=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False), flush=True)
    best = df.iloc[0]
    print(f"\n[honest verdict] best config = {best['config']} @ {best['pooled_auc_mean']:.3f} "
          f"+/- {best['pooled_auc_std']:.3f} -> {best['verdict']}", flush=True)
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
