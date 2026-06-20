#!/usr/bin/env python
"""
The TNSRE main figure: ROC-AUC vs training-set size N, per adaptation mode
(frozen / full / LoRA), with 95% CIs, marking the crossover N* where fine-tuning
overtakes frozen. Reads results/crossover_<cohort>_<mode>_N<N>.csv.

    python experiments/make_crossover_figure.py            # -> figures/crossover.png
Run on the cluster (matplotlib available there).
"""
import csv
import glob
import re
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

PAT = re.compile(r"crossover_(labram|cbramod|eegpt)_(\w+?)_(frozen|full|lora)_N(\d+)\.csv$")
COL = {"frozen": "#2E6FB5", "full": "#B0303A", "lora": "#2E8B57"}
MARK = {"labram": "o", "cbramod": "s", "eegpt": "^"}     # marker per backbone, color per mode


def load(p):
    v = []
    for r in csv.reader(open(p)):
        if r and r[0] not in ("auc", ""):
            try:
                v.append(float(r[0]))
            except ValueError:
                pass
    return v


pts = {}                                  # (backbone, mode) -> list of (N, mean, ci)
for f in glob.glob("results/crossover_*_N*.csv"):
    m = PAT.search(Path(f).name)
    if not m:
        continue
    bb, _, mode, N = m.group(1), m.group(2), m.group(3), int(m.group(4))
    v = load(f)
    if not v:
        continue
    ci = 1.96 * st.pstdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0
    pts.setdefault((bb, mode), []).append((N, sum(v) / len(v), ci))

if not pts:
    raise SystemExit("no crossover_*_N*.csv yet — run the grid first")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for (bb, mode), series in pts.items():
    series.sort()
    xs = [s[0] for s in series]; ys = [s[1] for s in series]; es = [s[2] for s in series]
    ax.errorbar(xs, ys, yerr=es, marker=MARK.get(bb, "o"), color=COL.get(mode, "gray"),
                capsize=3, lw=1.8, label=f"{bb}:{mode}")
ax.axhline(0.5, ls=":", c="gray", lw=1, label="chance")
ax.set_xscale("log")
ax.set_xlabel("training-set size N (subjects)")
ax.set_ylabel("ROC-AUC")
ax.set_title("Adaptation x sample-size: when does fine-tuning beat frozen?")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
Path("figures").mkdir(exist_ok=True)
fig.tight_layout(); fig.savefig("figures/crossover.png", dpi=150, bbox_inches="tight")
print("[fig] wrote figures/crossover.png")
