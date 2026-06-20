#!/usr/bin/env python
"""
Statistical significance for the adaptation comparison (for the paper's rigor).

Frozen vs each adaptation, from the committed per-fold AUC CSVs. Where two runs share
the SAME repeated-stratified-CV seed/splits (e.g. *_5x5 frozen vs lora), the comparison
is PAIRED (paired t-test + Wilcoxon signed-rank + paired Cohen's d); otherwise unpaired
Welch t-test. Pure numpy + optional scipy (stdlib fallback for the t-test p-value).

    python experiments/stats_tests.py
"""
import csv
import math
from pathlib import Path

import numpy as np

try:
    from scipy import stats as ss
except Exception:  # noqa: BLE001
    ss = None


def load(name):
    p = Path("results") / name
    if not p.exists():
        return None
    out = []
    for r in csv.reader(open(p)):
        if r and r[0] not in ("auc", ""):
            try:
                out.append(float(r[0]))
            except ValueError:
                pass
    return np.array(out) if out else None


def _t_sf(t, df):
    """Two-sided p for Student-t; scipy if available, else a stable approximation."""
    if ss is not None:
        return 2 * ss.t.sf(abs(t), df)
    # Normal approx (df large) — adequate for reporting when scipy absent.
    return 2 * 0.5 * math.erfc(abs(t) / math.sqrt(2))


def paired(a, b, na, nb):
    d = a - b
    n = len(d); mean = d.mean(); sd = d.std(ddof=1)
    t = mean / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    p = _t_sf(t, n - 1)
    dz = mean / sd if sd > 0 else float("inf")            # paired Cohen's dz
    w = ss.wilcoxon(a, b).pvalue if (ss is not None and n >= 6) else float("nan")
    print(f"  {na} vs {nb}  PAIRED (n={n}): delta={mean:+.3f}  paired-t p={p:.4f}  "
          f"Wilcoxon p={w:.4f}  dz={dz:+.2f}")


def welch(a, b, na, nb):
    ma, mb = a.mean(), b.mean()
    t = (ma - mb) / math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    df = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2 / (
        (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    p = _t_sf(t, df)
    ps = a.std(ddof=1)
    d = (ma - mb) / ps if ps > 0 else float("inf")
    print(f"  {na} vs {nb}  Welch (n={len(a)},{len(b)}): delta={ma-mb:+.3f}  p={p:.4f}  d~{d:+.2f}")


print(f"[stats] scipy {'available' if ss else 'ABSENT (normal-approx p; Wilcoxon skipped)'}")
fr5 = load("finetune_labram_frozen_5x5.csv"); lo5 = load("finetune_labram_lora_5x5.csv")
fr2 = load("finetune_labram_frozen_r2.csv");  lo2 = load("finetune_labram_lora.csv")
full = load("finetune_labram_full.csv");      tapt = load("finetune_labram_tapt_5x2.csv")

print("\n=== Frozen vs adaptation (does frozen significantly win?) ===")
if fr5 is not None and lo5 is not None and len(fr5) == len(lo5):
    paired(fr5, lo5, "frozen_5x5", "LoRA_5x5")          # same CV splits -> paired
if fr2 is not None and tapt is not None and len(fr2) == len(tapt):
    paired(fr2, tapt, "frozen_5x2", "TAPT_5x2")         # same 5x2 splits -> paired
if fr5 is not None and full is not None:
    welch(fr5, full, "frozen_5x5", "full_FT")           # unmatched n -> Welch
print("\n(delta=>0 => frozen higher; p<0.05 => significant. Use for the 'freeze' claim's rigor.)")
