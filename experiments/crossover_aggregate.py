#!/usr/bin/env python
"""
Assemble the adaptation x sample-size crossover from per-cell CSVs
(results/crossover_<cohort>_<mode>_N<N>.csv, each a column of fold AUCs).

    python experiments/crossover_aggregate.py            # all cohorts
    python experiments/crossover_aggregate.py hbn        # one cohort

Prints a table (rows = N, cols = frozen/full/lora) of mean AUC and flags the
crossover N* where fine-tuning first matches/beats frozen. Writes
results/crossover_summary.csv. No torch/sklearn — stdlib only.
"""
import csv
import glob
import math
import re
import statistics as st
import sys
from pathlib import Path

try:
    from scipy import stats as _ss
except Exception:  # noqa: BLE001
    _ss = None


def _p_twosided(t, df):
    if _ss is not None:
        return float(2 * _ss.t.sf(abs(t), df))
    return float(math.erfc(abs(t) / math.sqrt(2)))        # normal approx if no scipy


def sig_full_vs_frozen(fr, fu):
    """p-value, full vs frozen fold AUCs. Paired t if same n (same CV splits), else Welch."""
    if not fr or not fu:
        return None, None
    a, b = list(fu), list(fr)                              # full - frozen
    if len(a) == len(b):                                   # paired
        d = [x - y for x, y in zip(a, b)]
        n = len(d); mean = sum(d) / n
        sd = (sum((x - mean) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else 0.0
        if sd == 0:
            return mean, 0.0
        return mean, _p_twosided(mean / (sd / n ** 0.5), n - 1)
    ma, mb = sum(a) / len(a), sum(b) / len(b)              # Welch
    va = st.pvariance(a) * len(a) / (len(a) - 1) if len(a) > 1 else 0.0
    vb = st.pvariance(b) * len(b) / (len(b) - 1) if len(b) > 1 else 0.0
    se = (va / len(a) + vb / len(b)) ** 0.5
    if se == 0:
        return ma - mb, 0.0
    df = (va / len(a) + vb / len(b)) ** 2 / ((va / len(a)) ** 2 / max(len(a) - 1, 1)
                                             + (vb / len(b)) ** 2 / max(len(b) - 1, 1))
    return ma - mb, _p_twosided((ma - mb) / se, df)

# Match BOTH namings: new crossover_<backbone>_<cohort>_<mode>_N<N> and the older
# crossover_<cohort>_<mode>_N<N> (LaBraM-only, before the --backbone update).
PAT = re.compile(r"crossover_(?:(labram|cbramod|eegpt)_)?(\w+?)_(frozen|full|lora)_N(\d+)\.csv$")
want = sys.argv[1] if len(sys.argv) > 1 else None       # optional cohort filter


def load(path):
    vals = []
    for r in csv.reader(open(path)):
        if r and r[0] not in ("auc", ""):
            try:
                vals.append(float(r[0]))
            except ValueError:
                pass
    return vals


rows = {}                                   # (backbone, cohort, N) -> {mode: (mean, ci, n)}
arr = {}                                    # (backbone, cohort, N) -> {mode: [fold AUCs]}
for f in glob.glob("results/crossover_*_N*.csv"):
    m = PAT.search(Path(f).name)
    if not m:
        continue
    bb = m.group(1) or "labram"                          # old name => LaBraM
    cohort, mode, N = m.group(2), m.group(3), int(m.group(4))
    if want and cohort != want:
        continue
    v = load(f)
    if not v:
        continue
    ci = 1.96 * st.pstdev(v) / len(v) ** 0.5 if len(v) > 1 else float("nan")
    rows.setdefault((bb, cohort, N), {})[mode] = (sum(v) / len(v), ci, len(v))
    arr.setdefault((bb, cohort, N), {})[mode] = v

if not rows:
    sys.exit("no crossover_*_N*.csv found yet — run scripts/crossover_sbatch.sh cells first")

out = [["backbone", "cohort", "N", "frozen", "full", "lora", "crossover(full>=frozen?)"]]
print(f"\n{'backbone':<9}{'cohort':<7}{'N':>6}{'frozen':>11}{'full':>11}{'lora':>11}   note")
for (bb, cohort, N) in sorted(rows, key=lambda k: (k[0], k[1], k[2])):
    d = rows[(bb, cohort, N)]
    def cell(mode):
        return f"{d[mode][0]:.3f}" if mode in d else "  -  "
    fr = d.get("frozen", (None,))[0]; fu = d.get("full", (None,))[0]
    note = ""
    if fr is not None and fu is not None:
        note = "FT>=frozen" if fu >= fr else "frozen>FT"
    print(f"{bb:<9}{cohort:<7}{N:>6}{cell('frozen'):>11}{cell('full'):>11}{cell('lora'):>11}   {note}")
    out.append([bb, cohort, N, cell("frozen"), cell("full"), cell("lora"), note])

Path("results").mkdir(exist_ok=True)
with open("results/crossover_summary.csv", "w", newline="") as fh:
    csv.writer(fh).writerows(out)

# Per-N significance: full vs frozen at each (backbone, cohort, N).
print(f"\n=== full vs frozen significance per N {'(scipy)' if _ss else '(normal-approx p)'} ===")
print(f"{'backbone':<9}{'cohort':<7}{'N':>6}{'delta(full-frozen)':>20}{'p':>9}   ")
for (bb, cohort, N) in sorted(arr, key=lambda k: (k[0], k[1], k[2])):
    d = arr[(bb, cohort, N)]
    if "frozen" in d and "full" in d:
        delta, p = sig_full_vs_frozen(d["frozen"], d["full"])
        flag = "FT>frozen*" if (delta and delta > 0 and p is not None and p < 0.05) else \
               ("FT>frozen" if (delta and delta > 0) else "frozen>=FT")
        print(f"{bb:<9}{cohort:<7}{N:>6}{delta:>+20.3f}{p:>9.4f}   {flag}")
print("\n[done] wrote results/crossover_summary.csv")
print("Crossover N* = smallest N where full-FT is significantly > frozen (p<0.05, '*').")
