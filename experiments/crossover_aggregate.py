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
import re
import statistics as st
import sys
from pathlib import Path

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
print("\n[done] wrote results/crossover_summary.csv")
print("Crossover N* = smallest N where 'FT>=frozen' (full fine-tuning overtakes frozen).")
