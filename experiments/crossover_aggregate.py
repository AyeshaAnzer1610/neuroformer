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

PAT = re.compile(r"crossover_(\w+?)_(frozen|full|lora)_N(\d+)\.csv$")
want = sys.argv[1] if len(sys.argv) > 1 else None


def load(path):
    vals = []
    for r in csv.reader(open(path)):
        if r and r[0] not in ("auc", ""):
            try:
                vals.append(float(r[0]))
            except ValueError:
                pass
    return vals


rows = {}                                   # (cohort, N) -> {mode: (mean, ci, n)}
for f in glob.glob("results/crossover_*_N*.csv"):
    m = PAT.search(Path(f).name)
    if not m:
        continue
    cohort, mode, N = m.group(1), m.group(2), int(m.group(3))
    if want and cohort != want:
        continue
    v = load(f)
    if not v:
        continue
    ci = 1.96 * st.pstdev(v) / len(v) ** 0.5 if len(v) > 1 else float("nan")
    rows.setdefault((cohort, N), {})[mode] = (sum(v) / len(v), ci, len(v))

if not rows:
    sys.exit("no crossover_*_N*.csv found yet — run scripts/crossover_sbatch.sh cells first")

out = [["cohort", "N", "frozen", "full", "lora", "crossover(full>=frozen?)"]]
print(f"\n{'cohort':<8}{'N':>6}{'frozen':>12}{'full':>12}{'lora':>12}   note")
for (cohort, N) in sorted(rows, key=lambda k: (k[0], k[1])):
    d = rows[(cohort, N)]
    def cell(mode):
        return f"{d[mode][0]:.3f}" if mode in d else "  -  "
    fr = d.get("frozen", (None,))[0]; fu = d.get("full", (None,))[0]
    note = ""
    if fr is not None and fu is not None:
        note = "FT>=frozen" if fu >= fr else "frozen>FT"
    print(f"{cohort:<8}{N:>6}{cell('frozen'):>12}{cell('full'):>12}{cell('lora'):>12}   {note}")
    out.append([cohort, N, cell("frozen"), cell("full"), cell("lora"), note])

Path("results").mkdir(exist_ok=True)
with open("results/crossover_summary.csv", "w", newline="") as fh:
    csv.writer(fh).writerows(out)
print("\n[done] wrote results/crossover_summary.csv")
print("Crossover N* = smallest N where 'FT>=frozen' (full fine-tuning overtakes frozen).")
