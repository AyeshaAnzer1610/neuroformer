#!/usr/bin/env python
"""
Efficiency table: trainable parameters per backbone x adaptation mode (the
'frozen is also the cheapest' argument). Parses the [backbone]/[freeze]/[lora]
diagnostic lines that finetune_labram.py prints into the SLURM logs.

    python experiments/efficiency_table.py            # scans mh-*.out
Run on the cluster (where the logs live).
"""
import glob
import re
import sys

logs = sys.argv[1:] or glob.glob("mh-*.out")
BACK = re.compile(r"\[backbone\]\s+(\w+)\s+n_chans=(\d+)\s+n_times=(\d+)\s+total_params=([\d,]+)")
LORA = re.compile(r"\[lora\].*trainable\s+([\d,]+)/([\d,]+)\s+=\s+([\d.]+)%")
FRZ = re.compile(r"\[freeze\]\s+trainable\s+([\d,]+)/([\d,]+)\s+=\s+([\d.]+)%")


def n(s):
    return int(s.replace(",", ""))


seen = {}                         # (backbone, mode) -> (trainable, total, pct)
for lg in logs:
    try:
        txt = open(lg, errors="ignore").read()
    except OSError:
        continue
    bbm = BACK.search(txt)
    backbone = bbm.group(1) if bbm else "?"
    total_bb = n(bbm.group(4)) if bbm else None
    for mode, pat in (("LoRA", LORA), ("frozen", FRZ)):
        m = pat.search(txt)
        if m:
            seen[(backbone, mode)] = (n(m.group(1)), n(m.group(2)), float(m.group(3)))
    # full FT = everything trainable (no freeze/lora line); record from backbone total
    if bbm and not LORA.search(txt) and not FRZ.search(txt):
        seen[(backbone, "full")] = (total_bb, total_bb, 100.0)

if not seen:
    sys.exit("no [backbone]/[freeze]/[lora] lines found in logs — point at mh-*.out files")

print(f"\n{'backbone':<10}{'mode':<8}{'trainable':>14}{'total':>14}{'% trainable':>13}")
for (bb, mode) in sorted(seen):
    tr, tot, pct = seen[(bb, mode)]
    print(f"{bb:<10}{mode:<8}{tr:>14,}{tot:>14,}{pct:>12.2f}%")
print("\n(Use for the efficiency claim: frozen trains a tiny head; LoRA ~few %; full = 100%.)")
