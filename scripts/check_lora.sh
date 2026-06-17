#!/usr/bin/env bash
# One-shot status for the LoRA fine-tune run. Run on a Nibi LOGIN node:
#   bash scripts/check_lora.sh
# Auto-finds the latest mh-lora log (no job-id juggling) and shows the key lines.
set -uo pipefail
cd ~/mental-health 2>/dev/null || true

echo "=== your queue ==="
squeue -u "$USER" 2>/dev/null || true

LOG=$(ls -t mh-lora-*.out 2>/dev/null | head -1)
if [ -n "${LOG:-}" ]; then
    echo; echo "=== latest log: $LOG ==="
    echo "--- env / LoRA injection / smoke (must show 'wrapped N>0' + '[smoke] OK') ---"
    grep -E "\[env\]|\[lora\]|\[smoke\]" "$LOG" | head -25 || true
    echo "--- folds / AUC ---"
    grep -E "fold|AUC|ROC|finetune|done" "$LOG" | tail -25 || true
    echo "--- last 15 lines (errors surface here) ---"
    tail -15 "$LOG"
else
    echo; echo "(no mh-lora-*.out yet — job not started or not submitted)"
fi

echo; echo "=== result CSVs (present once folds finish) ==="
for f in results/finetune_labram_lora.csv results/finetune_labram_frozen_r2.csv results/finetune_labram_full.csv; do
    if [ -f "$f" ]; then echo "# $f"; cat "$f"; echo; fi
done
