#!/usr/bin/env bash
# Run ONCE on a Nibi LOGIN node (has internet) before sbatch'ing the cohort.
#
#   bash scripts/warm_cache_and_check.sh [SUBJECT_DIR]
#
# Does two things the offline batch job can't:
#   1. Downloads + caches the LaBraM pretrained weights to ~/.cache/huggingface
#      (compute nodes have no internet, so this MUST happen on login first).
#   2. Runs one subject end-to-end as a sanity check — watch for the
#      "[win ] amplitude ... uV" line (~tens of uV = unit scaling is correct).
set -euo pipefail

module load StdEnv/2023 python/3.11
source ~/envs/mh311/bin/activate
cd ~/mental-health

SUBJECT=${1:-$SCRATCH/mh/data/mipdb/A00051955}
echo "=== sanity-check subject: $SUBJECT ==="

# 1. dry-run: validates load/preprocess/window + prints the uV amplitude line.
python src/extract_eeg_embeddings.py --input "$SUBJECT" --dry-run

# 2. real single-subject extraction: downloads/caches weights on this login node.
python src/extract_eeg_embeddings.py \
    --input  "$SUBJECT" \
    --output "$SCRATCH/mh/data/processed/$(basename "$SUBJECT")_labram.npz"

echo
echo "=== cache warmed + single subject extracted ==="
echo "If the [win ] amplitude line read ~tens of uV, you're good to:"
echo "    sbatch scripts/extract_cohort_sbatch.sh"
