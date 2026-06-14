#!/usr/bin/env bash
#SBATCH --job-name=mh-hbn-all
#SBATCH --account=def-abais
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#
# Phase 4 (scaled): extract frozen LaBraM RestingState embeddings for the FULL
# multi-release HBN cohort, then evaluate within-HBN (well-powered N). Submit
# AFTER:  bash scripts/download_data.sh hbn-rest-releases 1 11
# (and after the LaBraM weights are cached from Phase 1).
#   sbatch scripts/extract_hbn_all_sbatch.sh
set -euo pipefail
module load StdEnv/2023 python/3.11
source ~/envs/mh311/bin/activate
cd ~/mental-health

HBN=$SCRATCH/mh/data/hbn_all
HBN_EMB=$SCRATCH/mh/data/processed_hbn_all
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "=== [1/2] HBN RestingState embeddings, full cohort (jobs=8) ==="
python src/extract_eeg_embeddings.py --batch --jobs 8 --match RestingState \
    --input "$HBN" --output-dir "$HBN_EMB"

echo "=== [2/2] within-HBN + transfer evaluation (merged labels) ==="
python src/eval_hbn.py \
    --hbn-dir   "$HBN_EMB" \
    --hbn-part  "$HBN/participants_all.tsv" \
    --mipdb-dir "$SCRATCH/mh/data/processed" \
    --mipdb-labels data/labels.csv \
    --out results/phase4_hbn_all.csv

echo "=== done; results/phase4_hbn_all.csv ==="
