#!/usr/bin/env bash
#SBATCH --job-name=mh-hbn
#SBATCH --account=def-abais
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#
# Phase 4: extract frozen LaBraM embeddings from HBN RestingState EEG, then run
# the within-HBN + transfer evaluation. Submit from a LOGIN node AFTER the full
# HBN sync (scripts/download_data.sh hbn-r1) and after the weights are cached.
#   sbatch scripts/extract_hbn_sbatch.sh
set -euo pipefail
module load StdEnv/2023 python/3.11
source ~/envs/mh311/bin/activate
cd ~/mental-health

HBN=$SCRATCH/mh/data/hbn_r1
HBN_EMB=$SCRATCH/mh/data/processed_hbn
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "=== [1/2] HBN RestingState LaBraM embeddings (jobs=4) ==="
python src/extract_eeg_embeddings.py --batch --jobs 4 --match RestingState \
    --input "$HBN" --output-dir "$HBN_EMB"

echo "=== [2/2] within-HBN + transfer evaluation ==="
python src/eval_hbn.py \
    --hbn-dir   "$HBN_EMB" \
    --hbn-part  "$HBN/participants.tsv" \
    --mipdb-dir "$SCRATCH/mh/data/processed" \
    --mipdb-labels data/labels.csv

echo "=== done; results/phase4_hbn.csv ==="
