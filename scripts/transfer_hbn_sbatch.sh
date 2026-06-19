#!/usr/bin/env bash
#SBATCH --job-name=mh-transfer
#SBATCH --account=def-abais              # CPU allocation (no GPU needed — cached embeddings)
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out               # job log: mh-transfer-<jobid>.out
#
# Does identity-disentanglement improve MIPDB -> HBN transfer? Trains a window-level
# head on MIPDB (frozen LaBraM embeddings) with the disentangler OFF vs ON, applies to
# HBN, reports median-split AUC on p_factor. CPU-only (operates on cached *_labram.npz).
#
# Submit ONCE from a LOGIN node:  sbatch scripts/transfer_hbn_sbatch.sh
# PREREQS: HBN embeddings extracted (scripts/extract_hbn_all_sbatch.sh) and
#          ~/envs/mh311 built (scripts/setup_env_nibi.sh).
set -euo pipefail

module load StdEnv/2023 python/3.11
source ~/envs/mh311/bin/activate
cd ~/mental-health

# All-releases HBN (~2000 subj). For single-release R1 instead, swap to:
#   processed_hbn  and  hbn_r1/participants.tsv
HBN_EMB=$SCRATCH/mh/data/processed_hbn_all
HBN_PART=$SCRATCH/mh/data/hbn_all/participants_all.tsv
if [ ! -d "$HBN_EMB" ]; then           # fall back to single-release if _all not present
    HBN_EMB=$SCRATCH/mh/data/processed_hbn
    HBN_PART=$SCRATCH/mh/data/hbn_r1/participants.tsv
fi
echo "[paths] HBN_EMB=$HBN_EMB  HBN_PART=$HBN_PART"

python experiments/transfer_hbn.py \
    --mipdb-dir "$SCRATCH/mh/data/processed" \
    --hbn-dir   "$HBN_EMB" \
    --hbn-part  "$HBN_PART" \
    --mipdb-labels data/labels.csv \
    --seeds 3 \
    --out results/transfer_hbn_disentangle.csv

echo "=== done; results/transfer_hbn_disentangle.csv (OFF vs disentangle-ON transfer AUC) ==="
