#!/usr/bin/env bash
#SBATCH --job-name=mh-fusion
#SBATCH --account=def-abais              # Ayesha's Alliance allocation (PI: Bais)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out               # job log: mh-fusion-<jobid>.out
#
# Re-run the Phase 2 missing-modality fusion eval as 2-MODALITY (EEG + eye).
# Behaviour was dropped, so the previous results/phase2_fusion.csv fusion/concat/
# drop rows (computed on 3 modalities) are STALE; this regenerates them honestly.
#
# Submit from a LOGIN node:  sbatch scripts/eval_fusion_sbatch.sh
# PREREQS: ~/envs/mh311 built (scripts/setup_env_nibi.sh) and embeddings extracted
#          to $SCRATCH/mh/data/processed (EEG LaBraM) + processed_eye (eye features).
set -euo pipefail

module load StdEnv/2023 python/3.11
source ~/envs/mh311/bin/activate

cd ~/mental-health

PROC=$SCRATCH/mh/data
echo "=== Phase 2 fusion (EEG + eye, 2-modality), 5x10 subject-level CV ==="
python src/eval_fusion.py \
    --eeg-dir "$PROC/processed" \
    --eye-dir "$PROC/processed_eye" \
    --labels data/labels.csv \
    --out results/phase2_fusion.csv

echo "=== done; results/phase2_fusion.csv refreshed (EEG-alone bar = 0.66) ==="
