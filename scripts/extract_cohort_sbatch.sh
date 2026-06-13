#!/usr/bin/env bash
#SBATCH --job-name=mh-extract
#SBATCH --account=def-REPLACE_ME        # <-- your Alliance allocation (e.g. def-supervisor)
#SBATCH --cpus-per-task=8               # parallel subject workers (matches --jobs below)
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out              # job log: mh-extract-<jobid>.out
#
# Extract BOTH Phase 1 EEG feature sets (LaBraM + hand-crafted) for the whole
# MIPDB cohort, in parallel across CPU cores, then run the head-to-head.
#
# Submit from a LOGIN node:  sbatch scripts/extract_cohort_sbatch.sh
#
# PREREQUISITE (do once, on a LOGIN node — compute nodes have NO internet):
#   the LaBraM pretrained weights must already be cached. Running the single
#   subject once on login (or `phase1_explore.ipynb` cell 3) downloads them to
#   ~/.cache/huggingface. After that this offline batch job works.
set -euo pipefail

module load StdEnv/2023 python/3.11
source ~/envs/mh311/bin/activate

cd ~/mental-health

NJOBS=${SLURM_CPUS_PER_TASK:-8}
MIPDB=$SCRATCH/mh/data/mipdb
FM_DIR=$SCRATCH/mh/data/processed
HC_DIR=$SCRATCH/mh/data/processed_handcrafted

# Compute nodes are offline: force HF to use the pre-cached weights, fail fast
# (with a clear error) if they were never downloaded on a login node.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "=== [1/3] LaBraM embeddings (jobs=$NJOBS) ==="
python src/extract_eeg_embeddings.py  --batch --jobs "$NJOBS" \
    --input "$MIPDB" --output-dir "$FM_DIR"

echo "=== [2/3] hand-crafted EEG features (jobs=$NJOBS) ==="
python src/extract_handcrafted_eeg.py --batch --jobs "$NJOBS" \
    --input "$MIPDB" --output-dir "$HC_DIR"

echo "=== [3/3] head-to-head comparison ==="
python src/compare_fm_vs_handcrafted.py \
    --fm-dir "$FM_DIR" --hc-dir "$HC_DIR" --labels data/labels.csv

echo "=== done; results in results/phase1_comparison.csv ==="
