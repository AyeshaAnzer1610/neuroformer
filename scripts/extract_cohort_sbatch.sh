#!/usr/bin/env bash
#SBATCH --job-name=mh-extract
#SBATCH --account=def-abais             # Ayesha's Alliance allocation (PI: Bais)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                        # LaBraM attention over ~1935 tokens is memory-heavy
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

# FM is memory-heavy (LaBraM attention) -> fewer workers; hand-crafted is light.
FM_JOBS=4
HC_JOBS=8
MIPDB=$SCRATCH/mh/data/mipdb
FM_DIR=$SCRATCH/mh/data/processed
HC_DIR=$SCRATCH/mh/data/processed_handcrafted

# Compute nodes are offline: force HF to use the pre-cached weights, fail fast
# (with a clear error) if they were never downloaded on a login node.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "=== [1/3] LaBraM embeddings (jobs=$FM_JOBS) ==="
python src/extract_eeg_embeddings.py  --batch --jobs "$FM_JOBS" \
    --input "$MIPDB" --output-dir "$FM_DIR"

echo "=== [2/3] hand-crafted EEG features (jobs=$HC_JOBS) ==="
python src/extract_handcrafted_eeg.py --batch --jobs "$HC_JOBS" \
    --input "$MIPDB" --output-dir "$HC_DIR"

echo "=== [3/3] head-to-head comparison ==="
python src/compare_fm_vs_handcrafted.py \
    --fm-dir "$FM_DIR" --hc-dir "$HC_DIR" --labels data/labels.csv

echo "=== done; results in results/phase1_comparison.csv ==="
