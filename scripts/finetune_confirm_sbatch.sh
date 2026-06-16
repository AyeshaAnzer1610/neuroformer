#!/usr/bin/env bash
#SBATCH --job-name=mh-ft-confirm
#SBATCH --account=def-abais_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#
# Confirmation run: the WINNING regime (frozen backbone + window-level training)
# at the full 5x10 repeated CV, to lock the headline ROC-AUC and its CI.
#
# Submit from a LOGIN node:  sbatch scripts/finetune_confirm_sbatch.sh
set -euo pipefail

module load StdEnv/2023 gcc python/3.11 cuda cudnn
# Strip stale ~/cuda-toolkit (old cu121 leftover) that breaks torch import.
export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
    | grep -vF "$HOME/cuda-toolkit" | paste -sd:)
source ~/envs/mh311gpu/bin/activate

cd ~/mental-health
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python src/finetune_labram.py --input "$SCRATCH/mh/data/mipdb" --labels data/labels.csv \
    --all-recordings --freeze-backbone --epochs 8 --batch 16 --max-windows 40 \
    --folds 5 --repeats 10 --out results/finetune_labram_frozen_r10.csv

echo "=== done; 5x10 frozen confirmation in results/finetune_labram_frozen_r10.csv ==="
