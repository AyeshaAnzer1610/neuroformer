#!/usr/bin/env bash
#SBATCH --job-name=mh-finetune
#SBATCH --account=def-abais_gpu          # Ayesha's GPU allocation (PI: Bais)
#SBATCH --gres=gpu:h100:1                 # Nibi requires an explicit GPU type
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out               # job log: mh-finetune-<jobid>.out
#
# End-to-end LaBraM fine-tuning for transdiagnostic screening (the one legitimate
# lever above the frozen ~0.66 bar). Window-level training, subject-level CV.
#
# Submit from a LOGIN node:  sbatch scripts/finetune_sbatch.sh
#
# PREREQUISITES (do once, on a LOGIN node):
#   1. bash scripts/setup_env_gpu.sh          # builds ~/envs/mh311gpu (Alliance torch)
#   2. cache LaBraM weights (run any extraction once -> ~/.cache/huggingface)
set -euo pipefail

# Load the DEFAULT cuda/cudnn from StdEnv/2023 (matches the Alliance torch wheel;
# do NOT pin a specific cuda module -- a mismatch causes the cusparse/nvJitLink
# "undefined symbol" errors we hit before).
module load StdEnv/2023 gcc python/3.11 cuda cudnn

# Strip any stale user-local CUDA toolkit (~/cuda-toolkit) whose old libnvJitLink
# shadows the cluster's and breaks torch import (cusparse undefined symbol).
export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
    | grep -vF "$HOME/cuda-toolkit" | paste -sd:)

source ~/envs/mh311gpu/bin/activate

cd ~/mental-health
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

MIPDB=$SCRATCH/mh/data/mipdb

echo "=== [0/2] preflight: validate CUDA env (fast pass/fail) ==="
python src/finetune_labram.py --input "$MIPDB" --labels data/labels.csv \
    --max-subj 4 --smoke

echo "=== [1/2] fine-tune (frozen backbone + trainable head, AMP) ==="
python src/finetune_labram.py --input "$MIPDB" --labels data/labels.csv \
    --all-recordings --freeze-backbone --epochs 8 --batch 16 --max-windows 40 \
    --folds 5 --repeats 1 --out results/finetune_labram_frozen.csv

echo "=== [2/2] fine-tune (full backbone, smaller LR, AMP) ==="
python src/finetune_labram.py --input "$MIPDB" --labels data/labels.csv \
    --all-recordings --epochs 6 --batch 12 --lr 3e-5 --max-windows 40 \
    --folds 5 --repeats 1 --out results/finetune_labram_full.csv

echo "=== done; compare results/finetune_labram_*.csv against the 0.66 frozen bar ==="
