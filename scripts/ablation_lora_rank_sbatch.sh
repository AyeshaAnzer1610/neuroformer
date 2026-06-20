#!/usr/bin/env bash
#SBATCH --job-name=mh-abl-lora
#SBATCH --account=def-abais_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#
# LoRA-rank ablation (MIPDB, full cohort): does PEFT capacity (r) change the result?
# Reviewer-expected sensitivity for the PEFT comparison.
#   sbatch scripts/ablation_lora_rank_sbatch.sh <r>     # e.g. 4, 8, 16, 32
set -euo pipefail
R=${1:?need LoRA rank r}

module load StdEnv/2023 gcc python/3.11 cuda cudnn
export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -vF "$HOME/cuda-toolkit" | paste -sd:)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/mh311gpu/bin/activate
cd ~/mental-health
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[abl] LoRA rank r=$R on MIPDB (5x5)"
python src/finetune_labram.py --input "$SCRATCH/mh/data/mipdb" --labels data/labels.csv \
    --all-recordings --lora --lora-r "$R" --epochs 8 --batch 16 --max-windows 40 \
    --folds 5 --repeats 5 --out "results/ablation_lora_r${R}.csv"
echo "[abl] done -> results/ablation_lora_r${R}.csv (compare to frozen 0.739, LoRA-r8 0.701)"
