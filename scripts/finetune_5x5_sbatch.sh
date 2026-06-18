#!/usr/bin/env bash
#SBATCH --job-name=mh-5x5
#SBATCH --account=def-abais_gpu          # Ayesha's GPU allocation (PI: Bais)
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=11:59:00
#SBATCH --output=%x-%j.out               # job log: mh-5x5-<jobid>.out
#
# MATCHED 5x5 CV confirmation: frozen (head only) vs LoRA (PEFT) vs full fine-tune.
# Tighter CIs to settle whether backbone adaptation beats a frozen LaBraM + head on
# N~118. Identical data/QC (--all-recordings --max-windows 40); arm-appropriate
# training budgets. Writes three *_5x5.csv files for a clean comparison.
#
# Submit ONCE from a LOGIN node:  sbatch scripts/finetune_5x5_sbatch.sh
# (do NOT submit repeatedly -- one job is enough). Check: bash scripts/check_lora.sh
set -euo pipefail

module load StdEnv/2023 gcc python/3.11 cuda cudnn
export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
    | grep -vF "$HOME/cuda-toolkit" | paste -sd:)
source ~/envs/mh311gpu/bin/activate

cd ~/mental-health
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MIPDB=$SCRATCH/mh/data/mipdb
COMMON="--input $MIPDB --labels data/labels.csv --all-recordings --max-windows 40 \
        --folds 5 --repeats 5"

echo "=== [1/3] frozen (head only), 5x5 ==="
python src/finetune_labram.py $COMMON --freeze-backbone --epochs 8 --batch 16 \
    --out results/finetune_labram_frozen_5x5.csv

echo "=== [2/3] LoRA r=8 (PEFT), 5x5 ==="
python src/finetune_labram.py $COMMON --lora --lora-r 8 --epochs 8 --batch 16 \
    --out results/finetune_labram_lora_5x5.csv \
    --save-preds results/finetune_labram_lora_5x5_preds.csv

echo "=== [3/3] full fine-tune, 5x5 ==="
python src/finetune_labram.py $COMMON --epochs 6 --batch 12 --lr 3e-5 \
    --out results/finetune_labram_full_5x5.csv

echo "=== done. Compare the three *_5x5.csv means/CIs (bash scripts/check_lora.sh) ==="
