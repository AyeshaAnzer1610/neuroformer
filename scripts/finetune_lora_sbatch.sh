#!/usr/bin/env bash
#SBATCH --job-name=mh-lora
#SBATCH --account=def-abais_gpu          # Ayesha's GPU allocation (PI: Bais)
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out               # job log: mh-lora-<jobid>.out
#
# PEFT experiment: does LoRA-adapting frozen LaBraM beat BOTH the frozen-readout
# bar (~0.66) and full fine-tuning (~0.69) on small-N transdiagnostic screening?
# This is the headline parameter-efficient claim and answers "model too big".
# Same subject-level CV / data loading as the other fine-tune runs -> directly
# comparable numbers.
#
# Submit from a LOGIN node:  sbatch scripts/finetune_lora_sbatch.sh
# PREREQS: bash scripts/setup_env_gpu.sh  (~/envs/mh311gpu) ; LaBraM weights cached.
set -euo pipefail

module load StdEnv/2023 gcc python/3.11 cuda cudnn
export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
    | grep -vF "$HOME/cuda-toolkit" | paste -sd:)
source ~/envs/mh311gpu/bin/activate

cd ~/mental-health
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MIPDB=$SCRATCH/mh/data/mipdb

echo "=== [0/2] preflight: LoRA injection + one fwd/bwd (validates injection on Nibi) ==="
# Confirms LORA_PATTERNS actually match LaBraM's Linear layers; prints trainable %.
python src/finetune_labram.py --input "$MIPDB" --labels data/labels.csv \
    --max-subj 4 --lora --smoke

echo "=== [1/2] LoRA PEFT (frozen backbone + rank-8 LoRA + head, AMP) ==="
python src/finetune_labram.py --input "$MIPDB" --labels data/labels.csv \
    --all-recordings --lora --lora-r 8 --epochs 8 --batch 16 --max-windows 40 \
    --folds 5 --repeats 2 --out results/finetune_labram_lora.csv \
    --save-preds results/finetune_labram_lora_preds.csv

echo "=== [2/2] frozen-readout baseline repro (head only) for a same-run comparison ==="
python src/finetune_labram.py --input "$MIPDB" --labels data/labels.csv \
    --all-recordings --freeze-backbone --epochs 8 --batch 16 --max-windows 40 \
    --folds 5 --repeats 2 --out results/finetune_labram_frozen_r2.csv

echo "=== done. Compare: LoRA vs frozen (~0.66) vs full FT (~0.69, results/finetune_labram_full.csv) ==="
