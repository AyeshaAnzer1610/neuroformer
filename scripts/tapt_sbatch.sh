#!/usr/bin/env bash
#SBATCH --job-name=mh-tapt
#SBATCH --account=def-abais_gpu          # GPU allocation (backbone trains)
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out               # job log: mh-tapt-<jobid>.out
#
# TAPT Stage 1 (MIPDB only): self-supervised continued pretraining of LaBraM on
# UNLABELED MIPDB windows, then frozen-readout 5x2 CV to test if it beats the
# frozen-pretrained ~0.74 ceiling. If it helps, Stage 2 adds HBN (--extra-dirs).
#
# Submit ONCE from a LOGIN node:  sbatch scripts/tapt_sbatch.sh
# PREREQS: scripts/setup_env_gpu.sh (~/envs/mh311gpu); LaBraM weights cached.
set -euo pipefail

module load StdEnv/2023 gcc python/3.11 cuda cudnn
export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
    | grep -vF "$HOME/cuda-toolkit" | paste -sd:)
source ~/envs/mh311gpu/bin/activate

cd ~/mental-health
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Full-backbone + two-view contrastive over ~1935 tokens is memory-heavy; reduce
# fragmentation so a moderate batch fits on the H100.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MIPDB=$SCRATCH/mh/data/mipdb
# SCRATCH is quota-bound; write the checkpoint to project space (925 GiB free).
CKPT=/project/def-abais/$USER/mh/ckpt/labram_tapt_mipdb.pt
mkdir -p "$(dirname "$CKPT")"

echo "=== [0/3] smoke: build backbone + projector + one contrastive step (batch 12) ==="
python src/tapt_pretrain.py --input "$MIPDB" --max-subj 4 --batch 12 --smoke

echo "=== [1/3] TAPT: contrastive continued pretraining on unlabeled MIPDB ==="
python src/tapt_pretrain.py --input "$MIPDB" --all-recordings \
    --epochs 50 --batch 12 --lr 1e-4 --out "$CKPT"

echo "=== [2/3] frozen-readout eval of the TAPT-adapted backbone (vs ~0.74 frozen bar) ==="
python src/finetune_labram.py --input "$MIPDB" --labels data/labels.csv \
    --all-recordings --freeze-backbone --backbone-weights "$CKPT" \
    --epochs 8 --batch 16 --max-windows 40 --folds 5 --repeats 2 \
    --out results/finetune_labram_tapt_5x2.csv

echo "=== [3/3] done. Compare results/finetune_labram_tapt_5x2.csv to frozen 0.739 (5x5) ==="
echo "If TAPT > frozen-pretrained, Stage 2: add HBN via --extra-dirs \$SCRATCH/mh/data/hbn_all"
