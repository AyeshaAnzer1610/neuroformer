#!/usr/bin/env bash
#SBATCH --job-name=mh-frozen
#SBATCH --account=def-abais_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#
# Frozen-readout MIPDB screening for one backbone, saving per-subject OOF predictions
# (for the multi-backbone ensemble). Try a stronger FM (CBraMod = TUAB SOTA) and EEGPT
# to see if a richer frozen representation bumps the 0.74 LaBraM number.
#   sbatch scripts/frozen_backbone_sbatch.sh <labram|cbramod|eegpt>
set -euo pipefail
BB=${1:?need backbone}
if [ "$BB" = "labram" ]; then SF=200; WIN=15; else SF=250; WIN=4; fi

module load StdEnv/2023 gcc python/3.11 cuda cudnn
export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -vF "$HOME/cuda-toolkit" | paste -sd:)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/mh311gpu/bin/activate
cd ~/mental-health
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "=== frozen $BB on MIPDB (sfreq=$SF win=${WIN}s), 5x5, saving preds ==="
python src/finetune_labram.py --input "$SCRATCH/mh/data/mipdb" --labels data/labels.csv \
    --backbone "$BB" --sfreq "$SF" --window-sec "$WIN" --freeze-backbone --all-recordings \
    --epochs 8 --batch 16 --max-windows 40 --folds 5 --repeats 5 \
    --out "results/frozen_${BB}.csv" --save-preds "results/frozen_${BB}_preds.csv"
echo "=== done -> results/frozen_${BB}.csv (+ _preds). Compare to LaBraM 0.74; then ensemble. ==="