#!/usr/bin/env bash
#SBATCH --job-name=mh-nf
#SBATCH --account=def-abais_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#
# END-TO-END complete architecture: the NeuroFormer cross-modal FUSION TRANSFORMER
# over EEG (frozen LaBraM embeddings) + eye-tracking features, subject-level leak-free
# CV with modality dropout. This is "the transformer we had", run end-to-end -> the
# definitive complete-architecture screening AUC (vs EEG-alone ~0.66 / frozen 0.74).
#
# Submit:  sbatch scripts/train_neuroformer_sbatch.sh
set -euo pipefail
module load StdEnv/2023 gcc python/3.11 cuda cudnn
export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -vF "$HOME/cuda-toolkit" | paste -sd:)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/mh311gpu/bin/activate
cd ~/mental-health
PROC=$SCRATCH/mh/data

echo "=== NeuroFormer fusion transformer (EEG LaBraM + eye), 5x3 subject-level CV ==="
python src/train_neuroformer.py \
    --eeg-dir "$PROC/processed" \
    --eye-dir "$PROC/processed_eye" \
    --labels data/labels.csv \
    --folds 5 --repeats 3 --epochs 60 \
    --out results/neuroformer_cv.csv

echo "=== done -> results/neuroformer_cv.csv  (compare to EEG-alone 0.66 / frozen-window 0.74) ==="