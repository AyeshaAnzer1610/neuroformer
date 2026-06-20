#!/usr/bin/env bash
#SBATCH --job-name=mh-ocufm
#SBATCH --account=def-abais_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#
# Pretrain OcuFM (gaze FM) on GazeBase. PREREQS done ONCE on a LOGIN node (internet):
#   deactivate 2>/dev/null
#   module load gcc arrow            # provides pyarrow (Alliance blocks the pip version)
#   source ~/envs/mh311gpu/bin/activate
#   pip install pymovements
#   GAZE=/project/def-abais/$USER/mh/data/gazebase
#   python -c "import pymovements as pm; pm.Dataset('GazeBase', path='$GAZE').download()"
# (compute nodes are offline, so download + pip must be on the login node first.)
#
# Submit:  sbatch scripts/ocufm_pretrain_sbatch.sh
set -euo pipefail
# 'arrow' module provides pyarrow (pymovements/polars need it); must precede venv activate.
module load StdEnv/2023 gcc python/3.11 arrow cuda cudnn
export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -vF "$HOME/cuda-toolkit" | paste -sd:)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/mh311gpu/bin/activate
cd ~/mental-health
GAZE=/project/def-abais/$USER/mh/data/gazebase
CKPT=/project/def-abais/$USER/mh/ckpt/ocufm_gazebase.pt
mkdir -p "$(dirname "$CKPT")"

echo "=== [0/2] smoke: GazeBase load + one masked-patch step ==="
python src/ocufm_pretrain.py --path "$GAZE" --smoke

echo "=== [1/2] pretrain OcuFM on GazeBase ==="
python src/ocufm_pretrain.py --path "$GAZE" --epochs 30 --batch 64 --out "$CKPT"

echo "=== [2/2] done -> $CKPT  (next: downstream gaze eval to validate the FM) ==="
