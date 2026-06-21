#!/usr/bin/env bash
#SBATCH --job-name=mh-ocufm
#SBATCH --account=def-abais_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#
# Pretrain OcuFM (gaze FM) on GazeBase. NO pymovements (its polars/pyarrow deps don't
# build on the cluster) -- we parse GazeBase CSVs directly with pandas.
# PREREQ (ONCE, login node): download + unzip GazeBase CSVs to
#   /project/def-abais/$USER/mh/data/gazebase/   (from figshare: "GazeBase" Sci Data 2021)
#   e.g.  wget <figshare round zips> -P $GAZE && (cd $GAZE && for z in *.zip; do unzip -q $z; done)
#
# Submit:  sbatch scripts/ocufm_pretrain_sbatch.sh
set -euo pipefail
module load StdEnv/2023 gcc python/3.11 cuda cudnn
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
