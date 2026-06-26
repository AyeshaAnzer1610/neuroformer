#!/usr/bin/env bash
#SBATCH --job-name=nf-regularize
#SBATCH --account=def-abais_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#
# Is NeuroFormer's 0.62 a capacity/regularization artifact? Pre-declared config sweep,
# pooled-OOF mean +/- std over seeds, FULL sweep reported (no cherry-pick).
#   sbatch scripts/nf_regularize_sbatch.sh
set -euo pipefail
module load StdEnv/2023 gcc python/3.11 cuda cudnn
export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -vF "$HOME/cuda-toolkit" | paste -sd:)
source ~/envs/mh311gpu/bin/activate
cd ~/mental-health

echo "=== NeuroFormer regularization sweep (5 configs x 4 seeds, pooled-OOF) ==="
python experiments/nf_regularize.py --seeds 4 --nf-repeats 5 --out results/nf_regularize.csv
echo "=== done -> results/nf_regularize.csv ==="
