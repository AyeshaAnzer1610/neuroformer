#!/usr/bin/env bash
#SBATCH --job-name=nf-stability
#SBATCH --account=def-abais_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#
# HONEST NeuroFormer stability: pooled-OOF AUC + confound-adjusted increment reported as
# mean +/- std over multiple seeds (NeuroFormer is seed-fragile). Skips slow Table 3.
#   sbatch scripts/nf_stability_sbatch.sh
set -euo pipefail
module load StdEnv/2023 gcc python/3.11 cuda cudnn
export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -vF "$HOME/cuda-toolkit" | paste -sd:)
source ~/envs/mh311gpu/bin/activate
cd ~/mental-health

echo "=== NeuroFormer seed-stability: Table 2 (pooled-OOF) + confound, 8 seeds ==="
python experiments/neuroformer_baselines.py --repeats 5 --nf-repeats 5 --nf-seeds 8 --skip-t3 \
    --out-t2 results/table2_baselines.csv --out-t3 results/table3_ablation.csv
echo "=== done -> table2_baselines.csv, confound_adjusted.csv, nf_seed_stability.csv ==="
