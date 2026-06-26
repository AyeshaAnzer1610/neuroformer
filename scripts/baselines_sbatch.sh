#!/usr/bin/env bash
#SBATCH --job-name=nf-baselines
#SBATCH --account=def-abais_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#
# HONEST Table 2 (baselines) + Table 3 (modality ablation, +/- aux loss) for the
# corrected NeuroFormer paper, under one leak-free 5x5 protocol. Hand-crafted features
# are tiny -> a MIG slice is plenty.  sbatch scripts/baselines_sbatch.sh
set -euo pipefail
module load StdEnv/2023 gcc python/3.11 cuda cudnn
export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -vF "$HOME/cuda-toolkit" | paste -sd:)
source ~/envs/mh311gpu/bin/activate
cd ~/mental-health

echo "=== NeuroFormer baselines: Table 2 + Table 3 (leak-free 5x5) ==="
python experiments/neuroformer_baselines.py --repeats 5 --nf-repeats 5 --nf-seeds 8 \
    --out-t2 results/table2_baselines.csv --out-t3 results/table3_ablation.csv
echo "=== done -> results/table2_baselines.csv + results/table3_ablation.csv ==="
