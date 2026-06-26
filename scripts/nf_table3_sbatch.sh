#!/usr/bin/env bash
#SBATCH --job-name=nf-table3
#SBATCH --account=def-abais_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#
# Table 3 ONLY: NeuroFormer modality ablation -- 7 non-empty subsets x {with, without aux
# reconstruction loss} = 14 leak-free 5x5 CV runs (pooled-OOF AUC, paired Wilcoxon for Daux).
# --only-t3 skips the Table 2 baselines + 8-seed NF stability loop, so it does NOT touch/clobber
# the already-committed table2_baselines.csv / perfold_table2.csv.
#   sbatch scripts/nf_table3_sbatch.sh
# Wall time: 14 NF runs x 25 folds. 6h is a safe ceiling on the 10GB H100 MIG slice; bump if it
# nears the limit. Outputs: results/table3_ablation.csv + results/perfold_table3.csv
set -euo pipefail
module load StdEnv/2023 gcc python/3.11 cuda cudnn
export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -vF "$HOME/cuda-toolkit" | paste -sd:)
source ~/envs/mh311gpu/bin/activate
cd ~/mental-health

echo "=== Table 3 ONLY: NeuroFormer modality ablation (7 subsets x +/- aux), leak-free 5x5 ==="
python experiments/neuroformer_baselines.py --repeats 5 --nf-repeats 5 --only-t3 \
    --out-t3 results/table3_ablation.csv
echo "=== done -> results/table3_ablation.csv (+ perfold_table3.csv) ==="
