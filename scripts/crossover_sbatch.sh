#!/usr/bin/env bash
#SBATCH --job-name=mh-xover
#SBATCH --account=def-abais_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=11:59:00
#SBATCH --output=%x-%j.out
#
# ADAPTATION x SAMPLE-SIZE CROSSOVER (the TNSRE contribution).
# One cell = one (N, mode, cohort): does fine-tuning beat frozen at training size N?
# Hypothesis: frozen wins at small N; full/LoRA overtake as N grows -> characterize N*.
#
#   sbatch scripts/crossover_sbatch.sh <N> <frozen|full|lora> <mipdb|hbn> [labram|cbramod|eegpt]
# e.g.  sbatch scripts/crossover_sbatch.sh 400 full hbn cbramod
#
# Run the grid (frozen + full [+ lora] across N, x backbones) to trace the crossover.
# SMOKE-test a new backbone first:  add --smoke via finetune_labram (see notes).
set -euo pipefail
N=${1:?need N}; MODE=${2:?need frozen|full|lora}; COHORT=${3:-hbn}; BB=${4:-labram}
# per-backbone sampling/window (each FM expects its own): LaBraM 200Hz/15s; CBraMod/EEGPT 250Hz/4s
if [ "$BB" = "labram" ]; then SF=200; WIN=15; else SF=250; WIN=4; fi

module load StdEnv/2023 gcc python/3.11 cuda cudnn
export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -vF "$HOME/cuda-toolkit" | paste -sd:)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/envs/mh311gpu/bin/activate
cd ~/mental-health
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

if [ "$COHORT" = "mipdb" ]; then
    INPUT=$SCRATCH/mh/data/mipdb; LABELS=data/labels.csv; MATCH=""
else
    # HBN raw: prefer /project (moved off the full SCRATCH), else SCRATCH
    INPUT=/project/def-abais/$USER/mh/data/hbn_all
    [ -d "$INPUT" ] || INPUT=$SCRATCH/mh/data/hbn_all
    LABELS=$(dirname "$INPUT")/participants_all.tsv
    [ -f "$LABELS" ] || LABELS=$SCRATCH/mh/data/hbn_all/participants_all.tsv
    MATCH="--match RestingState"
fi

case "$MODE" in
    frozen) FLAG="--freeze-backbone" ;;
    full)   FLAG="" ;;
    lora)   FLAG="--lora --lora-r 8" ;;
    *) echo "bad MODE $MODE"; exit 1 ;;
esac

echo "[xover] backbone=$BB cohort=$COHORT N=$N mode=$MODE sfreq=$SF win=${WIN}s input=$INPUT"
python src/finetune_labram.py --input "$INPUT" --labels "$LABELS" $MATCH \
    --backbone "$BB" --sfreq "$SF" --window-sec "$WIN" \
    --max-subj "$N" $FLAG --all-recordings --epochs 8 --batch 16 --max-windows 30 \
    --folds 3 --repeats 1 \
    --out "results/crossover_${BB}_${COHORT}_${MODE}_N${N}.csv"
echo "[xover] done -> results/crossover_${BB}_${COHORT}_${MODE}_N${N}.csv"
