#!/usr/bin/env bash
# Claude's lane of the split run: 20_cv_schemes + 23_ds003478_confound, in
# parallel (distinct output files; the user runs 17 + 21 separately).
set -u
cd "$(dirname "$0")/.."
mkdir -p results results/logs
export SUITE_THREADS=${SUITE_THREADS:-4}
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] my-lane start: 20_cv_schemes + 23_ds003478_confound (parallel)"

( echo "[$(ts)] START 20_cv_schemes"
  python src/20_cv_schemes.py --seeds 20 --n-boot 2000 --b632 200 \
      --out results/table_cv_schemes.csv --dist results/cv_fold_distribution.csv
  echo "[$(ts)] 20_cv_schemes exit=$?" ) > results/logs/20_cv_schemes.log 2>&1 &
P20=$!

( echo "[$(ts)] START 23_ds003478_confound"
  python src/23_ds003478_confound.py --n-perm 1000 --out results/ds2_confound.csv
  echo "[$(ts)] 23_ds003478_confound exit=$?" ) > results/logs/23_ds2_confound.log 2>&1 &
P23=$!

wait $P20; echo "[$(ts)] 20 joined"
wait $P23; echo "[$(ts)] 23 joined"
echo "[$(ts)] my-lane complete"
