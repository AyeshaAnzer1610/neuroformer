#!/usr/bin/env bash
# Full-scale run of the whole model-suite battery (scripts 17-23), sequential,
# cheapest-first so usable results land early. Each job is isolated (|| true) so
# one failure does not stop the rest. Logs to results/logs/.
# Launched in the background by Claude on 2026-07-06.
set -u
cd "$(dirname "$0")/.."
mkdir -p results results/logs
export SUITE_THREADS=${SUITE_THREADS:-4}
ts() { date '+%Y-%m-%d %H:%M:%S'; }
run() { echo "[$(ts)] START $1"; shift; ( "$@" ) && echo "[$(ts)] DONE" || echo "[$(ts)] FAILED"; }

echo "[$(ts)] ===== full suite start ====="

# 1. extras (cheap: DeLong + calibration + per-subject OOF)
run "22_extras" python src/22_extras.py \
    --out-delong results/delong_pairwise.csv \
    --out-calib results/calibration_suite.csv \
    --out-oof results/per_subject_oof.csv 2>&1 | tee results/logs/22_extras.log

# 2. per-model CIs (moderate)
run "18_bootstrap_ci" python src/18_bootstrap_ci.py --n-boot 2000 \
    --out results/table_ci.csv 2>&1 | tee results/logs/18_bootstrap_ci.log

# 3. ds003478 full suite (moderate; refresh alongside the rest)
run "19_ds003478_suite" python src/19_ds003478_suite.py --n-boot 2000 \
    --out results/ds2_suite.csv 2>&1 | tee results/logs/19_ds2_suite.log

# 4. alternative CV schemes (LOSO / multiseed / .632+ / nested)
run "20_cv_schemes" python src/20_cv_schemes.py --seeds 20 --n-boot 2000 --b632 200 \
    --out results/table_cv_schemes.csv \
    --dist results/cv_fold_distribution.csv 2>&1 | tee results/logs/20_cv_schemes.log

# 5. ds003478 confound + permutation (moderate-heavy)
run "23_ds003478_confound" python src/23_ds003478_confound.py --n-perm 1000 \
    --out results/ds2_confound.csv 2>&1 | tee results/logs/23_ds2_confound.log

# 6. 14-model permutation table (heavy)
run "17_permutation_suite" python src/17_permutation_suite.py --n-perm 1000 \
    --out results/table_perm.csv 2>&1 | tee results/logs/17_permutation_suite.log

# 7. confound designs (heaviest: matched 200 + conditioned permutation 1000)
run "21_confound_designs" python src/21_confound_designs.py --matched-n 200 --n-perm 1000 \
    --outdir results 2>&1 | tee results/logs/21_confound_designs.log

echo "[$(ts)] ===== full suite complete ====="
