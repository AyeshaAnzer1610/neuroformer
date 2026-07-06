# Model-suite runbook — permutation table, per-model CIs, ds003478 suite

Three artifacts that were **not finished on the N=92 pipeline**. Scripts are built;
**you run them** (cluster) and paste the real numbers — nothing is fabricated.

All three share one model definition file so the permutation test, the CIs, and the
ds003478 replication evaluate the **identical** models under the same leakage-free
subject-level CV.

- `src/model_suite.py` — 14-model registry + data loaders + CV utilities (shared)
- `src/17_permutation_suite.py` — per-model label-permutation p-values → `results/table_perm.csv`
- `src/18_bootstrap_ci.py` — per-model per-fold + bootstrap 95% CIs → `results/table_ci.csv`
- `src/19_ds003478_suite.py` — same suite on ds003478 → `results/ds2_suite.csv`
- `src/20_cv_schemes.py` — LOSO / multi-seed / .632+ / nested CV → `results/table_cv_schemes.csv` (+ `cv_fold_distribution.csv`)
- `src/21_confound_designs.py` — matched subsampling / conditioned permutation / regress-out-vs-feature / honest-vs-leaky CIs → `results/confound_*.csv`, `honest_vs_leaky_ci.csv`
- `src/22_extras.py` — DeLong pairwise / calibration (Brier+ECE) / per-subject OOF → `results/delong_pairwise.csv`, `calibration_suite.csv`, `per_subject_oof.csv`
- `src/23_ds003478_confound.py` — ds003478 confound + permutation mirror → `results/ds2_confound.csv`

### Coverage of the requested checklist
| Requested | Script |
|---|---|
| Leave-one-subject-out CV | `20` (`loso`) |
| Nested CV | `20` (`nested`: sklearn via GridSearchCV, torch via generic cfg-grid on weight-decay — all 14) |
| Repeated k-fold across more seeds + spread | `20` (`multiseed`, full distribution in `cv_fold_distribution.csv`) |
| Bootstrap .632+ AUC | `20` (`b632plus`) |
| Seed/fold stability spread | `20` (`--dist`) |
| Full per-model permutation p | `17` |
| Covariate-matched subsampling (sex, joint) | `21` (`matched`) |
| Confound-conditioned permutation | `21` (`condperm`) |
| Regress-out vs confound-as-feature | `21` (`confvariants`) |
| Per-model CIs on honest-vs-leaky | `21` (`honestleaky`, all 4 concat sklearn models) + `18` |
| ds003478 full model suite | `19` |
| ds003478 confound + permutation | `23` |
| DeLong pairwise (N=92) | `22` (`delong`) |
| Calibration Brier/ECE (N=92) | `22` (`calibration`) |
| Per-subject OOF by group | `22` (`peroof`) |

## Models (14)
LogReg / SVM-RBF / RandomForest / MLP (concat) · Late-fusion · 1D-CNN / LSTM / TCN
(temporal, EEG windows) · Demographics (age+sex) · GMU · Cross-modal Transformer ·
Tensor Fusion Network · Low-rank Multimodal Fusion · NeuroFormer.

## Honesty notes (read before citing)
1. The authors' **original** NeuroFormer training pipeline is not in this repo. These are
   faithful, standard **reference implementations** (GMU: Arévalo 2017; TFN: Zadeh 2017;
   LMF: Liu 2018; cross-modal attn as in `src/11_fusion_models.py`) run on the real locked
   N=92 features. Report them as produced by *these* scripts, not as the external pipeline.
2. `advanced_multimodal.create_model()` (NeuroFormer) had a shape bug
   (`FrequencyAwareAttention` collapsed hidden_dim→band_dim, so it crashed for any input).
   **Fixed 2026-07-06** — the weighted per-band representations are now recombined by
   concatenation back to hidden_dim. NeuroFormer now runs and populates every table.
   If you have a *different* canonical NeuroFormer, drop it into `create_model()` and the
   whole suite picks it up. (The drivers still capture any model error as a `status`/`N/A`
   row rather than crashing the run.)
3. **ds003478 is EEG-only.** Every multimodal-fusion model (GMU, Cross-modal, TFN, LMF,
   NeuroFormer) is written as `N/A — requires eye/behaviour (absent in ds003478)`, not run
   degenerately and not fabricated. The EEG models (LogReg/SVM/RF/MLP, 1D-CNN/LSTM/TCN,
   demographics) run for real.

## Cohort
N=92 (20 positive) = EEG+eye present, locked in `cohort_final.txt`
(`data/aligned_resting` subject-level + `data/resting_features` per-window EEG).

## Cluster commands (full runs)
```bash
# 1. 14-model permutation table (heaviest — deep models use reduced CV in the null)
python src/17_permutation_suite.py --n-perm 1000 --out results/table_perm.csv

# 2. per-model CIs (per-fold + subject-bootstrap)
python src/18_bootstrap_ci.py --n-boot 2000 --out results/table_ci.csv

# 3. full suite on ds003478
python src/19_ds003478_suite.py --n-boot 2000 --out results/ds2_suite.csv
```

## Cluster commands — extended battery
```bash
python src/20_cv_schemes.py --seeds 20 --n-boot 2000 --b632 200 \
    --out results/table_cv_schemes.csv --dist results/cv_fold_distribution.csv
python src/21_confound_designs.py --matched-n 200 --n-perm 1000 --outdir results
python src/22_extras.py     # DeLong + calibration + per-subject OOF
python src/23_ds003478_confound.py --n-perm 1000 --out results/ds2_confound.csv
```

## Smoke tests (seconds, verify before submitting a job)
```bash
python src/17_permutation_suite.py --n-perm 20 --obs-repeats 2 --null-repeats 1 \
    --epochs 5 --only "LogReg (concat)" "Demographics (age+sex)"
python src/18_bootstrap_ci.py --n-boot 200 --repeats 2 --epochs 5 --only "LogReg (concat)"
python src/19_ds003478_suite.py --n-boot 100 --repeats 2 --epochs 5
python src/20_cv_schemes.py --seeds 2 --n-boot 100 --b632 20 --epochs 5 --only "LogReg (concat)"
python src/21_confound_designs.py --matched-n 3 --n-perm 20 --obs-repeats 2 --null-repeats 1 \
    --epochs 5 --only "LogReg (concat)" --outdir /tmp/s21
python src/22_extras.py --epochs 5 --only "LogReg (concat)" "Demographics (age+sex)"
python src/23_ds003478_confound.py --n-perm 20 --obs-repeats 2 --null-repeats 1 --epochs 5 \
    --only "LogReg (concat)"
```

### Cost warning
The permutation-based analyses (17, 21 `condperm`, 23) and `20`'s LOSO/.632+ train
every model many times. For the deep models at `--n-perm 1000` this is hours per model.
Use `--only` to restrict to the models you actually report, and keep `--null-repeats 2`.
Anything you drop for cost, note it in the manuscript rather than implying full coverage.

## Cost / knobs
- `--n-perm` dominates permutation cost. Deep models train per fold per permutation; keep
  `--null-repeats 2` (default) and consider `--only` to permutation-test just the reported
  models if 1000×14 is too slow. `log()` what you drop.
- `--epochs` (default 60) controls torch-model training length.
- `--k-eeg` (default 128) = top-variance EEG features kept (matches `11_fusion_models.py`).
- Set `SUITE_THREADS` env var to control torch CPU threads.
