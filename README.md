# NeuroFormer

**A foundation-model, missing-modality transformer for transdiagnostic
mental-health screening from EEG, eye-tracking, and behaviour — with external
validation.**

NeuroFormer fuses three non-invasive modalities through modality-specific
transformer encoders and a cross-modal fusion transformer. This repository
contains the model, the leak-free evaluation pipeline, and the analysis behind the
IEEE Access submission.

## Summary of findings (leak-free, subject-level evaluation)

- **Foundation-model EEG is the locus of signal.** Frozen LaBraM embeddings reach
  ROC-AUC ≈ 0.66 for transdiagnostic screening on MIPDB (N≈118), versus ≈ 0.45 for
  the prior hand-crafted EEG features.
- **Honest multimodal analysis.** Eye-tracking and behaviour are near chance in
  this cohort; multimodal fusion does not exceed EEG-alone — reported transparently
  rather than overclaimed.
- **External validation.** On the independent Healthy Brain Network (HBN) cohort,
  the frozen EEG representation predicts the general psychopathology factor
  out-of-distribution — scaled to ~2,000+ subjects across releases.
- **Calibrated, referral-aware** predictions (temperature scaling + conformal) for
  screening triage.

All performance is measured under strict subject-level, leakage-free
cross-validation.

## Repository structure

| Path | Contents |
|---|---|
| `src/` | NeuroFormer model, EEG/eye/behaviour feature extraction, fusion, calibration, fine-tuning, evaluation |
| `scripts/` | Data download + Alliance-cluster (SLURM) job scripts |
| `notebooks/` | Exploratory / live analysis |
| `docs/resubmission/` | Method writeups, results, reviewer responses, literature grounding |
| `figures/` | Manuscript figures |
| `results/` | Metrics tables (small CSVs) |

## Data

- **MIPDB** (Child Mind Institute) — primary cohort. Public; downloaded from S3.
- **HBN-EEG** (OpenNeuro ds005505–ds005515) — external validation. Public.
- Both use the 128/129-channel EGI HydroCel net (same montage → clean cross-dataset
  transfer). Raw data is public and re-downloadable (`scripts/download_data.sh`); it
  is **not** committed to git.

## Reproduce

```bash
# 1. environment (Alliance cluster login node)
bash scripts/setup_env_nibi.sh
# 2. data (public S3)
bash scripts/download_data.sh mipdb
# 3. EEG foundation-model embeddings + the head-to-head + comparison
sbatch scripts/extract_cohort_sbatch.sh
# 4. external validation on HBN (all releases)
bash scripts/download_data.sh hbn-rest-releases 1 11
sbatch scripts/extract_hbn_all_sbatch.sh
```
See `docs/resubmission/` for method details and the full result writeups
(`Phase1-Results.md`, `Phase2-Results.md`, `Phase4-Results.md`).

## Key methods

| File | Role |
|---|---|
| `src/neuroformer.py` | The NeuroFormer architecture (encoders + cross-modal fusion + classifier) |
| `src/extract_eeg_embeddings.py` | Frozen LaBraM EEG embeddings (QC, leak-free, batched) |
| `src/extract_eye_features.py`, `src/extract_behaviour_features.py` | Eye / behaviour features from raw |
| `src/compare_fm_vs_handcrafted.py` | Head-to-head + fusion evaluation |
| `src/train_neuroformer.py` | Leak-free training with modality dropout |
| `src/eval_hbn.py` | External HBN validation |
| `src/calibrate.py` | Calibration / selective (referral-aware) prediction |
| `src/finetune_labram.py` | End-to-end LaBraM fine-tuning |
| `src/make_figures.py` | Manuscript figures |

## Citation

A. Anzer and A. Bais, "NeuroFormer: A Foundation-Model, Missing-Modality
Transformer for Transdiagnostic Mental-Health Screening," *IEEE Access* (under
review).

## License

Code: MIT. Data: per the original dataset licenses (MIPDB, HBN).
