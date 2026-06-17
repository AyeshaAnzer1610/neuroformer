# NeuroFormer

**A two-foundation-model, missing-modality transformer for transdiagnostic
mental-health screening from EEG and eye-tracking — identity-disentangled, with
external validation.**

NeuroFormer fuses two non-invasive modalities, each through its own *foundation-model*
representation, adapted by a shared identity-disentangling adapter (CIDA) and combined
by a missing-modality fusion transformer:

- **EEG** → frozen **LaBraM** + **CIDA** ([`src/cida.py`](src/cida.py))
- **Eye-tracking** → frozen **OcuFM** (a proposed gaze foundation model on the LaBraM
  recipe, [`src/ocufm.py`](src/ocufm.py)) + **CIDA**

The unifying idea: both EEG and gaze are *identity-dominated* signals (which is what
produced our original data-leakage result), so one disentanglement adapter — **CIDA** —
serves both. Behaviour was dropped (below chance, ~0.40, in this cohort). This repo
contains the models, the leak-free evaluation pipeline, and the analysis behind the
IEEE Access submission. See [docs/resubmission/CIDA-Architecture-Design.md](docs/resubmission/CIDA-Architecture-Design.md)
and [docs/resubmission/OcuFM-Architecture-Design.md](docs/resubmission/OcuFM-Architecture-Design.md).

## Summary of findings (leak-free, subject-level evaluation)

- **Foundation-model EEG is the locus of signal.** Frozen LaBraM embeddings reach
  ROC-AUC ≈ 0.66 for transdiagnostic screening on MIPDB (N≈118), versus ≈ 0.45 for
  the prior hand-crafted EEG features.
- **Identity-disentangled adaptation (CIDA).** Both EEG and gaze foundation embeddings
  encode subject/site identity; CIDA factors identity out of the clinical signal via a
  parameter-efficient adapter (~2–3% trainable params) on the frozen backbones — directly
  targeting the leakage/overfit/external-collapse failure modes.
- **A foundation model for gaze (OcuFM).** Eye-tracking gains the same kind of pretrained
  representation EEG has, closing the modality asymmetry. *Proposed + runnable; not yet
  pretrained — no accuracy gain is claimed until it is.*
- **External validation.** On the independent Healthy Brain Network (HBN) cohort,
  the frozen EEG representation predicts the general psychopathology factor
  out-of-distribution — scaled to ~2,000+ subjects across releases.
- **Calibrated, referral-aware** predictions (temperature scaling + conformal) for
  screening triage.

All performance is measured under strict subject-level, leakage-free
cross-validation. Empirical headline signal remains EEG; OcuFM/CIDA gains are reported
honestly once run.

## Repository structure

| Path | Contents |
|---|---|
| `src/` | NeuroFormer model, LaBraM/OcuFM + CIDA adapters, EEG/eye feature extraction, fusion, calibration, fine-tuning, evaluation |
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
| `src/cida.py` | CIDA: connectivity-aware, identity-disentangled adapter for a frozen FM |
| `src/ocufm.py` | OcuFM: proposed gaze foundation model (LaBraM recipe) |
| `src/multimodal_cida.py` | Two-FM screener: CIDA-on-LaBraM ⊕ CIDA-on-OcuFM → fusion |
| `src/extract_eeg_embeddings.py` | Frozen LaBraM EEG embeddings (QC, leak-free, batched) |
| `src/extract_eye_features.py` | Eye-tracking features from raw |
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
