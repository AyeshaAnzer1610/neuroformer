# Phase 5 Results — Training-regime ablation on foundation-model EEG (GPU)

**Question:** the frozen-embedding linear probe caps at ROC-AUC ≈0.66. Can we do
better by *using* the frozen LaBraM representation differently, or by fine-tuning
the backbone end-to-end? (The accuracy lever, addressing R3.8 / R3.4 / R3.3.)

## Setup
- **MIPDB**, transdiagnostic any-DX vs none, **N≈118**, all QC-passing recordings
  per subject (`--all-recordings`), 15 s / 3000-sample windows, frozen LaBraM
  (`InterpolatedLaBraM`, `braindecode/labram-pretrained`).
- **Window-level training**, **strict subject-level CV** (every window of a held-out
  subject excluded; no test-fold standardization); per-subject prediction = mean
  window probability; class-weighted loss; mixed precision (AMP).
- Run on Nibi **H100** (`scripts/finetune_sbatch.sh`, job 16145801; confirmation
  5×10 job 16164082). `src/finetune_labram.py`.

## Result — frozen-backbone window-level training is the winner

| Regime (frozen LaBraM) | CV | ROC-AUC |
|---|---|---|
| Subject-pooled linear probe (Phase 1) | 5×10 | 0.66 |
| **Window-level head training (MIL)** | **5×10** | **0.748 ± 0.029** |
| Window-level head training (MIL) — preliminary | 5×1 | 0.795 ± 0.041 |
| End-to-end fine-tuning (full backbone) | 5×1 | 0.686 ± 0.120 |

- **Window-level training (headline):** the well-powered **5×10** run gives **0.748 ±
  0.029** — a **+0.09** ROC-AUC gain over linear probing, leak-free, tight CI. (The
  preliminary 5×1 estimate of 0.795 ± 0.04 was optimistically high, as single-repeat
  CV is noisier; the 5×10 value is what we quote.) `results/finetune_labram_frozen_r10.csv`.
- **End-to-end fine-tuning**: folds 0.884 / 0.611 / 0.632 / 0.539 / 0.763 → **0.686 ±
  0.120** (5×1). Lower mean **and ~4× the fold-to-fold variance** — it overfits the
  ~5–6M-parameter backbone at N≈118.

## Interpretation (honest)
The gain over the 0.66 baseline comes from the **window-level multiple-instance
training regime** (≈3,700 window instances + all recordings, per-subject pooling),
not from unfreezing the backbone — full fine-tuning is *worse* and unstable. So the
recommended configuration is **frozen foundation model + lightweight window-level
head training**.

## What this means for the paper
- **Headline upgrade:** within-MIPDB foundation-model EEG reaches **ROC-AUC 0.748 ±
  0.029** (leak-free, subject-level, 5×10 CV) — a substantive, defensible result with
  a tight CI, not a fragile one (+0.09 over linear probing).
- **Answers reviewers directly:** the frozen-vs-fine-tuned ablation is a clean
  empirical answer to **R3.4** (overparameterization — added capacity hurts) and
  **R3.3** (fold sensitivity — full-backbone CI ±0.12 vs frozen ±0.04).
- The multimodal null (Phase 2) and the well-powered external HBN null (Phase 4) are
  **unchanged** and still reported candidly.

## Artifacts
`results/finetune_labram_frozen.csv` (5×1), `results/finetune_labram_full.csv` (5×1);
`results/finetune_labram_frozen_r10.csv` (5×10 confirmation, pending).
