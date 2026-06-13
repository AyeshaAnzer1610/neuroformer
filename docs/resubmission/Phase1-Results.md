# Phase 1 Results — LaBraM vs hand-crafted EEG (the gate)

**Question (retires R3.8, R2.1/2.4):** do frozen foundation-model EEG embeddings
beat the old hand-crafted EEG features, and is EEG worth including at all?

## Setup
- **Cohort:** MIPDB, **N = 118** subjects (of 126; 8 have no `.raw` EEG — likely
  `.mat`-only / not downloaded — and 3 dirs are non-subjects). Binary task:
  **any diagnosis (DX_Status ∈ {1,2}) vs none (0)** → 23 positives / 118.
- **EEG features (both on identical recordings/windows):**
  - **LaBraM:** frozen `InterpolatedLaBraM`, per-15 s-window CLS embedding (200-d).
  - **Hand-crafted:** Hjorth (activity/mobility/complexity) + Welch spectral
    entropy per channel (516-d) — re-implements the original Multisource EEG set.
- **Preprocessing:** 0.1–75 Hz band-pass, 60 Hz notch, 200 Hz, volts→µV, clip ±800 µV.
  **QC recording selection:** per subject, accept the largest file whose signal is
  physiological (median |x| 3–150 µV, <15 % clipped) — drops impedance/flat files.
- **Eval:** per-subject window pooling → standardized logistic regression
  (class-balanced) → 5×10 repeated stratified CV, identical splits per feature set
  (paired). Metrics: ROC-AUC + balanced accuracy, 95 % CI.

## Result

| pool    | feature set  | ROC-AUC (95% CI) | bal-acc (95% CI) | dim  |
|---------|--------------|------------------|------------------|------|
| mean    | hand-crafted | 0.466 ± 0.040    | 0.497 ± 0.024    | 516  |
| mean    | **LaBraM**   | **0.634 ± 0.025**| 0.550 ± 0.030    | 200  |
| mean    | fused        | 0.537 ± 0.039    | 0.502 ± 0.027    | 716  |
| meanstd | hand-crafted | 0.429 ± 0.037    | 0.474 ± 0.017    | 1032 |
| meanstd | **LaBraM**   | **0.660 ± 0.031**| 0.532 ± 0.023    | 400  |
| meanstd | fused        | 0.484 ± 0.036    | 0.479 ± 0.022    | 1432 |

Paired AUC delta (LaBraM − hand-crafted): **+0.167** (mean), **+0.231** (mean+std);
both CIs exclude 0.

## Conclusions
1. **LaBraM significantly beats hand-crafted EEG.** Hand-crafted sits at/below
   chance (~0.43–0.47); the foundation model reaches ~0.66.
2. **mean+std pooling > mean** for LaBraM (0.634 → 0.660): cross-window variability
   is informative. (It *hurts* hand-crafted — that variance is noise there.)
3. **Naive concatenation fusion hurts** (0.48 < 0.66): adding uninformative
   hand-crafted features dilutes the signal under a single logreg → motivates the
   **learned / missing-modality fusion** of Phase 2 rather than concatenation.
4. **Honest caveat:** absolute AUC ~0.66 is modest — EEG-alone transdiagnostic
   screening is hard at N≈118. The story is the *method comparison* + the
   multimodal fusion to come, not a high single-modality number.

## Gate decision
**PASS** → proceed to Phase 2 (missing-modality fusion). Drop hand-crafted EEG as a
serious contender (keep only as a reported baseline). Use LaBraM mean+std embeddings
as the EEG pathway. Artifacts: `results/phase1_comparison.csv`,
`results/phase1_artifacts.tar.gz` (embeddings).
