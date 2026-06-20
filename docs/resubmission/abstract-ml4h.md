# Abstract draft — ML4H (Machine Learning for Health) Findings

**Target:** ML4H Findings track (welcomes rigorous empirical + negative results).
Backup venues: IEEE EMBC, NeurIPS/ICLR foundation-model or clinical-ML workshop.

## Title (options)
1. **Freeze, Don't Adapt: Frozen EEG Foundation Models Outperform Fine-Tuning, PEFT,
   and Continued Pretraining for Small-Sample Transdiagnostic Screening**
2. When Not to Adapt: A Rigorous Study of EEG Foundation-Model Adaptation for
   Small-Cohort Psychiatric Screening

## Abstract (~220 words)
EEG foundation models (e.g., LaBraM) are increasingly adapted to clinical tasks by
fine-tuning, but clinical cohorts are small and the right adaptation strategy is
unclear. We rigorously evaluate adaptation strategies for **transdiagnostic psychiatric
screening** (any-DSM-diagnosis vs none) on the MIPDB cohort (N≈118), under strict
**subject-level, leakage-free repeated cross-validation** with confidence intervals.

Using a frozen LaBraM backbone as the reference, we compare full fine-tuning,
parameter-efficient fine-tuning (LoRA), and self-supervised contrastive continued
pretraining on unlabeled in-domain EEG. **The frozen pretrained representation with a
trained head is best (ROC-AUC 0.74 ± 0.05); every adaptation strategy degrades it** —
full fine-tuning 0.69, LoRA 0.70, contrastive continued pretraining 0.68. At this
sample size, adaptation adds variance and overfits, providing a clear "when-not-to-
adapt" guideline and directly addressing over-parameterization concerns.

We further test **external generalization** on the Healthy Brain Network cohort
(~2,000 subjects): cross-cohort transfer is at chance (AUC 0.52). An identity-
disentangling adapter removes subject identity from the representation at no accuracy
cost, yet does **not** improve transfer — isolating the barrier as **domain and label
shift, not subject-identity confound**. Finally, we propose OcuFM, a gaze foundation
model on the LaBraM recipe, to close the eye-tracking modality gap (future work).

Our contribution is a rigorous, leakage-free, externally-validated account of EEG
foundation-model adaptation in the low-data clinical regime.

## Contributions (for intro)
1. A matched, leak-free comparison showing **frozen > full-FT > LoRA ≈ SSL-continued-
   pretraining** for small-cohort EEG-FM screening — adaptation uniformly hurts.
2. An external-validation + **transfer-failure analysis** (HBN, ~2k) localizing the
   barrier to domain/label shift, not identity, via an identity-disentangling adapter.
3. **OcuFM**, a proposed gaze foundation model (LaBraM recipe), as a concrete future
   direction for the unaddressed eye-tracking modality.

## Honest scope (state in paper, do not overclaim)
- One primary cohort (N≈118); ~20% positive; single montage family.
- Results are **negative for adaptation**; the value is rigor + practical guidance.
- OcuFM is proposed/runnable, **not pretrained** — no accuracy claimed.
- Not a SOTA/leaderboard paper.
