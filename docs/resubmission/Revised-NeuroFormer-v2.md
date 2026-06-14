# NeuroFormer v2 — Revised manuscript plan (keep architecture, add SOTA novelty)

Same NeuroFormer hierarchical/missing-modality transformer; four novel
contributions that fill current literature gaps and answer every reviewer.
Numbers are the **leak-free, subject-level** values we measured; HBN figures
firm up with the multi-release scale-up (placeholders marked [[HBN]]).

## Revised title (keeps the name)
**NeuroFormer: A Unified Foundation-Model, Missing-Modality Transformer for
Transdiagnostic Mental-Health Screening, with External Validation.**

## Revised abstract (draft)
Psychiatric disorders are widely underdiagnosed, and existing screening relies on
subjective assessment. We present **NeuroFormer**, a unified transformer framework
for transdiagnostic mental-health screening that (i) replaces hand-crafted EEG with
**pretrained EEG foundation-model (LaBraM) representations**, (ii) fuses EEG,
eye-tracking, and behaviour under a **missing-modality regime** (modality dropout +
a learned availability token) so one model serves any subset, (iii) is **externally
validated** on the independent Healthy Brain Network (HBN) cohort, and (iv) provides
**calibrated, referral-aware** predictions for screening triage. Under strict
subject-level, leak-free cross-validation on MIPDB (N≈118), foundation-model EEG is
the dominant signal (ROC-AUC ≈0.66), substantially exceeding hand-crafted EEG
(≈0.45); a transparent ablation shows eye-tracking and behaviour add little in this
cohort, so we report the multimodal result honestly rather than overclaim. On HBN,
frozen NeuroFormer-EEG features predict the general psychopathology factor
out-of-distribution ([[HBN AUC]], r=[[HBN r]]), the first such external check in this
setting. We position NeuroFormer as a rigorously validated proof-of-concept whose
novelty is the *first* unification of EEG foundation models, missing-modality fusion,
external validation, and calibrated referral for transdiagnostic screening.

## Four contributions (each fills a gap + answers reviewers)
1. **Foundation-model EEG in a multimodal psychiatric transformer** (Gap 2). LaBraM
   embeddings replace the hand-crafted 5-channel features; EEG goes from the weakest
   to the strongest modality (0.45→0.66). *Answers R3.8, R2.1/2.4.*
2. **Missing-modality NeuroFormer** (Gap 1). Modality dropout + availability token +
   set-based asynchronous fusion → one model for any subset, with graceful
   degradation. *Answers R3.6, R3.9, R3.10.*
3. **External / OOD validation on HBN** (field-wide gap). Honest cross-cohort
   generalization of the FM-EEG signal. *Answers R1.3, R3.2, R3.5.*
4. **Calibrated, referral-aware screening** (Gap 3 — NEW). Temperature scaling /
   conformal prediction + risk–coverage and ECE; abstain → "refer to clinician."
   *Strengthens clinical framing; novel in this space.*
Plus a **leak-free subject-level protocol** with repeated CV + bootstrap CIs.
*Answers R3.3, R3.4.*

## Reviewer responses that CHANGE vs the old draft (because of real results)
- **R3.8 (is EEG worth including?)** OLD: "eye dominates, EEG weak, drop it." NEW:
  "with a foundation model, **EEG is the core signal** — that's contribution #1."
- **R2.1/2.4 (why eye dominates / theory)** OLD: oculomotor-dominance rationale.
  NEW: explain FM-EEG transdiagnostic signal; eye/behaviour honestly ≈ chance here.
- **R3.9 (soften 3-modality claim)** Now backed by a rigorous null: fusion doesn't
  beat EEG-alone; we state it plainly.
- All other R-responses in `Response-to-Reviewers.md` stay, with `[[XX]]`
  placeholders filled by Phase 1/2/4 numbers.

## What's SOTA here (honest)
Not a leaderboard accuracy. The novelty is the **combination no prior psychiatric
EEG/eye/behaviour paper has**: foundation-model EEG + missing-modality fusion +
external OOD validation + calibrated referral, under leak-free evaluation. That is a
defensible, gap-filling contribution for IEEE Access.

## Build/Write order
1. (running) HBN multi-release scale-up → firms up contribution #3.
2. **Calibration module** (`src/calibrate.py`): temperature scaling + conformal +
   risk–coverage/ECE on the existing CV folds → contribution #4.
3. Fill `Response-to-Reviewers.md` placeholders with real numbers; flip R2.1/R3.8.
4. Rewrite manuscript Results/Abstract/Contributions from Phase 1/2/4 + calibration.
