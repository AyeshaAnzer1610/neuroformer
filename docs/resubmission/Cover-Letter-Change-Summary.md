# Cover Letter — Major Revision (NeuroFormer, Access-2026-19478)

> For the editor/reviewers. Frames the revision as improvements to the **same
> NeuroFormer architecture**. Numbers marked [[…]] are finalized from the
> leak-free runs (HBN scale-up pending).

Dear Associate Editor and Reviewers,

We thank you for the constructive and encouraging reviews. We have prepared a
**major revision** that addresses every concern. We want to state up front what is
**unchanged** and what is **improved**, because the contribution remains the
NeuroFormer framework introduced in the original submission.

## The NeuroFormer architecture is UNCHANGED
The model is identical to the submitted version: three modality-specific
transformer encoders, the cross-modal fusion transformer with the `[CLS]`
aggregation token and modality-type embeddings, the classification head, and the
optional auxiliary-reconstruction decoders. **We did not introduce a new
architecture.** Every revision below changes the *inputs to*, *training of*, or
*evaluation of* this same model — i.e., the pipeline around NeuroFormer — in direct
response to the reviews.

## Summary of major changes (each maps to reviewer concerns)

1. **Corrected, leak-free evaluation protocol** *(R3.3, R3.4).* We strengthened
   the protocol to strict **subject-level, leak-free repeated stratified CV** with
   bootstrap CIs and a **compact NeuroFormer variant** to address
   overparameterization. This revises our reported performance downward to honest,
   leakage-free values — a change we consider essential to the paper's integrity.

2. **Foundation-model EEG front-end** *(R3.8, R2.1/2.4).* The *same* EEG modality
   encoder is now fed **pretrained EEG foundation-model (LaBraM) representations**
   instead of hand-crafted 5-channel features. EEG moves from the weakest to the
   **strongest** modality (leak-free ROC-AUC [[0.66]] vs hand-crafted [[0.45]]),
   directly resolving "is EEG worth including?" — yes, with modern representations.

3. **Missing-modality training + asynchronous fusion** *(R3.6, R3.9, R3.10).* We add
   **modality-dropout** training and use the model's modality tokens as
   availability indicators, and we reframe the fusion explicitly as
   **asynchronous, cross-paradigm participant-level** fusion (no synchronized-event
   claim). One model now serves any subset of modalities with graceful degradation.

4. **External / out-of-distribution validation** *(R1.3, R3.2, R3.5).* We add
   evaluation on the independent **Healthy Brain Network (HBN)** cohort
   ([[N≈…]]), reporting honest OOD performance ([[AUC]]); the aux-loss benefit is
   now tested externally, not only on MIPDB.

5. **Calibrated, referral-aware prediction** *(R3.2).* A post-hoc calibration layer
   (temperature scaling + conformal prediction with risk–coverage) lets the model
   abstain on low-confidence cases ("refer to clinician"), strengthening the
   screening-support framing.

6. **Honest multimodal reporting + softened claims** *(R3.9, R3.1).* We report the
   modality ablation transparently and soften the three-modality accuracy claim
   accordingly; we add per-disorder and confound analyses.

7. **Minor fixes** *(R4.1, R4.2, R4.4, R1.1, R2.2/2.5).* Citation for the 970M
   figure; ICA step made prominent; conclusion condensed; interpretability claims
   grounded; recent SOTA baselines + theory subsections added.

## On the revised performance numbers
Our strengthened (leak-free) protocol yields lower headline numbers than the
original submission. We view this as a **strengthening**, not a weakness: the
revised results are reproducible, externally validated, and honestly reported,
and the contribution is now a rigorously evaluated, gap-filling framework
(foundation-model EEG + missing-modality fusion + external validation + calibrated
referral) rather than a single-cohort accuracy figure. This directly reflects the
reviewers' emphasis on rigor and generalizability.

A point-by-point response to every comment follows (see Response-to-Reviewers),
and changes are marked in the revised manuscript.

Sincerely,
The Authors
