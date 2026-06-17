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

2. **Foundation-model EEG front-end** *(R3.8, R2.1/2.4, R3.4, R3.3).* The *same* EEG
   modality encoder is now fed **pretrained EEG foundation-model (LaBraM)
   representations** instead of hand-crafted 5-channel features. EEG moves from the
   weakest to the **strongest** modality (leak-free ROC-AUC 0.66 vs hand-crafted
   0.45). We further show the representation is best used via **window-level training
   on the frozen backbone (ROC-AUC 0.748 ± 0.029, 5×10 CV)**; **end-to-end fine-tuning
   overfits** at N≈118 (0.69, ~4× the fold variance), which directly addresses the
   overparameterisation and fold-sensitivity concerns — added capacity hurts here.
   This resolves "is EEG worth including?" — emphatically yes, with modern
   representations used appropriately.

3. **Missing-modality training + asynchronous fusion** *(R3.6, R3.9, R3.10).* We add
   **modality-dropout** training and use the model's modality tokens as
   availability indicators, and we reframe the fusion explicitly as
   **asynchronous, cross-paradigm participant-level** fusion (no synchronized-event
   claim). One model now serves any subset of modalities with graceful degradation.
   We also propose a **best-available-representation** framework (foundation model per
   modality where one exists, self-supervised + cross-modal contrastive alignment where
   none does); it is implemented and reported candidly as a framework/future direction,
   without claiming an accuracy gain on this cohort.

4. **External / out-of-distribution validation** *(R1.3, R3.2, R3.5).* We add a
   **well-powered** evaluation on the independent **Healthy Brain Network (HBN)**
   cohort (**N≈1,980**, RestingState, all releases). We report the honest result:
   the within-MIPDB EEG signal **does not generalize** out-of-distribution under
   this stringent test (general-psychopathology AUC ≈ 0.50). We present this as a
   rigorous, candid quantification of the within-cohort↔OOD gap rather than
   overclaiming generalization.

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
Our strengthened (leak-free, subject-level) protocol yields lower, honest numbers
than the original submission, and our well-powered external test returns a
negative. We view this as a **strengthening of the work's integrity**, not a
weakness: within MIPDB, foundation-model EEG substantially beats hand-crafted EEG
(ROC-AUC 0.66 vs 0.45 by linear probing) and reaches **0.748 ± 0.029 (5×10 CV) with
window-level training on the frozen backbone**, while end-to-end fine-tuning overfits
at this sample size; the multimodal and external-generalization claims are reported
candidly (fusion does not exceed EEG-alone; the signal does not transfer to HBN at
N≈1,980). The contribution is a **rigorously evaluated proof-of-concept** —
foundation-model EEG (used appropriately), missing-modality design, and an honest
characterization of the within-cohort↔out-of-distribution gap — directly
addressing the reviewers' emphasis on rigor and generalizability rather than a
fragile single-cohort accuracy figure.

A point-by-point response to every comment follows (see Response-to-Reviewers),
and changes are marked in the revised manuscript.

Sincerely,
The Authors
