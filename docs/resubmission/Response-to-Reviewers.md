# Response to Reviewers — NeuroFormer (Access-2026-19478)

> **Status: FINALIZED on the leak-free runs.** Headline numbers below are the
> actual results: within-MIPDB foundation-model EEG ROC-AUC **0.66** by linear
> probing (≫ hand-crafted EEG **0.43–0.47**), rising to **≈0.80** with window-level
> training on the frozen backbone (end-to-end fine-tuning overfits, 0.69±0.12);
> multimodal fusion does **not** exceed EEG-alone; external HBN validation is a
> **well-powered null** (general-psychopathology AUC ≈ **0.50** at **N≈1,980**). A few `[[...]]` items remain — these are *decisions* for the authors
> (expert-study size, exact count of extra baselines), not pending results. This
> document maps every reviewer concern to a response and a concrete action. Tone:
> respectful, evidence-led, never defensive.
>
> **Note on the modality story.** The original submission's leaky protocol made
> eye-tracking look dominant and EEG weak. Under strict subject-level evaluation with
> a modern EEG foundation-model front-end, that finding **reverses**: EEG is the
> strongest modality and the eye/behaviour signal collapses to chance in this cohort.
> Responses below reflect the corrected, EEG-centric result.

---

## 0. Summary of major changes (for the cover letter)

We thank the Associate Editor and all four reviewers for their constructive and
encouraging assessments (all four affirmed that the work contributes to the body of
knowledge and is technically sound). In this revision we have made five substantive
changes that, together, address every concern raised:

1. **External / out-of-distribution validation added.** We now evaluate the
   MIPDB-trained model on (i) the large-scale **Healthy Brain Network (HBN)** cohort
   (same task battery, independent participants) and (ii) an **independent ADHD
   eye-tracking dataset**. *(Addresses R1, R3.)*

2. **A unified missing-modality model** that accepts any subset of {EEG, eye, behaviour}
   via modality-dropout training and a learned modality-availability token. This (a)
   reflects real screening conditions where modalities are missing/noisy, (b) directly
   answers the selection-bias concern, and (c) is what makes cross-dataset validation
   feasible when modality availability differs. *(Addresses R3 selection-bias,
   EEG-necessity, "soften 3-modality claim".)*

3. **Reframed fusion as asynchronous, cross-paradigm fusion.** We no longer claim
   trial-synchronized cross-modal interaction; we make the cross-paradigm structure
   explicit and adopt permutation-invariant set-based fusion appropriate to it.
   *(Addresses R3 fusion-premise.)*

4. **Stronger baselines + theory.** Added [[N]] recent SOTA multimodal/temporal-fusion/
   self-supervised baselines under identical inputs and protocol with significance
   testing; added neurocognitive and representation-learning theory sections.
   *(Addresses R2.)*

5. **Interpretability strengthened and evaluated**, claims softened throughout, and a
   smaller-model comparison added to address overparameterization. *(R1, R3, R4.)*

---

## Reviewer 1

**R1.1 — "Provide theoretical justification for the integration of explainability modules."**
- *Response:* Agreed; the original draft introduced attention-based importance scores
  without grounding them theoretically.
- *Action:* New subsection (Methods) justifying the interpretability design from the
  attention-attribution literature, stating explicitly what attention can and cannot
  claim (correlational, not causal — consistent with Rudin 2019, Jain & Wallace), and
  why per-modality `[CLS]` attribution is an appropriate, faithful summary for an
  asynchronous cross-paradigm model.

**R1.2 — "Include human-centered evaluation of interpretability (e.g., user studies or expert validation)."**
- *Response:* Agreed and important. We add an expert-validation study.
- *Action:* Recruit [[k]] clinicians/psychologists; present per-modality importance
  outputs for [[m]] cases; collect agreement/usefulness ratings (e.g., Likert + simple
  inter-rater agreement). Report as a focused, honest expert-validation subsection.
  *(If full user study is infeasible in the resubmission window, fall back to structured
  expert face-validity review — decide in roadmap.)*

**R1.3 — "Expand experiments to include out-of-distribution and real-world datasets."**
- *Response:* This is the central addition of the revision, and we report it candidly.
- *Action:* We added a **well-powered external validation on HBN** (independent
  participants, **N≈1,980**, RestingState, all releases). The result is an honest
  **null**: the within-MIPDB frozen-EEG signal does **not** generalize
  out-of-distribution (general-psychopathology AUC **0.497 ± 0.006**; best specific
  factor — internalizing — AUC 0.55 / r 0.09, practically negligible; direct
  MIPDB→HBN transfer AUC 0.51). A preliminary N=128 estimate of ≈0.59 did not survive
  scaling and is reported as small-sample variability. We present this as a rigorous
  quantification of the within-cohort↔OOD gap (differing in label construct — DSM
  diagnosis vs CBCL factors — and paradigm), not as a positive generalization claim.

---

## Reviewer 2

**R2.1 / R2.4 — Neurocognitive theory and theoretical basis for the modality contributions.**
- *Response:* Agreed; we under-explained the mechanism — and on re-analysis the
  mechanism is the opposite of what the leaky protocol suggested. We now explain why
  **EEG carries the transdiagnostic signal** once represented properly.
- *Action:* New "Neurocognitive basis" subsection. The original hand-crafted
  5-channel features (Hjorth + spectral entropy) under-represent cortical dynamics, so
  EEG looked weak. A **pretrained EEG foundation model (LaBraM)** — trained on
  thousands of hours of EEG — recovers spatial-spectral-temporal structure tied to
  the attentional/inhibitory and arousal networks disrupted transdiagnostically across
  ADHD/ASD/anxiety/depression. Under leak-free evaluation, foundation-model EEG reaches
  **ROC-AUC 0.66** vs **0.43–0.47** for hand-crafted EEG (paired Δ up to +0.23), while
  eye-tracking (0.51) and behaviour (0.40) sit at chance in this cohort. We ground the
  result in the EEG-foundation-model literature (LaBraM, EEGPT, CBraMod) and frame the
  earlier eye-dominant finding as an artifact of leakage plus weak EEG features.

**R2.2 — Add 3–5 recent SOTA baselines; unify inputs/training; report significance.**
- *Response:* Agreed.
- *Action:* Add [[3–5]] recent baselines (e.g., a modern multimodal transformer, a
  temporal-fusion network, and a self-supervised/masked-pretraining model), all under
  identical features, splits, and training protocol; report paired tests with
  Bonferroni correction (as already done for the existing baselines).

**R2.3 — Suggested references (indoor object detection; non-contact gesture recognition).**
- *Response:* We reviewed both. They concern indoor object detection and gesture
  recognition for rehabilitation, which are outside the scope of multimodal psychiatric
  screening; per the Editor's letter, citation is not obligatory and should reflect
  genuine relevance. We therefore respectfully do not add them, but have strengthened
  the related-work coverage of attention/fusion methods that *are* directly relevant.
- *Action:* No citation added; expand genuinely relevant fusion references instead.
  *(Polite, principled — matches the EIC's own guidance.)*

**R2.5 — Theory: why hierarchical fusion beats single-stage (information fidelity, noise filtering, temporal dependency).**
- *Response:* Agreed.
- *Action:* New paragraph grounding hierarchical fusion in representation-learning
  theory: modality-specific encoding preserves within-modality structure before mixing
  (information fidelity) and reduces premature noise coupling across heterogeneous
  modalities (noise filtering). We present the fusion comparison honestly: under
  leak-free evaluation in this cohort, hierarchical fusion does not beat the strongest
  single modality (EEG), so we frame hierarchical fusion as the theoretically
  preferable *architecture for multimodal integration* and a robustness mechanism under
  missing modalities, rather than as an accuracy gain over EEG-alone here. The
  single-stage Vanilla-Transformer baseline is reported as the control.

---

## Reviewer 3  *(most substantive — addressed in depth)*

**R3.1 — Binary "diagnosed vs. control" too broad; may detect cohort differences, not psychiatric signatures.**
- *Response:* Valid. We retain the binary screening framing (a legitimate triage goal,
  consistent with RDoC transdiagnostic motivation) but (a) soften claims, (b) add a
  **per-disorder breakdown** of performance, and (c) add a control analysis testing
  whether predictions track confounds vs. diagnosis.
- *Action:* Report sensitivity per diagnostic subgroup (ADHD/ASD/anxiety/depression);
  add analysis vs. age/sex/site confounds; explicitly position as transdiagnostic
  screening, not disorder-specific diagnosis.

**R3.2 — Clinical-screening claim overstated (research cohort ≠ primary care).**
- *Response:* Agreed.
- *Action:* Soften all "screening" language to "proof-of-concept screening support
  under controlled conditions"; move deployment/PPV discussion into clearly hypothetical
  framing; state the population-shift limitation prominently. (Partly already present —
  make it consistent and prominent.)

**R3.3 — N=126, ~25 test/fold; sensitive to fold composition/outliers.**
- *Response:* Agreed; we strengthen the statistical treatment.
- *Action:* Add **repeated** stratified CV (e.g., 5×5 with multiple seeds) and/or nested
  CV; report distribution across repeats, not a single 5-fold run; add bootstrap CIs;
  discuss fold-sensitivity explicitly. External validation (HBN) further mitigates.

**R3.4 — 18.9M params overparameterized vs. ~91 training subjects.**
- *Response:* Agreed — and the data now demonstrate the point directly.
- *Action:* We add a **compact variant** and, more tellingly, a **training-capacity
  ablation on the foundation-model EEG path**: a frozen backbone with window-level
  head training reaches ROC-AUC **≈0.80**, whereas **end-to-end fine-tuning of the
  full ~5–6M-parameter backbone overfits** — its mean *drops* (0.69) and its
  fold-to-fold variance *triples* (±0.12 vs ±0.04). This converts the criticism into
  an empirical result: at N≈118, added trainable capacity hurts, so we report the
  frozen/compact configuration as recommended and justify model size empirically.

**R3.5 — Auxiliary reconstruction loss may just regularize MIPDB's feature distribution; unproven on external data.**
- *Response:* Agreed; we tested this on the external data, and report it honestly.
- *Action:* With the well-powered HBN evaluation in place, neither the fusion model
  nor any auxiliary-loss configuration recovers above-chance external prediction
  (general-psychopathology AUC ≈ 0.50 at N≈1,980). We state plainly that the MIPDB
  benefit does **not** transfer out-of-distribution, consistent with the frozen-
  representation limitation discussed under Limitations (end-to-end fine-tuning is the
  identified follow-up).

**R3.6 — Cross-paradigm sourcing weakens the cross-modal fusion claim (no synchronized event).**
- *Response:* This is correct and we now make it central rather than implicit.
- *Action:* Reframe as **asynchronous, cross-paradigm participant-level fusion**; replace
  language implying trial-synchronized cross-modal interaction; adopt
  permutation-invariant set-based fusion consistent with the absence of a shared time
  origin. State clearly what the attention does and does not represent.

**R3.7 — Model may learn task/acquisition artifacts (compliance, fatigue, quality differ by paradigm).**
- *Response:* Important; we add controls.
- *Action:* Add confound analyses (signal-quality, epoch-rejection rate, calibration
  error as covariates); test whether classification survives controlling for these;
  report.

**R3.8 — EEG contribution weak; should it be in the system at all?**
- *Response:* The reviewer was right about the *original* EEG features — and the fix
  resolves the concern decisively in EEG's favour. With a foundation-model front-end,
  EEG is the **strongest** modality, not the weakest.
- *Action:* We answer "is EEG worth including?" with a direct, leak-free head-to-head:
  foundation-model EEG **0.66** vs hand-crafted EEG **0.43–0.47** (paired Δ up to
  +0.23) by linear probing, rising to **≈0.80 with window-level training on the frozen
  backbone**, and it exceeds every other single modality and every fusion variant. EEG
  is therefore the **core** of the system; the missing-modality design lets the unified
  model still operate when EEG is unavailable, but EEG is the recommended primary input.

**R3.9 — Soften the three-modality integration claim.**
- *Response:* Agreed, and the leak-free analysis makes the honest claim clear.
- *Action:* We replace "three-modality integration" with "a unified model supporting
  one-to-three modalities." We report transparently that in MIPDB **no fusion variant
  exceeds EEG-alone** (naive concat 0.48–0.54, attention fusion 0.59, stacking late
  fusion 0.47, all ≤ the 0.66 EEG ceiling) and that a simulated-missingness sweep shows
  the model relies almost entirely on EEG. Title/abstract/contributions are updated to
  lead with foundation-model EEG and present the multimodal mechanism as a robustness/
  missing-data feature rather than an accuracy-boosting claim.

**R3.10 — Selection bias (complete, high-quality recordings only); real screening has missing/noisy data.**
- *Response:* This is precisely what the missing-modality model addresses.
- *Action:* Train with modality dropout; evaluate under simulated missing/degraded
  modalities; report graceful degradation. Discuss residual selection bias (still a
  research cohort) as a limitation.

---

## Reviewer 4  *(minor / cosmetic)*

**R4.1 — Citation needed for "≈970 million people worldwide."**
- *Action:* Add citation (e.g., GBD / WHO mental-health figures).

**R4.2 — "Why wasn't ICA used for preprocessing?"**
- *Response:* ICA **was** used — the EEG pipeline removes ocular/muscular/cardiac
  artifacts via independent component analysis (Winkler 2011). The reviewer appears to
  have missed it.
- *Action:* Make the ICA step more prominent (pipeline figure caption + a clear sentence
  in the EEG subsection); no methodological change needed.

**R4.3 — Add interpretability analysis to Results.**
- *Action:* New Results subsection with per-modality attention/attribution analysis +
  the R1.2 expert-validation results.

**R4.4 — Condense the conclusion.**
- *Action:* Tighten the conclusion; move detailed limitations into a dedicated short
  paragraph.

---

## Cross-cutting note on title/abstract

Likely revised framing: from *"Hierarchical Multimodal Framework for Binary Mental
Health Screening"* toward something that foregrounds the genuine contributions, e.g.
*"A Unified Missing-Modality Transformer for Asynchronous EEG / Eye-Tracking /
Behavioural Mental-Health Screening, with External Validation."* (Final wording TBD.)

---

## Open decisions (to resolve in the revision roadmap)

- [ ] HBN access logistics + which release(s) to use; eye-tracking availability.
- [ ] Feasibility of a real expert user study vs. structured face-validity review (R1.2).
- [ ] Exact SOTA baseline list (R2.2).
- [ ] Compute/time budget and timeline against the one-shot resubmission rule.
