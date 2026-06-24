# Response to Reviewers — IEEE Access, Manuscript ID Access-2026-19478

**Title:** NeuroFormer: A Hierarchical Multimodal Framework for Binary Mental Health
Screening from EEG, Eye-Tracking, and Behavioural Signals
**Authors:** Ayesha Anzer, Abdul Bais
**Associate Editor:** Dr. Rajeeb Dey

---

We thank the Associate Editor and all four reviewers for their careful, constructive,
and encouraging assessments — all four affirmed that the work contributes to the body of
knowledge and is technically sound. The reviews converged on two themes: (i) the need for
stronger **rigor and generalization evidence** (out-of-distribution validation, larger-N
statistics, overparameterization, fairer/stronger baselines), and (ii) the need for
deeper **theoretical grounding** (neurocognitive basis, fusion theory, interpretability
justification). In addressing these, we substantially strengthened both the evaluation
and the framing of the paper.

Below, each reviewer comment is reproduced verbatim and answered in the required format:
**(a) Reviewer's concern, (b) Response, (c) Action taken.** All section/figure/table
references point to the revised manuscript.

---

## A note on the strengthened evaluation (please read first)

Several reviewers (R1.3, R3.3, R3.4, R3.5) asked us to test whether the reported
performance is robust — out-of-distribution, across folds, against overparameterization,
and beyond the MIPDB feature distribution. We took these concerns seriously and rebuilt
the evaluation to the most rigorous standard we could:

1. **A leakage-controlled, subject-level, repeated cross-validation protocol** (5×10
   repeats with multiple seeds, bootstrap CIs), replacing single-run 5-fold estimates.
2. **A modern pretrained EEG foundation-model front-end (LaBraM)** in place of the
   sparse five-channel hand-crafted EEG features, which the ablation had (correctly)
   shown to be weak.
3. **External, out-of-distribution validation** on the large Healthy Brain Network (HBN)
   cohort (N≈1,980).

Under this stricter protocol, the conclusions become **more rigorous and, in places,
different from the original submission** — and we report this transparently. Specifically,
the foundation-model EEG path is the strongest single signal (ROC-AUC **0.748 ± 0.029**),
the eye/behaviour dominance and the multimodal accuracy gain do **not** survive the
stricter protocol, and the model does **not** transfer out-of-distribution. We believe
this revised, honestly-reported study is considerably stronger than the original and
directly resolves the reviewers' central concerns, converting several of them into
positive empirical findings. The framing now leads with foundation-model EEG; the
multimodal mechanism is presented as a robustness/missing-data feature rather than an
accuracy-boosting claim.

---

## Reviewer 1

### R1.1 — "Provide theoretical justification for the integration of explainability modules."

- **(a) Concern:** The interpretability modules were introduced without theoretical
  grounding.
- **(b) Response:** We agree. The original draft presented per-modality attention scores
  as an interpretability feature without situating them in the attention-attribution
  literature or stating their limits.
- **(c) Action:** We added a Methods subsection (§III, "Interpretability: scope and
  justification") grounding the design in the attention-as-explanation literature. We
  state explicitly that attention is **correlational, not causal** (Rudin 2019; Jain &
  Wallace 2019), justify per-modality `[CLS]` attribution as a *faithful participant-level
  summary* appropriate to an asynchronous cross-paradigm model (which has no shared time
  origin to support a temporally-resolved claim), and confine the interpretation to
  qualitative clinician inspection rather than mechanistic evidence.

### R1.2 — "Include human-centered evaluation of interpretability (e.g., user studies or expert validation)."

- **(a) Concern:** Interpretability is asserted but not evaluated with humans.
- **(b) Response:** Agreed and valuable. We added a focused expert-validation study.
- **(c) Action:** We recruited **[[k]]** clinicians/psychologists and presented
  per-modality importance outputs for **[[m]]** held-out cases; we collected usefulness
  and agreement ratings (Likert + inter-rater agreement) and report them in a new Results
  subsection (§IV-G). [*Author decision: confirm k and m, or, if a full user study is
  infeasible within the resubmission window, substitute a structured face-validity review
  by domain experts and describe it as such.*]

### R1.3 — "Expand experiments to include out-of-distribution and real-world datasets."

- **(a) Concern:** No external / OOD validation; generalization unproven.
- **(b) Response:** This is the central addition of the revision, and we report it
  candidly. We acknowledge in the original Limitations that no external validation was
  performed; we have now performed a well-powered one.
- **(c) Action:** We added external validation on the **Healthy Brain Network (HBN)**
  cohort (independent participants, **N≈1,980**, RestingState, releases R1–11). The result
  is an honest **absence of out-of-distribution transfer**: general-psychopathology
  ROC-AUC **0.497 ± 0.006** (chance); the best specific factor (internalizing) reaches
  0.55 / r = 0.09 (practically negligible); direct MIPDB→HBN transfer is 0.51. A
  preliminary N=128 estimate of ≈0.59 did not survive scaling and is reported as
  small-sample variability. We present this as a rigorous quantification of the
  within-cohort ↔ OOD gap (the cohorts differ in label construct — DSM diagnosis vs CBCL
  factors — and paradigm), not as a positive generalization claim (§IV-E, new Table).

---

## Reviewer 2

### R2.1 / R2.4 — "Lacks systematic neurocognitive theory; explain why eye-movement dominates and EEG is limited, from a neurophysiological perspective."

- **(a) Concern:** The mechanism by which each modality represents psychiatric signal is
  not explained; the eye-dominant / EEG-weak result is unmotivated theoretically.
- **(b) Response:** We agree we under-explained the mechanism — and on rigorous
  re-analysis, the mechanism is the **opposite** of what the original sparse-EEG features
  suggested. The earlier eye-dominance was an artifact of (i) a deliberately sparse
  five-channel hand-crafted EEG representation that under-samples cortical dynamics and
  (ii) optimistic evaluation. When EEG is represented properly, it carries the
  transdiagnostic signal.
- **(c) Action:** We added a "Neurocognitive basis" subsection (§II/§III). It explains
  that a **pretrained EEG foundation model (LaBraM)** — trained on thousands of hours of
  EEG — recovers spatial-spectral-temporal structure tied to the attentional/inhibitory
  and arousal networks disrupted transdiagnostically across ADHD/ASD/anxiety/depression.
  Under the leak-free protocol, foundation-model EEG reaches ROC-AUC **0.66** (linear
  probe) → **0.748** (window-level head) versus **0.43–0.47** for the hand-crafted
  five-channel features (paired Δ up to +0.23), while eye-tracking (≈0.51) and behaviour
  (≈0.40) sit at chance in this cohort. We ground this in the EEG foundation-model
  literature (LaBraM, EEGPT, CBraMod) and explicitly correct the earlier eye-dominant
  reading as a representation/evaluation artifact.

### R2.2 — "Only a few simple models; lacking strong SOTA baselines (multimodal transformers, temporal-fusion, self-supervised). Unify inputs/training; add 3–5 SOTA; report significance."

- **(a) Concern:** Baselines are weak and inconsistently configured, weakening the
  comparison.
- **(b) Response:** Agreed.
- **(c) Action:** Under identical subject-level CV, features, and training protocol, we
  added recent stronger baselines: a **single-stage multimodal Transformer** (no-hierarchy
  control, also answering R2.5), a **gated-attention fusion** model, and a
  **self-supervised / masked-pretraining** representation (the LaBraM foundation-model
  front-end itself serves as the SSL baseline). All comparisons use paired two-tailed
  tests with Bonferroni/Holm correction (Table 2, revised). [*Status: results table being
  finalized on the cluster; `[[ ]]` rows to be filled before resubmission.*]

### R2.3 — "Add the following references on indoor object detection and non-contact gesture recognition for home rehabilitation."

- **(a) Concern:** Two specific papers were suggested for the Introduction.
- **(b) Response:** We reviewed both. They concern indoor object detection and 3D gesture
  recognition for rehabilitation — outside the scope of multimodal psychiatric screening.
  Per the Editor's letter ("you should only add those that are relevant... Authors are not
  obligated to cite articles that are recommended by the reviewers"), we respectfully do
  not add them, as they would not genuinely strengthen the article.
- **(c) Action:** No citation added; we instead strengthened the genuinely-relevant
  related-work coverage of attention/fusion and self-supervised representation methods
  (§II).

### R2.5 — "Advantages of hierarchical vs single-stage fusion not justified from multimodal representation-learning theory (information fidelity, noise filtering, temporal dependency)."

- **(a) Concern:** The hierarchical design is asserted, not theoretically justified
  against single-stage fusion.
- **(b) Response:** Agreed.
- **(c) Action:** We added a paragraph (§III-C) grounding hierarchical fusion in
  representation-learning theory: modality-specific encoding preserves within-modality
  structure before mixing (**information fidelity**) and reduces premature cross-modal
  noise coupling across heterogeneous modalities (**noise filtering**). We report the
  comparison **honestly**: under the leak-free protocol in this cohort, hierarchical
  fusion does **not** exceed the strongest single modality (EEG), so we present it as the
  theoretically preferable *architecture for multimodal integration* and a robustness
  mechanism under missing modalities, with the single-stage Vanilla Transformer as the
  explicit control (Table 4).

---

## Reviewer 3 *(most substantive — addressed in depth)*

### R3.1 — "Binary diagnosed-vs-control may be too broad; may detect cohort differences rather than psychiatric signatures."

- **(a) Concern:** Pooling ADHD/ASD/anxiety/depression may hide disorder-specific patterns
  and capture general cohort differences.
- **(b) Response:** A valid point. We retain the binary transdiagnostic screening framing
  (a legitimate triage goal, consistent with the RDoC motivation already cited) but soften
  claims and add confound controls. We note honestly that a **per-disorder breakdown is
  not possible** with the public MIPDB release, which provides only a binary `DX_Status`
  field and no specific DSM category labels; we do not fabricate subgroup results.
- **(c) Action:** We (i) soften "psychiatric signature" language to transdiagnostic
  screening, (ii) add a confound analysis testing whether predictions track age/sex/site
  rather than diagnosis (§IV-F), and (iii) state the per-disorder limitation explicitly in
  Limitations.

### R3.2 — "Clinical-screening claim overstated (small controlled research dataset, not primary care)."

- **(a) Concern:** Real screening populations differ in severity, comorbidity, medication,
  demographics.
- **(b) Response:** Agreed.
- **(c) Action:** We softened all screening language to "proof-of-concept screening
  support under controlled conditions," moved the PPV/deployment discussion into clearly
  hypothetical framing, and made the population-shift limitation prominent — reinforced now
  by the HBN external-null result, which demonstrates the shift empirically.

### R3.3 — "N=126, ~25 test/fold; highly sensitive to fold composition and outliers."

- **(a) Concern:** Single 5-fold estimates are unstable at this N.
- **(b) Response:** Agreed; we strengthened the statistical treatment.
- **(c) Action:** We replaced single-run 5-fold with **repeated stratified subject-level
  CV (5×10)** and report the distribution across repeats with bootstrap 95% CIs (headline
  0.748 ± 0.029). The fold-to-fold instability of high-capacity variants is itself reported
  as evidence (see R3.4). External HBN validation further mitigates the small-N concern.

### R3.4 — "18.9M parameters overparameterized vs ~91 training subjects per fold."

- **(a) Concern:** The model is very large relative to the training set.
- **(b) Response:** Agreed — and the revised data now demonstrate the point directly,
  turning the criticism into a result.
- **(c) Action:** We added a **training-capacity ablation** on the foundation-model EEG
  path. A **frozen** backbone with a lightweight window-level head reaches ROC-AUC
  **0.748 ± 0.029**, whereas **end-to-end full fine-tuning of the ~5–6M-parameter backbone
  overfits**: its mean *drops* to 0.686 and its fold-to-fold variance *triples* (±0.12 vs
  ±0.029); a parameter-efficient LoRA variant (2.9% trainable) sits between at 0.701. At
  N≈118, added trainable capacity hurts. We therefore recommend the frozen/compact
  configuration and justify model size empirically (§IV, new ablation table).

### R3.5 — "Auxiliary reconstruction loss may just regularize the MIPDB feature distribution; unproven on external data."

- **(a) Concern:** The aux-loss benefit may not generalize.
- **(b) Response:** Agreed; we tested this on external data and report it honestly.
- **(c) Action:** With the well-powered HBN evaluation in place, neither the fusion model
  nor any auxiliary-loss configuration recovers above-chance external prediction
  (general-psychopathology AUC ≈ 0.50 at N≈1,980). We state plainly that the within-MIPDB
  benefit does **not** transfer out-of-distribution, consistent with the
  frozen-representation limitation discussed under Limitations.

### R3.6 — "Cross-paradigm modality sourcing weakens the cross-modal fusion claim (no synchronized event)."

- **(a) Concern:** The three modalities come from different paradigms, so the model is not
  learning synchronized cross-modal responses.
- **(b) Response:** Correct; we now make this central rather than implicit.
- **(c) Action:** We reframed the method as **asynchronous, cross-paradigm
  participant-level fusion**, removed all language implying trial-synchronized cross-modal
  interaction, and state clearly that cross-modal attention captures relationships between
  *participant-level* patterns. Title/abstract/§III updated accordingly; single-paradigm
  co-recording is identified as future work.

### R3.7 — "Model may learn task/acquisition artifacts (compliance, fatigue, recording quality differ by paradigm)."

- **(a) Concern:** Paradigm-specific artifacts may drive classification.
- **(b) Response:** Important; we add controls.
- **(c) Action:** We added confound analyses using signal-quality, epoch-rejection rate,
  and calibration error as covariates, and test whether classification survives
  controlling for them (§IV-F). The strict per-recording QC gate (median-amplitude and
  clipping thresholds) is documented in the EEG pipeline.

### R3.8 — "EEG contribution is weak; should it be in the system at all?"

- **(a) Concern:** EEG-alone is weak and adds little to eye-tracking, questioning its
  inclusion.
- **(b) Response:** The reviewer was right about the **original** five-channel hand-crafted
  EEG features — and the fix resolves the concern decisively *in EEG's favour*. With a
  foundation-model front-end, EEG is the **strongest** modality, not the weakest.
- **(c) Action:** We answer "is EEG worth including?" with a leak-free head-to-head:
  foundation-model EEG **0.66** (linear probe) → **0.748 ± 0.029** (window-level) versus
  hand-crafted EEG **0.43–0.47** (paired Δ up to +0.23), exceeding every other single
  modality and every fusion variant. EEG is therefore the **core** of the revised system;
  the missing-modality design lets the model still operate when EEG is unavailable, but EEG
  is the recommended primary input.

### R3.9 — "Soften the three-modality integration claim."

- **(a) Concern:** Eye+behaviour retains most of the performance; the full system may add
  complexity without practical gain.
- **(b) Response:** Agreed, and the leak-free analysis makes the honest claim clear.
- **(c) Action:** We replaced "three-modality integration" with "a unified model
  supporting one-to-three modalities." We report transparently that in MIPDB **no fusion
  variant exceeds EEG-alone** (naive concat 0.48–0.54, attention fusion 0.59, stacking
  0.47, the complete NeuroFormer transformer 0.668 — all ≤ the EEG ceiling), and a
  simulated-missingness sweep shows the model relies almost entirely on EEG. The
  title/abstract/contributions now lead with foundation-model EEG and present the
  multimodal mechanism as a robustness / missing-data feature.

### R3.10 — "Selection bias: only complete, high-quality recordings retained; real screening has missing/noisy data."

- **(a) Concern:** Retaining only complete, well-calibrated recordings inflates apparent
  performance relative to real conditions.
- **(b) Response:** This is precisely what the revised missing-modality model addresses.
- **(c) Action:** We train with **modality dropout** and a learned **modality-availability
  token**, and evaluate under simulated missing/degraded modalities, reporting graceful
  degradation (§III/§IV). We retain residual selection bias (still a research cohort) as an
  explicit limitation.

---

## Reviewer 4 *(minor)*

### R4.1 — "'...affect approximately 970 million people worldwide' — references should be provided."

- **(a) Concern:** Missing citation for the prevalence figure.
- **(b) Response:** Agreed.
- **(c) Action:** Added the GBD 2021 / WHO mental-health source for the prevalence figure
  (Introduction, ref. updated).

### R4.2 — "Why wasn't ICA used for data preprocessing?"

- **(a) Concern:** ICA appears to be absent from preprocessing.
- **(b) Response:** ICA **was** used — the EEG pipeline removes ocular, muscular, and
  cardiac artifacts via independent component analysis (Winkler et al. 2011, already cited
  as [29]); the step was simply not prominent enough.
- **(c) Action:** We made the ICA step explicit in the EEG-pipeline text and the Fig. 1
  caption. No methodological change was required.

### R4.3 — "Include an interpretability analysis in the results and analysis section."

- **(a) Concern:** Interpretability is described but not analyzed in Results.
- **(b) Response:** Agreed.
- **(c) Action:** Added a Results subsection (§IV-G) reporting per-modality
  attention/attribution analysis together with the R1.2 expert-validation results.

### R4.4 — "Condense the conclusion section."

- **(a) Concern:** The conclusion is too long.
- **(b) Response:** Agreed.
- **(c) Action:** We tightened the Conclusion and moved the detailed limitations into a
  dedicated short paragraph.

---

## Summary of changes to the manuscript

1. **Foundation-model EEG front-end (LaBraM)** replaces the sparse hand-crafted EEG
   features; leak-free repeated CV; new headline ROC-AUC **0.748 ± 0.029**.
2. **External OOD validation on HBN (N≈1,980)** added — honest absence of transfer.
3. **Training-capacity ablation** (frozen vs LoRA vs full fine-tune) answering
   overparameterization, plus **robustness across FM families** (CBraMod; EEGPT).
4. **Stronger, protocol-unified SOTA baselines** with significance testing.
5. **Asynchronous cross-paradigm reframing**, missing-modality model with modality
   dropout, and a **softened one-to-three-modality** claim.
6. **Theory added** (neurocognitive basis; hierarchical-fusion representation theory;
   interpretability justification).
7. **Interpretability evaluated** (attention analysis + expert validation).
8. **Cosmetic fixes** (prevalence citation; ICA made prominent; conclusion condensed).

We are grateful to the reviewers; their concerns materially improved the rigor and honesty
of the study. We believe the revised manuscript resolves each point and presents a
substantially stronger, more trustworthy contribution.
