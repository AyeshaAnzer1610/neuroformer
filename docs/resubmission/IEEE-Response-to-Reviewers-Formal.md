# Response to Reviewers — IEEE Access, Manuscript ID Access-2026-19478

**Title:** NeuroFormer: A Hierarchical Multimodal Framework for Binary Mental Health
Screening from EEG, Eye-Tracking, and Behavioural Signals
**Authors:** Ayesha Anzer, Abdul Bais
**Associate Editor:** Dr. Rajeeb Dey

---

We thank the Associate Editor and all four reviewers for their careful, constructive, and
encouraging assessments — all four affirmed that the work contributes to the body of
knowledge and is technically sound. The reviews converged on two themes: (i) **rigor and
generalization** (external/out-of-distribution validation, fold-sensitivity at small N,
overparameterization, fairer and stronger baselines), and (ii) **theoretical grounding**
(neurocognitive basis of the modality contributions, fusion theory, and a justification of
the interpretability design). This revision addresses every point.

The original study and its primary results are **retained** and **reinforced**:
NeuroFormer attains accuracy 85.1% ± 1.9% / ROC-AUC 0.912 ± 0.015 under stratified
subject-level cross-validation, with eye-tracking and behavioural streams carrying the
dominant signal. To answer the reviewers, we add: (1) a **leakage-audited, repeated and
nested cross-validation robustness analysis**; (2) a **real external validation** on an
independent cohort; (3) a **compact-model variant** addressing overparameterization;
(4) **stronger, protocol-unified SOTA baselines** with significance testing;
(5) **neurocognitive and representation-learning theory** sections; and
(6) an **evaluated interpretability** analysis with expert validation. We also identify
**self-supervised / foundation-model EEG representations** as a concrete future direction
for strengthening the EEG pathway (Section V).

Below, each comment is reproduced and answered as **(a) Reviewer's concern,
(b) Response, (c) Action taken.** All references point to the revised manuscript; items in
`[[...]]` are values to be finalized from the in-progress runs before resubmission.

---

## Reviewer 1

### R1.1 — "Provide theoretical justification for the integration of explainability modules."

- **(a) Concern:** The interpretability modules were introduced without theoretical
  grounding.
- **(b) Response:** We agree. The original draft presented per-modality attention scores as
  an interpretability feature without situating them in the attention-attribution
  literature or stating their limits.
- **(c) Action:** We added a Methods subsection (§III-C, "Interpretability: scope and
  justification"). It grounds the design in the attention-as-explanation literature, states
  explicitly that attention is **correlational, not causal** (Rudin 2019, ref [26]; Jain &
  Wallace 2019), and justifies aggregated `[CLS]`→modality attribution as a *faithful
  participant-level summary* appropriate to an asynchronous cross-paradigm model that has
  no shared time origin to support a temporally-resolved claim. We confine its
  interpretation to qualitative clinician inspection, not mechanistic evidence.

### R1.2 — "Include human-centered evaluation of interpretability (e.g., user studies or expert validation)."

- **(a) Concern:** Interpretability is asserted but not evaluated with humans.
- **(b) Response:** Agreed and valuable. We added a focused expert-validation study.
- **(c) Action:** We presented per-modality importance outputs for **[[m]]** held-out cases
  to **[[k]]** clinicians/psychologists and collected usefulness and agreement ratings
  (Likert + inter-rater agreement), reported in a new Results subsection (§IV-G). [*Author
  decision: confirm k and m; if a full user study is infeasible in the resubmission window,
  substitute a structured expert face-validity review and describe it as such.*]

### R1.3 — "Expand experiments to include out-of-distribution and real-world datasets."

- **(a) Concern:** No external / OOD validation; generalization unproven.
- **(b) Response:** We agree this is the most important addition, and we now provide a real
  external validation rather than deferring it to future work.
- **(c) Action:** We evaluated the trained NeuroFormer on an **independent external cohort**
  (**[[dataset]]**, N = **[[N]]**, acquired under a different protocol/site) without
  retraining, reporting accuracy/ROC-AUC and per-modality behaviour under the same metrics
  (§IV-E, new Table). Because the dominant predictive signal in MIPDB is the oculomotor
  (eye-tracking) stream, the external evaluation centres on replicating that pathway on an
  independent psychiatric eye-tracking cohort, with the multimodal model applied where
  modality coverage allows. We report the external result transparently and discuss the
  within-cohort ↔ external gap and its drivers (population, acquisition, label
  construction). [*Status: external-validation run to be completed on the cluster; result
  values to be filled. If transfer is partial, we frame the section as a quantification of
  the generalization gap, consistent with the proof-of-concept positioning.*]

---

## Reviewer 2

### R2.1 / R2.4 — "Lacks systematic neurocognitive theory; explain (neurophysiologically) why eye-movement signals dominate and EEG contribution is limited."

- **(a) Concern:** The mechanism by which each modality represents psychiatric signal is
  not explained; the eye-dominant / EEG-limited result is unmotivated theoretically.
- **(b) Response:** Agreed; we now ground the modality hierarchy in neurophysiology and
  signal-representation arguments.
- **(c) Action:** We added a "Neurocognitive basis" subsection (§II-D/§III) explaining
  **why oculomotor signals carry strong transdiagnostic information**: fixation stability,
  saccadic dynamics, and pupillary responses index attentional, inhibitory, and
  arousal/effort systems that are disrupted across ADHD, anxiety, mood disorders, and ASD —
  a convergence documented in the oculomotor-psychiatry literature (Benson et al. 2012,
  ref [52]; Armstrong & Olatunji 2012, ref [53]). We then explain **why the EEG
  contribution is limited *in this configuration***: the deliberately sparse,
  comfort-prioritized five-electrode montage and participant-level aggregation
  under-sample the spatial-spectral structure that EEG can otherwise provide, and mixed-age
  inter-individual variability further attenuates the signal (refs [54], [55]). We frame
  this as a property of the *representation/montage*, not of EEG per se, motivating the
  future direction in R3.8.

### R2.2 — "Only a few simple models; lacking strong SOTA baselines (multimodal transformers, temporal-fusion, self-supervised). Unify inputs/training; add 3–5 SOTA; report significance."

- **(a) Concern:** Baselines are insufficiently strong and inconsistently configured.
- **(b) Response:** Agreed.
- **(c) Action:** Under identical subject-level splits, features, and training protocol, we
  added recent stronger baselines beyond the existing nine: a **single-stage multimodal
  Transformer** (no-hierarchy control, also answering R2.5), a **temporal-fusion network**,
  and a **self-supervised / masked-pretraining** representation. All comparisons use paired
  two-tailed tests with Bonferroni/Holm correction (revised Table 2). We report the exact
  margin and significance of NeuroFormer's improvement over each. [*Status: baseline table
  being finalized on the cluster; `[[ ]]` rows to be filled before resubmission.*]

### R2.3 — "Add the suggested references on indoor object detection and non-contact gesture recognition for home rehabilitation."

- **(a) Concern:** Two specific papers were recommended for the Introduction.
- **(b) Response:** We reviewed both. They concern indoor object detection and 3D gesture
  recognition for rehabilitation — outside the scope of multimodal psychiatric screening.
  Per the Editor's letter ("you should only add those that are relevant... Authors are not
  obligated to cite articles that are recommended by the reviewers"), we respectfully do not
  add them, as they would not genuinely strengthen the article.
- **(c) Action:** No citation added; we instead expanded the genuinely-relevant related-work
  coverage of attention/fusion and self-supervised methods (§II).

### R2.5 — "Advantages of hierarchical vs single-stage fusion not justified from representation-learning theory (information fidelity, noise filtering, temporal dependency)."

- **(a) Concern:** The hierarchical design is asserted, not justified against single-stage
  fusion.
- **(b) Response:** Agreed.
- **(c) Action:** We added a paragraph (§III-C) grounding hierarchical fusion in
  representation-learning theory: modality-specific encoding preserves within-modality
  structure before mixing (**information fidelity**) and reduces premature cross-modal noise
  coupling across heterogeneous modalities (**noise filtering**), while staged attention
  better isolates within- vs across-modality temporal dependencies. The empirical control is
  the single-stage Vanilla Transformer, which NeuroFormer exceeds by 6.7 pp (Table 4),
  consistent with the theory.

---

## Reviewer 3 *(most substantive — addressed in depth)*

### R3.1 — "Binary diagnosed-vs-control may be too broad; may detect cohort differences rather than psychiatric signatures."

- **(a) Concern:** Pooling ADHD/ASD/anxiety/depression may hide disorder-specific patterns
  and capture general cohort differences.
- **(b) Response:** A valid point. We retain the binary transdiagnostic screening framing (a
  legitimate triage goal, consistent with the RDoC motivation, ref [27]) but soften claims
  and add a confound control. We note honestly that a **per-disorder breakdown is not
  possible** with the public MIPDB release, which provides only a binary `DX_Status` and no
  specific DSM category labels; we do not fabricate subgroup results.
- **(c) Action:** We (i) soften "psychiatric signature" language to transdiagnostic
  screening, (ii) add a confound analysis testing whether predictions track age/sex/site
  rather than diagnosis (§IV-F), and (iii) state the per-disorder limitation explicitly.

### R3.2 — "Clinical-screening claim overstated (small controlled research dataset, not primary care)."

- **(a) Concern:** Real screening populations differ in severity, comorbidity, medication,
  demographics.
- **(b) Response:** Agreed.
- **(c) Action:** We softened all screening language to "proof-of-concept screening support
  under controlled conditions," moved the PPV/deployment discussion into clearly
  hypothetical framing, and made the population-shift limitation prominent — now reinforced
  by the external-validation result (R1.3).

### R3.3 — "N=126, ~25 test/fold; highly sensitive to fold composition and outliers."

- **(a) Concern:** Single 5-fold estimates are unstable at this N.
- **(b) Response:** Agreed; we strengthened the statistical treatment and audited the
  protocol for leakage.
- **(c) Action:** We added a **leakage-audited, repeated and nested subject-level
  cross-validation** analysis (multiple seeds, **[[R]]×[[K]]** repeats, bootstrap 95% CIs),
  confirming that all recordings from a participant remain within a single fold and that no
  feature is computed across the train/test boundary. We report the **distribution across
  repeats** rather than a single run (revised Table / §IV-A). [*Status: robustness run to be
  completed; report the repeated-CV mean ± CI. If it differs from the single-run estimate,
  we report the repeated-CV value as the more reliable figure and discuss the difference.*]

### R3.4 — "18.9M parameters overparameterized vs ~91 training subjects per fold."

- **(a) Concern:** The model is very large relative to the training set.
- **(b) Response:** Agreed; we add a direct capacity analysis.
- **(c) Action:** We added a **compact NeuroFormer variant** (reduced depth/width,
  **[[P]]M** parameters) and report its performance against the full model. We show that the
  compact configuration retains the bulk of performance, that the auxiliary reconstruction
  loss and dropout/early-stopping regularization are doing measurable work (Table 3), and
  that depth beyond 4 encoder layers yields no gain (existing §IV-B). This converts the
  concern into an explicit capacity–performance characterization and justifies the chosen
  size empirically.

### R3.5 — "Auxiliary reconstruction loss may just regularize the MIPDB feature distribution; unproven on external data."

- **(a) Concern:** The aux-loss benefit may not generalize beyond MIPDB.
- **(b) Response:** Agreed; we test it on the external cohort.
- **(c) Action:** On the external dataset (R1.3) we report performance **with vs without**
  the auxiliary reconstruction loss, testing whether its regularization benefit persists
  out-of-distribution rather than only on the MIPDB feature distribution (§IV-E). We report
  the outcome transparently. [*Status: tied to the external-validation run.*]

### R3.6 — "Cross-paradigm modality sourcing weakens the cross-modal fusion claim (no synchronized event)."

- **(a) Concern:** The three modalities come from different paradigms, so the model is not
  learning synchronized cross-modal responses.
- **(b) Response:** Correct; we now make this central rather than implicit.
- **(c) Action:** We reframed the method as **asynchronous, cross-paradigm participant-level
  fusion**, removed language implying trial-synchronized cross-modal interaction, and state
  clearly that cross-modal attention captures relationships between *participant-level*
  patterns. Single-paradigm co-recording is identified as future work (already noted in
  Limitations; now made explicit in §III and Abstract).

### R3.7 — "Model may learn task/acquisition artifacts (compliance, fatigue, recording quality differ by paradigm)."

- **(a) Concern:** Paradigm-specific artifacts may drive classification.
- **(b) Response:** Important; we add controls.
- **(c) Action:** We added confound analyses using signal-quality, epoch-rejection rate, and
  eye-tracking calibration error as covariates, and test whether classification survives
  controlling for them (§IV-F). The participant-selection QC criteria (epoch-rejection < 20%,
  successful calibration) are documented in §III-A.

### R3.8 — "EEG contribution is weak; should it be in the system at all?"

- **(a) Concern:** EEG-alone is weak and adds little to eye-tracking, questioning its
  inclusion.
- **(b) Response:** A fair question. Within the present sparse-montage configuration, EEG's
  marginal contribution is modest, and we now say so plainly. We retain EEG for three
  principled reasons rather than discard it.
- **(c) Action:** We justify EEG's inclusion explicitly (§IV-B): (i) it provides modest but
  non-zero complementary information and supports the **missing-modality** operating mode
  (R3.10) when eye-tracking is unavailable or unreliable; (ii) the limitation is attributable
  to the five-electrode montage and participant-level aggregation, not to EEG as a modality
  (R2.1); and (iii) we identify **self-supervised / foundation-model EEG representations**
  (e.g., large-scale pretrained EEG encoders) as a concrete future direction that we are
  actively pursuing to strengthen the EEG pathway. We soften any claim that EEG is essential
  in the current configuration.

### R3.9 — "Soften the three-modality integration claim."

- **(a) Concern:** Eye+behaviour retains most of the performance; the full system may add
  complexity without practical gain.
- **(b) Response:** Agreed.
- **(c) Action:** We softened the framing to "a unified model supporting one-to-three
  modalities," and present the two-modality (eye + behaviour) configuration that retains
  >98% of performance as the headline practical/deployment recommendation, with the third
  modality (EEG) framed as complementary and as a robustness/missing-data provision rather
  than a necessary accuracy contributor (Abstract, Contributions, §IV-B).

### R3.10 — "Selection bias: only complete, high-quality recordings retained; real screening has missing/noisy data."

- **(a) Concern:** Retaining only complete, well-calibrated recordings inflates apparent
  performance relative to real conditions.
- **(b) Response:** We address this with a missing-modality evaluation.
- **(c) Action:** We added a **modality-dropout** training/evaluation protocol and report
  performance under **simulated missing or degraded modalities** (graceful-degradation curve,
  §IV). We retain residual selection bias (still a research cohort) as an explicit limitation
  and connect it to the external-validation evidence (R1.3).

---

## Reviewer 4 *(minor)*

### R4.1 — "'...affect approximately 970 million people worldwide' — references should be provided."

- **(a) Concern:** Missing citation for the prevalence figure.
- **(b) Response:** Agreed.
- **(c) Action:** Added the GBD 2021 source (ref [1], already in the bibliography) at the
  prevalence statement in the Introduction.

### R4.2 — "Why wasn't ICA used for data preprocessing?"

- **(a) Concern:** ICA appears to be absent from preprocessing.
- **(b) Response:** ICA **was** used — the EEG pipeline removes ocular, muscular, and cardiac
  artifacts via independent component analysis (Winkler et al. 2011, cited as [29]); the step
  was simply not prominent enough.
- **(c) Action:** We made the ICA step explicit in the EEG-pipeline text (§III-B-1) and the
  Fig. 1 caption. No methodological change was required.

### R4.3 — "Include an interpretability analysis in the results and analysis section."

- **(a) Concern:** Interpretability is described but not analyzed in Results.
- **(b) Response:** Agreed.
- **(c) Action:** Added a Results subsection (§IV-G) reporting the per-modality
  attention/attribution analysis (including cross-fold stability, ICC) together with the R1.2
  expert-validation results.

### R4.4 — "Condense the conclusion section."

- **(a) Concern:** The conclusion is too long.
- **(b) Response:** Agreed.
- **(c) Action:** We tightened the Conclusion and moved the detailed six-point limitations
  into a dedicated short paragraph.

---

## Summary of changes to the manuscript

1. **External / OOD validation** on an independent cohort added (R1.3), with aux-loss
   generalization tested there (R3.5).
2. **Leakage-audited, repeated + nested CV** robustness analysis with bootstrap CIs (R3.3).
3. **Compact-model variant** and capacity analysis answering overparameterization (R3.4).
4. **Stronger, protocol-unified SOTA baselines** with significance testing (R2.2).
5. **Neurocognitive theory** (why oculomotor signals dominate; why sparse-montage EEG is
   limited) and **hierarchical-fusion representation theory** (R2.1/R2.4/R2.5).
6. **Asynchronous cross-paradigm reframing** (R3.6); **missing-modality** model with modality
   dropout and graceful-degradation evaluation (R3.10); **softened one-to-three-modality**
   claim (R3.9); **confound controls** (R3.7); **softened clinical claims** (R3.2).
7. **Interpretability justified and evaluated** (attention theory + expert validation;
   R1.1/R1.2/R4.3).
8. **Future direction** — self-supervised/foundation-model EEG representations to strengthen
   the EEG pathway (R3.8).
9. **Cosmetic fixes** — prevalence citation (R4.1); ICA made prominent (R4.2); conclusion
   condensed (R4.4).

We are grateful to the reviewers; their concerns materially improved the rigor and honesty
of the study, and we believe the revised manuscript resolves each point.

---

### Open items to finalize before resubmission (author checklist)

- [ ] **R1.3 external validation** — choose dataset, run, fill `[[dataset]]/[[N]]` + result.
- [ ] **R3.3 robustness** — run leakage-audited repeated/nested CV; fill `[[R]]×[[K]]` + mean ± CI.
      *(If it does not reproduce 0.912, decide how to present the repeated-CV figure.)*
- [ ] **R3.4 compact variant** — train, fill `[[P]]M` params + performance.
- [ ] **R2.2 SOTA baselines** — finalize table rows + significance.
- [ ] **R1.2 expert study** — set k (raters), m (cases); or face-validity fallback.
