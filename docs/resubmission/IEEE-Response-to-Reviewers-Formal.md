# Response to Reviewers — IEEE Access, Manuscript ID Access-2026-19478

**Title:** NeuroFormer: A Hierarchical Multimodal Framework for Binary Mental Health
Screening from EEG, Eye-Tracking, and Behavioural Signals
**Authors:** Ayesha Anzer, Abdul Bais
**Associate Editor:** Dr. Rajeeb Dey

---

We thank the Associate Editor and all four reviewers for their careful and constructive
assessments — all four affirmed that the work contributes to the body of knowledge and is
technically sound. In preparing this revision we took the reviewers' rigor concerns
(R3.3 fold sensitivity, R3.4 overparameterization, R3.5 generalization, R2.2 baselines) as
the priority and **rebuilt the entire evaluation under a strict, leakage-controlled,
subject-level, repeated (5×5) stratified cross-validation protocol, with every number
produced by released, reproducible code.**

**Important note on the corrected results.** Under this rigorous protocol the headline
figures change, and we report them transparently: NeuroFormer attains **ROC-AUC
0.722 ± 0.037** (not the previously reported 0.912), still **exceeding every baseline**
(gated fusion 0.675, cross-modal Transformer 0.685, classical methods ≤0.59). The earlier
number reflected a window-level leakage in the prior protocol; the corrected, leak-free
evaluation is more conservative and fully reproducible. We have also corrected the cohort
description (the complete-case all-three-modality sample is **N≈102, ~20% positive**, an
*unbalanced* cohort — not 63/63), and we now report **balanced metrics (ROC-AUC, balanced
accuracy)** with class-weighted training. We believe the corrected manuscript is
substantially more rigorous and trustworthy, and that its findings directly resolve the
reviewers' central concerns.

Each comment is answered below as **(a) Concern, (b) Response, (c) Action.**

---

## Reviewer 1

**R1.1 — Theoretical justification for the explainability modules.**
- (a) Interpretability was introduced without grounding.
- (b) Agreed.
- (c) New Methods subsection grounding per-modality `[CLS]` attention attribution in the
  attention-as-explanation literature; we state explicitly that attention is **correlational,
  not causal** (Rudin 2019; Jain & Wallace 2019) and is reported as a participant-level
  summary appropriate to an asynchronous cross-paradigm model.

**R1.2 — Human-centered evaluation of interpretability (user/expert study).**
- (a) Interpretability is asserted but not evaluated with humans.
- (b) Agreed.
- (c) We add a focused expert-validation study: per-modality importance outputs for **[[m]]**
  held-out cases rated by **[[k]]** clinicians (Likert usefulness + inter-rater agreement),
  §IV-G. *[Author decision: confirm k/m, or use a structured face-validity review if a full
  study is infeasible in the window.]*

**R1.3 — Out-of-distribution / real-world datasets.**
- (a) No external validation.
- (b) We address this candidly. The **full three-modality model cannot be externally
  validated**, because no public cohort provides EEG + eye-tracking + behaviour under
  matching protocols (this is the dataset constraint we noted originally).
- (c) We (i) keep this as an explicit limitation, and (ii) add an external **EEG-only**
  transfer analysis on the large **Healthy Brain Network** cohort (N≈1,980), where even a
  strong pretrained EEG representation does **not** transfer (general-psychopathology ROC-AUC
  ≈ 0.50) — reported honestly as a bound on out-of-distribution generalization and attributed
  to the EEG analysis, not to NeuroFormer. (§V Limitations + Discussion.)

---

## Reviewer 2

**R2.1 / R2.4 — Neurocognitive theory; why modalities contribute as they do.**
- (a) The mechanism by which each modality carries psychiatric signal is unexplained.
- (b) Agreed — and under the corrected leak-free analysis the picture is clearer and more
  honest than the original eye-dominant claim: **no single modality is informative in
  isolation** (EEG, eye, behaviour each ≈ chance, ROC-AUC 0.53–0.55 alone); the
  transdiagnostic signal emerges only from **cross-modal fusion** (0.722).
- (c) New "Neurocognitive basis" subsection: oculomotor, cortical-oscillatory, and
  behavioural measures each index attentional/inhibitory and arousal systems disrupted
  transdiagnostically, but are individually weak in this small mixed cohort; the hierarchical
  cross-modal model integrates their complementary, individually-subthreshold information. We
  no longer claim eye-tracking "dominates."

**R2.2 — Stronger/SOTA baselines, unified protocol, significance.**
- (a) Only a few simple baselines; inconsistent conditions.
- (b) Agreed.
- (c) Corrected **Table 2** now compares, under one identical leak-free protocol and inputs:
  classical (LogReg, SVM-RBF, RandomForest), MLP early-fusion, late-fusion ensemble, **Gated
  Multimodal Unit**, and a **single-stage cross-modal Transformer** — against NeuroFormer,
  with **paired significance tests (Wilcoxon) and Holm correction**. NeuroFormer (0.722) is
  the top model. [[Add 1–2 further recent multimodal/temporal-fusion baselines if desired.]]

**R2.3 — Suggested references (indoor object detection; gesture recognition).**
- (a) Two specific papers recommended for the Introduction.
- (b) We reviewed both; they concern indoor object detection and 3D gesture recognition for
  rehabilitation — outside the scope of multimodal psychiatric screening. Per the Editor's
  guidance (authors need not cite recommended references that are not genuinely relevant), we
  respectfully do not add them.
- (c) No citation added; we strengthened the genuinely-relevant attention/fusion related work.

**R2.5 — Hierarchical vs single-stage fusion: representation-learning theory.**
- (a) The hierarchical design is asserted, not theoretically justified.
- (b) Agreed.
- (c) New paragraph (§III-C): modality-specific encoding preserves within-modality structure
  before mixing (information fidelity) and reduces premature cross-modal noise coupling (noise
  filtering). The **single-stage cross-modal Transformer (0.685) is the empirical control**,
  which NeuroFormer's hierarchical design exceeds — consistent with the theory.

---

## Reviewer 3

**R3.1 — Binary diagnosed-vs-control too broad.**
- (a) Pooling disorders may capture cohort differences, not psychiatric signatures.
- (b) Valid. We retain the transdiagnostic (RDoC-consistent) framing but soften claims and
  add confound controls. We note honestly that a **per-disorder breakdown is impossible** with
  the public MIPDB release (only a binary `DX_Status`; no DSM categories) — we do not
  fabricate subgroup results.
- (c) Soften "signature" language; add age/sex/site confound analysis (§IV-F); state the
  per-disorder limitation explicitly.

**R3.2 — Clinical-screening claim overstated.**
- (a) Research cohort ≠ primary care.
- (b) Agreed.
- (c) Soften throughout to "proof-of-concept screening support under controlled conditions";
  hypothetical framing for deployment/PPV; the population-shift limitation made prominent and
  reinforced by the HBN ≈ chance external evidence.

**R3.3 — N=126, ~25 test/fold; fold sensitivity.**
- (a) Highly sensitive to fold composition.
- (b) Agreed — and central to the revision.
- (c) The entire evaluation is now **repeated (5×5) subject-level stratified CV with bootstrap
  CIs** (corrected headline 0.722 ± 0.037), reporting the distribution across repeats rather
  than a single 5-fold pass. We also corrected the cohort (N≈102, 21 positive).

**R3.4 — 18.9M parameters overparameterized.**
- (a) Far too large for ~91 training subjects.
- (b) Agreed — and we acted on it directly.
- (c) The corrected model is a **compact configuration** (d_model 128, 2 encoder + 2 fusion
  layers; ~2–4M parameters, not 18.9M), chosen to match the cohort size. This both reduces
  overparameterization and is what is now reported in all tables. Model size is justified
  empirically (larger depth did not help at this N).

**R3.5 — Auxiliary reconstruction loss may not generalize.**
- (a) Aux-loss gain may be MIPDB-specific.
- (b) Agreed; we test it directly.
- (c) The corrected **Table 3** reports the modality ablation **with and without** the
  auxiliary loss under the leak-free protocol. **We report the honest outcome: [[fill from
  table3_ablation.csv]].** If the aux loss does not provide a reproducible gain, we **withdraw
  it as a contribution** and present it as an ablation that did not help — rather than restate
  the previous "+4.8 pp" claim.

**R3.6 — Cross-paradigm sourcing weakens the cross-modal claim.**
- (a) Modalities come from different paradigms; not synchronized responses.
- (b) Correct; we make this explicit.
- (c) Reframe as **asynchronous, cross-paradigm participant-level fusion**; remove language
  implying trial-synchronized interaction; state that attention captures participant-level
  relationships, not trial-locked dependencies. Single-paradigm co-recording = future work.

**R3.7 — Possible task/acquisition artifacts.**
- (a) Compliance/fatigue/quality differ by paradigm.
- (b) Important; we add controls.
- (c) Confound analysis using signal-quality, epoch-rejection rate, and calibration error as
  covariates; test whether classification survives controlling for them (§IV-F).

**R3.8 — EEG contribution weak; include it at all?**
- (a) EEG alone is weak; adds little to eye-tracking.
- (b) The corrected analysis reframes this honestly: **no single modality is informative
  alone** (EEG, eye, behaviour all ≈ chance) — so the question is not "EEG vs eye" but whether
  *fusion* beats the parts, which it does (0.722 vs ≤0.59 unimodal, ≤0.685 other fusion). EEG
  is retained because it contributes to the cross-modal signal and to missing-modality
  robustness; we do not claim it is individually sufficient.
- (c) Report the corrected modality ablation (Table 3); justify EEG by its contribution to the
  fused model, not in isolation.

**R3.9 — Soften the three-modality integration claim.**
- (a) Eye+behaviour may retain most performance.
- (b) Agreed.
- (c) We replace "three-modality integration" with "a unified model over one-to-three
  modalities," and report the corrected Table 3 subset results transparently, letting the
  ablation — not a strong claim — characterize each modality's marginal value.

**R3.10 — Selection bias (complete, high-quality recordings only).**
- (a) Real screening has missing/noisy data.
- (b) Addressed via a missing-modality design.
- (c) Modality-dropout training + a learned availability token; evaluation under simulated
  missing/degraded modalities (graceful-degradation curve). Residual selection bias (still a
  research cohort) retained as a limitation.

---

## Reviewer 4

**R4.1 — Citation for "≈970 million people worldwide."**
- (c) Added (GBD 2021 / WHO mental-health source).

**R4.2 — Why no ICA?**
- (b) ICA **was** used (ocular/muscular/cardiac removal, Winkler 2011); it was simply not
  prominent.
- (c) Made the ICA step explicit in §III-B-1 and the Fig. 1 caption. No methodological change.

**R4.3 — Interpretability analysis in Results.**
- (c) New Results subsection (§IV-G) with per-modality attention/attribution analysis +
  the R1.2 expert validation.

**R4.4 — Condense the conclusion.**
- (c) Conclusion tightened; detailed limitations moved to a dedicated short paragraph.

---

## Summary of major changes
1. **Leakage-controlled, repeated (5×5) subject-level evaluation**; corrected headline
   **ROC-AUC 0.722 ± 0.037** (NeuroFormer still beats all baselines). (R3.3)
2. **Compact model** (~2–4M params, not 18.9M). (R3.4)
3. **Stronger baselines + significance** under one protocol (Table 2). (R2.2)
4. **Modality ablation ± auxiliary loss** (Table 3); aux-loss claim reported honestly. (R3.5/R3.9)
5. **Corrected cohort** (N≈102, ~20% positive, unbalanced; balanced metrics). 
6. **Neurocognitive + fusion theory**; the honest "no single modality is informative alone,
   fusion extracts the signal" framing. (R2.1/R2.4/R2.5/R3.8)
7. **Asynchronous cross-paradigm reframe**, **missing-modality** model, **confound controls**,
   softened clinical claims. (R3.6/R3.7/R3.10/R3.1/R3.2)
8. **External boundary** (HBN EEG ≈ chance) as an honest limitation. (R1.3)
9. **Interpretability justified + evaluated** (R1.1/R1.2/R4.3); cosmetic fixes (R4.1/R4.2/R4.4).

We are grateful to the reviewers; their rigor concerns drove a more honest and reproducible
study, and we believe the corrected manuscript resolves each point.

### Open items to finalize before submission
- [ ] **Table 3 run** (`baselines_sbatch.sh`) → fill the modality ablation + the **aux-loss
      verdict** (keep the contribution only if real).
- [ ] **R1.2** — confirm expert-study size (k, m) or face-validity fallback.
- [ ] **R2.2** — optionally add 1–2 further recent multimodal baselines.
- [ ] Apply the corrected numbers/cohort/claims to the LaTeX (see `Corrected-Manuscript-Content.md`).
