# New paper — outline (methods paper: CIDA + OcuFM, honestly framed)

**Type:** methods + rigorous-analysis paper (NOT a SOTA-accuracy paper). Fresh
submission, designed so the prior reviewers' criticisms are the paper's *structure*.

> Honesty constraint baked in: CIDA and OcuFM have **no accuracy/transfer win** on our
> data (CIDA removes identity but doesn't raise AUC or rescue transfer; OcuFM is
> proposed, not pretrained). So the contributions are **designs + mechanism results +
> honest scoping**, anchored by the one solid empirical result (freeze > fine-tune).

## Title options
1. *Parameter-Efficient, Identity-Disentangled Adaptation of EEG and Gaze Foundation
   Models for Small-Sample Clinical Screening*
2. *Freeze, Don't Fine-Tune: Adapting Brain and Gaze Foundation Models for
   Transdiagnostic Screening, with Disentanglement and Honest Limits*
3. *Foundation-Model Adaptation for Small-Sample Psychiatric Screening: PEFT,
   Identity-Disentanglement (CIDA), and a Gaze Foundation Model (OcuFM)*

## Thesis (one sentence)
*How to adapt foundation models (EEG: LaBraM; gaze: the proposed OcuFM) for
small-sample transdiagnostic screening — with the empirical finding that freezing
beats all fine-tuning, a connectivity-aware identity-disentangling adapter (CIDA) that
provably removes subject identity, and a rigorous, leak-free, externally-validated
protocol that delineates where these methods help and where they do not.*

## Contributions (honest, evidence-tagged)
- **C1 — CIDA (design + mechanism result).** A connectivity-aware, identity-disentangled,
  parameter-efficient adapter for a frozen EEG foundation model. *Result:* removes
  real subject-identity from the clinical representation (decodability 0.26→0.20) at
  **no accuracy cost** (0.681→0.670), with a tunable identity/accuracy trade-off.
  *Honest scope:* it does **not** raise in-distribution AUC nor rescue MIPDB→HBN
  transfer (transfer is domain/label-shift-bound, ~0.52) — a finding that delineates
  *when* identity-invariance is and isn't the right tool.
- **C2 — "Freeze, don't fine-tune" (empirical methods finding).** Matched subject-level
  5×5 CV: frozen head-only **0.739 ± 0.046** > LoRA 2.88% **0.701 ± 0.048** > full
  fine-tune **0.686**. More backbone training monotonically hurts at N≈118 — a
  practitioner-relevant result and a direct answer to "model too big."
- **C3 — OcuFM (design + runnable skeleton + plan).** The first gaze foundation model on
  the LaBraM recipe (VQ tokenizer over gaze kinematics/pupil + masked-patch
  pretraining), with the unifying observation that EEG and gaze are *both*
  identity-dominated, so a single CIDA adapter serves both. *Honest scope:* proposed +
  runnable, **not pretrained**; no accuracy claimed.
- **C4 — Rigorous protocol.** Leak-free subject-level repeated CV with CIs, external
  validation on HBN (~2,000 subjects) with a *characterized* null, calibrated
  refer-to-clinician output, recent fusion baselines. Behaviour dropped (below chance).

## Section plan
1. **Introduction** — EEG foundation models exist (LaBraM/EEGPT/CBraMod); gaze has none;
   small clinical N + identity confounds + domain shift make naive adaptation fail.
   State the four contributions; state up-front this is a methods/analysis paper.
2. **Related work** — EEG FMs; PEFT (LoRA/adapters, TaKF⁺, EEG-GraphAdapter); subject-
   invariance / DANN; clinical EEG FMs (LEAD); gaze biometrics + eye-tracking ML.
3. **Methods** — (a) CIDA (connectivity adapter, attention pool, disentanglement head,
   losses); (b) OcuFM (recipe, tokenizer target, pretraining objective); (c) protocol
   (QC, windows, leak-free CV, calibration, external test).
4. **Experiments & results** —
   - 4.1 PEFT analysis: freeze vs LoRA vs full (Table; the C2 finding).
   - 4.2 CIDA mechanism: subject-decodability vs AUC trade-off (the C1 result).
   - 4.3 External validation + transfer: HBN null, characterized (C4 + C1 scope).
   - 4.4 OcuFM: architecture, pretraining plan, synthetic-run sanity (C3).
   - 4.5 Baselines + fusion (honest: fusion ≈ EEG-alone).
5. **Discussion** — when invariance/adaptation help; small-N foundation-model practice;
   limitations (N, single primary cohort, OcuFM unpretrained, transfer null).
6. **Conclusion** — methods + honest scope; future work (OcuFM pretraining; transfer
   without label mismatch).

## Tables/figures (reuse existing real results)
- T1: PEFT 5×5 (frozen/LoRA/full) — `results/finetune_labram_*_5x5.csv`.
- T2: CIDA identity/accuracy trade-off — `experiments/disentangle_real.py`.
- T3: external + transfer (HBN) — `results/phase4_hbn*.csv`, `transfer_hbn_disentangle.csv`.
- F1: CIDA + OcuFM architecture (regenerate `make_figures.py`).
- F2: subject-decodability vs λ_adv curve.

## Claim / DO-NOT-claim
- **Claim:** novel adapter (CIDA) + gaze-FM design (OcuFM); freeze-beats-fine-tune at
  small N; rigorous leak-free + external protocol; honest scoping of invariance.
- **DO NOT claim:** SOTA / leaderboard win; accuracy gain from CIDA or OcuFM; multimodal
  gain; that disentanglement improves transfer.

## Risk & mitigation
- *Risk:* methods paper with no accuracy win reads thin. *Mitigation:* lead with the
  freeze finding (real), present CIDA's identity removal as a measurable mechanism +
  the transfer null as a scientific delineation, and target a rigor-friendly venue
  (IEEE Access). **Optional strengthener:** a within-HBN (N≈2,000) experiment to give
  CIDA one genuinely positive setting, if one exists.
