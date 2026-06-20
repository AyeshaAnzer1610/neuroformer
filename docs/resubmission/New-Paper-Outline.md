# Paper outline — framework + rigorous study (per-modality FMs, fused; freeze-don't-adapt)

**Type:** a **framework paper backed by a rigorous empirical study.** Two complementary
contributions: (1) a *proposed* framework — per-modality foundation-model representations
(EEG via the LaBraM recipe; eye via the proposed **OcuFM**) fused in a transformer — and
(2) a well-powered, leak-free evaluation of how to use it, whose headline finding is
**freeze, don't adapt** (a frozen EEG FM matches/beats every adaptation across a 25×
sample-size range). The framework supplies novelty/positioning; the study supplies rigor.

> **Honesty guardrails (non-negotiable, or it gets rejected like v1):** the framework is
> *proposed + runnable*, **not** claimed to beat EEG-alone — on this cohort eye is at
> chance and fusion ≈ EEG-alone, and we say so. The empirical *result* is the freeze
> finding + honest scope, NOT a fusion/FM accuracy win. No SOTA claim.
>
> Venue: framework + rigor + honest negatives → **EMBC / IEEE Access realistic; TNSRE a
> stretch.** The CBraMod generality run is what most raises the bar.

## Title options
1. *Freeze, Don't Fine-Tune: Frozen EEG Foundation Models Match or Beat Adaptation
   Across Sample Sizes for Transdiagnostic Screening*
2. *When Adaptation Doesn't Help: A Leak-Free, Externally-Validated Study of EEG
   Foundation-Model Adaptation for Small-Cohort Psychiatric Screening*

## Narrative arc
Multimodal fusion (NeuroFormer, EEG+eye) → didn't beat EEG-alone (eye ≈ chance) →
focus on the EEG foundation model → freezing beats full FT, LoRA, and SSL continued
pretraining across N=40–1000 → external transfer is chance (domain/label shift) →
practical guidance: at clinically realistic N, freeze the backbone.

## Contributions
- **C0 — Framework (proposed): per-modality foundation models, fused.** EEG via the
  LaBraM recipe; **OcuFM**, a proposed gaze foundation model (VQ tokenizer + masked-patch
  pretraining on the LaBraM recipe) that fills the "no gaze FM exists" gap; both fused in
  a NeuroFormer-style cross-modal transformer, with an identity-disentangling adapter
  (CIDA). Implemented + runnable skeletons. *Proposed/positioning — explicitly NOT claimed
  to beat EEG-alone here (eye ≈ chance); offered as a blueprint for richer paired data.*
- **C1 — Comprehensive freeze-vs-adapt across sample size (the empirical result).** Matched, leak-free,
  subject-level CV on MIPDB (N≈118) + within-HBN N-sweep (150→1000): frozen LaBraM+head
  is **never significantly beaten** by full FT, LoRA, or SSL-TAPT; **significantly
  better at small N** (HBN N=150 p=0.026; N=500 p=0.054) and tied at N=1000 (+0.013,
  p=0.34). No significant crossover. (`crossover_aggregate.py`, `stats_tests.py` w/ Holm.)
- **C2 — Multimodal fusion baseline (context).** NeuroFormer (EEG+eye cross-modal
  transformer): eye ≈ chance (0.51), fusion ≈ EEG-alone — motivates the EEG-FM focus
  and honestly retires the over-claimed 3-modality story. (`eval_fusion.py`.)
- **C3 — External validation + failure analysis.** MIPDB→HBN ≈ 0.52 (chance, 1,980
  subj); an identity-disentangling adapter removes subject identity at no accuracy cost
  yet doesn't improve transfer → barrier is **domain/label shift, not identity**.
- **C4 — Calibration + efficiency.** Raw ECE=0.14 (over-confident) → temperature/conformal;
  frozen trains a tiny head vs LoRA ~3% vs full 100% — frozen is best *and* cheapest.

## Section plan
1. **Introduction** — EEG-FMs exist; clinical N is small; common practice is to fine-tune;
   we ask whether adaptation actually helps. State C1–C4; note this is a rigorous study.
2. **Related work** — EEG-FMs (LaBraM/EEGPT/CBraMod); PEFT (LoRA, adapters); **LP-FT /
   "fine-tuning distorts features" (Kumar 2022)** — position our contribution as the
   *EEG-clinical, sample-size-resolved* evidence, not the bare phenomenon; EEG+eye fusion.
3. **Methods** — (a) NeuroFormer fusion baseline (context); (b) LaBraM+head adaptation
   regimes (frozen / LoRA / full / SSL-TAPT); (c) protocol (QC, leak-free subject-level
   repeated CV, sfreq/window, calibration); (d) statistics (paired+Wilcoxon+Holm).
4. **Results** — 4.1 fusion ≈ EEG-alone (context); 4.2 freeze-vs-adapt across N (the
   table + figure, the headline); 4.3 external null + identity analysis; 4.4 calibration
   + efficiency.
5. **Discussion** — when (not) to adapt EEG-FMs at small clinical N; why fusion/identity
   don't help here; limitations (modest within-HBN signal ~0.5–0.57; one primary cohort;
   weak labels).
6. **Conclusion** — freeze at realistic clinical N; adaptation and fusion don't help;
   honest external boundary.

## Figures/tables (from real results)
- T1 freeze-vs-adapt × N (frozen/full/LoRA/TAPT, AUC±CI, p) — `crossover_aggregate.py`.
- F1 AUC-vs-N curve per mode (the headline) — `make_crossover_figure.py`.
- T2 fusion baseline (eeg/eye/fusion) — `eval_fusion.py`.
- T3 external/transfer (HBN) + identity-decodability — `transfer_hbn.py`, `disentangle_real.py`.
- T4 calibration (ECE/Brier) + efficiency (trainable params) — `calibration_metrics.py`,
  `efficiency_table.py`.

## Claim / DO-NOT-claim
- **Claim:** frozen ≥ adaptation across N (well-powered, leak-free, external); fusion
  doesn't beat EEG-alone; honest scope. **DO NOT claim:** a significant crossover, a SOTA
  number, that fine-tuning helps, or that fusion/CIDA/OcuFM improve accuracy.

## Optional hardening (cluster)
- CBraMod (2nd FM) freeze-vs-adapt → "holds across foundation models" (generality).
- Frozen-model calibration with `--save-preds`; higher-repeat small-N MIPDB points.
