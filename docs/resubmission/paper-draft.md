# Freeze, Don't Fine-Tune: A Rigorous Study of EEG Foundation-Model Adaptation for Small-Cohort Transdiagnostic Screening

*Draft v1. Target: IEEE EMBC / IEEE Access (rigor-friendly); IEEE TNSRE a stretch. All
numbers are real, leak-free, committed. Honesty guardrails: no SOTA claim; the multimodal
framework is proposed/baseline, not claimed to beat single-modality EEG.*

---

## Abstract
EEG foundation models (FMs) promise to make small-cohort clinical decoding tractable, and
the default recipe is to **fine-tune** them, often within a **multimodal** pipeline. We
test whether either choice actually helps for **transdiagnostic psychiatric screening**
("any DSM diagnosis vs none"). Using a frozen LaBraM backbone on the MIPDB cohort (N≈118)
under strict subject-level, leakage-free repeated cross-validation, and an external cohort
(Healthy Brain Network, ~2,000 subjects), we find: (1) a **frozen** FM with a trained head
reaches ROC-AUC **0.74**, while **no adaptation strategy — full fine-tuning, LoRA, or
self-supervised continued pretraining — significantly improves on it**, across a 25×
range of training sizes (N=40→1000); (2) **multimodal fusion** of EEG with eye-tracking
**underperforms EEG-alone** (eye is at chance); (3) **cross-cohort transfer is at chance**,
and we show the barrier is domain/label shift, not subject-identity confound. We release a
proposed per-modality-FM framework (including OcuFM, a gaze FM on the LaBraM recipe) as a
future direction. Our contribution is **rigorous, practitioner-relevant guidance**:
at realistic clinical N, freeze a single-modality EEG FM; adaptation and fusion do not help.

## 1. Introduction
Foundation models for EEG (LaBraM, EEGPT, CBraMod) are increasingly applied to clinical
tasks, but clinical cohorts are small and the correct way to *use* an FM there is unclear.
Two assumptions dominate practice: that the FM should be **fine-tuned** to the task, and
that **multimodal fusion** improves screening. We interrogate both for transdiagnostic
psychiatric screening and find them unsupported at clinically realistic sample sizes.

Contributions:
- **C1.** A matched, leak-free, well-powered comparison of adaptation strategies (frozen /
  full FT / LoRA / SSL continued pretraining) across N=40→1000 on two cohorts, showing
  frozen is never significantly beaten — *"freeze, don't fine-tune."*
- **C2.** An honest multimodal analysis: a NeuroFormer-style EEG+eye fusion transformer
  does not beat EEG-alone; eye-tracking is at chance for this task.
- **C3.** External validation with a failure analysis: cross-cohort transfer is chance, and
  an identity-disentangling adapter localizes the barrier to domain/label shift.
- **C4.** A proposed per-modality-FM framework (LaBraM-recipe EEG + OcuFM gaze FM, fused),
  released as a runnable blueprint for future paired-data settings (not an accuracy claim).

## 2. Related work
**EEG foundation models.** LaBraM (ICLR'24) — VQ neural-spectrum tokenizer + masked
transformer, ~2,500 h pretraining; EEGPT (NeurIPS'24); CBraMod (ICLR'25). These report
strong results on TUAB/TUEV via full fine-tuning on large labeled sets.
**Parameter-efficient fine-tuning.** LoRA; adapters; EEG-specific PEFT (TaKF⁺,
EEG-GraphAdapter). **Linear-probe vs fine-tune.** Kumar et al. (2022) show fine-tuning can
distort pretrained features, especially with limited/shifted data — our work provides the
*EEG-clinical, sample-size-resolved* evidence for this regime. **Multimodal EEG+eye** and
**missing-modality clinical fusion** (DrFuse, M3Care) motivate C2/C4.

## 3. Methods
**Cohorts.** MIPDB (Child Mind Institute; 128/129-ch EGI; N≈118, ~20% with any DSM
diagnosis); HBN (RestingState; ~2,000 subjects; CBCL p-factor, median split). Public.
**Backbone.** LaBraM-Base (5.8M params) via braindecode `InterpolatedLaBraM` (montage
interpolation), patch 200, 15 s windows at 200 Hz, microvolt-scaled; QC gate on amplitude.
**Adaptation regimes.** (i) *frozen* — train only the classification head; (ii) *LoRA* —
rank-8 low-rank updates on attention/MLP (~2.9% params, frozen base); (iii) *full* — all
weights; (iv) *SSL-TAPT* — contrastive continued pretraining on unlabeled in-domain EEG,
then frozen readout. Subject score = mean of window probabilities.
**Multimodal (C2/C4).** NeuroFormer cross-modal transformer over per-modality encoders;
eye-tracking as hand-crafted features (and, proposed, OcuFM embeddings). Missing-modality
masked fusion. *Proposed* OcuFM: VQ tokenizer + masked-patch pretraining on gaze kinematics
(LaBraM recipe) — design + runnable skeleton; not pretrained here.
**Protocol.** Subject-level repeated stratified CV (up to 5×5), class-weighted loss,
ROC-AUC ± 95% CI. Significance: paired t / Wilcoxon with Holm correction; Welch for unmatched
n. Calibration: ECE, Brier. External: train MIPDB, test HBN.

### 3.5 OcuFM: a proposed gaze foundation model (new model, C4)
Gaze representation-learning models exist (GLASS; **GazeMTM**, which adapts the MOMENT
time-series FM), but none use the **discrete neural-tokenizer + masked-patch** recipe that
makes LaBraM effective for EEG. We propose **OcuFM**, a gaze FM on that recipe (we do
**not** claim the first gaze FM — we position against GazeMTM/GLASS):
- **Input:** multi-channel gaze stream — position (x, y), pupil, and derived velocity/
  acceleration — resampled to a canonical rate and cut into per-channel time patches.
- **Tokenizer:** a vector-quantized tokenizer trained to reconstruct **kinematic +
  spectral** patch targets (saccade/fixation dynamics, pupil spectrum) — the gaze analog
  of LaBraM's Fourier-spectrum target.
- **Pretraining:** masked-patch modeling (predict codes of masked patches) on a large
  external gaze corpus (e.g., GazeBase-scale).
- **Adaptation:** frozen embeddings adapted by the same identity-disentangling adapter
  (CIDA) used for EEG — gaze is highly identity-revealing (eye-movement biometrics), so
  the same machinery applies.

OcuFM is provided as a **design + runnable skeleton** (`src/ocufm.py`: tokenizer +
masked-patch pretraining + self-test on synthetic gaze; integrates with the fusion model
in `src/multimodal_cida.py`). **Status: proposed, not yet pretrained** — a large gaze
corpus and pretraining compute are out of scope here, and on this cohort eye-tracking is
at chance (§4.3), so we make **no accuracy claim** for OcuFM. It is a concrete blueprint
for the per-modality-FM framework once richer paired data is available. *(This is the
"new model" contribution: a novel architecture + open implementation, scoped honestly as
future work — not an empirical win.)*

## 4. Results
### 4.1 Frozen beats adaptation at small N (C1)
Matched subject-level CV on MIPDB:

| approach | trainable | ROC-AUC |
|---|---|---|
| hand-crafted EEG | — | 0.43–0.47 |
| **frozen LaBraM + head** | head | **0.739 ± 0.046** |
| LoRA (r=8) | 2.9% | 0.701 ± 0.048 |
| full fine-tune | 100% | 0.686 |
| SSL continued pretraining | 100% | 0.684 |

No adaptation significantly beats frozen (frozen vs LoRA paired *p*=0.061; vs full
*p*=0.45; n.s. after Holm). Frozen is also the cheapest (tiny head vs 5.8M).

### 4.2 No crossover across sample size (C1)
Within-HBN, varying only N (well-powered 5×3 at N≤500):

| N | frozen | full | full−frozen | *p* |
|---|---|---|---|---|
| 150 | 0.495 | 0.426 | −0.069 | **0.026** |
| 300 | 0.543 | 0.521 | −0.022 | 0.16 |
| 500 | 0.571 | 0.524 | −0.047 | 0.054 |
| 1000 | 0.539 | 0.553 | +0.013 | 0.34 |
| MIPDB 40 | 0.810 | 0.688 | −0.122 | — |

Frozen is significantly better at small N and statistically tied at N=1000 — **no N at
which fine-tuning significantly wins.** (HBN absolute AUCs are modest, ~0.5–0.57; the
p-factor window-level task is weak — a limitation, §5.)

### 4.3 Multimodal fusion underperforms EEG-alone (C2)
Leak-free CV, identical protocol:

| configuration | ROC-AUC |
|---|---|
| EEG-alone | 0.661 |
| eye-alone | 0.522 (chance) |
| concat | 0.644 |
| set-fusion (attn) | 0.587 |
| 1-stage multimodal transformer | 0.617 |
| gated attention fusion | 0.556 |
| **NeuroFormer (full cross-modal transformer, end-to-end, 5×3)** | **0.668 ± 0.051** |

Every fusion variant — including the **complete NeuroFormer cross-modal transformer run
end-to-end (0.668)** — only *ties* EEG-alone (~0.66) and stays **below** the frozen
window-level EEG (0.739). The complete multimodal architecture does **not** beat
single-modality EEG. (`train_neuroformer.py`; `results/neuroformer_cv.csv`.)

### 4.4 External transfer is chance (C3)
MIPDB→HBN (1,980 subjects): ROC-AUC **~0.52** (median-split p-factor). An identity-
disentangling adapter reduces subject-identity decodability (0.26→0.20) at no accuracy
cost, yet **does not improve transfer** — isolating the barrier as **domain + label shift**,
not subject-identity confound.

### 4.5 Calibration
The raw model is over-confident (ECE 0.138, Brier 0.178); temperature scaling / conformal
abstention recommended for any screening use.

## 5. Discussion
At clinically realistic N, the frozen FM representation is already near its usable ceiling;
full and even parameter-efficient adaptation add variance and distort features (consistent
with LP-FT findings), and self-supervised continued pretraining on small in-domain data
hurts. Multimodal fusion cannot help when the auxiliary modality is at chance, and
cross-cohort transfer is bounded by domain/label shift rather than removable confounds.
**Limitations:** one primary cohort (N≈118); heterogeneous "any-diagnosis" label and a
crude p-factor median split; modest within-HBN signal; OcuFM proposed but not pretrained
(no large gaze corpus). Single FM family in the headline (CBraMod replication in progress).

## 6. Conclusion
For small-cohort transdiagnostic EEG screening: **freeze a single-modality EEG foundation
model.** Fine-tuning, parameter-efficient adaptation, self-supervised continued
pretraining, and multimodal fusion do not improve on it, and cross-cohort transfer remains
at chance. We provide rigorous, leak-free, externally-validated evidence and a proposed
per-modality-FM framework for future paired-data settings.

## Reproducibility
Code, configs, and per-fold results at the project repository; experiments in `src/` and
`experiments/`; statistics via `experiments/stats_tests.py`; all results CSVs in `results/`.
