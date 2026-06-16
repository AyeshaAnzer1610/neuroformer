# NeuroFormer — Full Methodology (A→Z)

End-to-end methods for the revised submission. All performance is measured under
strict subject-level, leakage-free cross-validation. Numbers reflect the actual
runs; `~` denotes counts that vary slightly with QC.

---

## A. Datasets and cohorts

**Primary — MIPDB (Child Mind Institute).** Multimodal recordings (EEG,
eye-tracking, behaviour) over a six-paradigm battery, with clinician DSM-5
diagnoses. Public (FCP-INDI S3), 128-channel EGI HydroCel net at 500 Hz.
Analysis cohort: **N ≈ 118** subjects with usable EEG and a diagnostic label.

**External — Healthy Brain Network (HBN-EEG).** OpenNeuro ds005505–ds005515
(Releases 1–11), BIDS/EEGLAB, 128-ch EGI, 500 Hz. **N ≈ 1,980** subjects with a
usable RestingState recording and CBCL-derived labels.

**Task.** Transdiagnostic binary screening — **any DSM diagnosis vs none** on
MIPDB (`DX_Status > 0`). HBN provides continuous CBCL bifactor scores
(`p_factor` general + attention/internalizing/externalizing); for a comparable
metric we use a **median split** of each factor.

---

## B. EEG preprocessing (identical across cohorts)

Per recording, in MNE-Python:
1. Load (`.raw`/`.mff` EGI for MIPDB, `.set` EEGLAB for HBN); pick EEG channels.
2. Rename the EGI vertex reference `E129 → Cz`; attach the **GSN-HydroCel-129**
   montage (3-D positions, required for spatial interpolation).
3. Band-pass **0.1–75 Hz** (FIR), **60 Hz** notch (US line noise), resample to
   **200 Hz** (LaBraM's expected rate).
4. Scale **volts → microvolts** (LaBraM's training unit; the model applies no
   internal scaling).
5. Segment into **non-overlapping 15 s windows** (3,000 samples = 15 × 200 Hz),
   matching LaBraM's temporal embedding.
6. **Artifact clip** to ±800 µV (above the 99.5th percentile of clean EEG; bounds
   gross artifacts without touching physiological signal).

**Quality-control recording selection.** Subjects can have multiple recordings of
varying quality. We iterate candidate files (largest first) and accept the first
whose windowed signal is physiological — **median |x| ∈ [3, 150] µV** and
**< 15 % clipped** — discarding flat/dead files and impedance/calibration files
(median in the hundreds–thousands of µV). For HBN we restrict to the
**RestingState** task for cross-subject consistency.

---

## C. Foundation-model EEG representation (the core change)

Each 15 s window is mapped to a **200-dimensional embedding** by a **frozen**
pretrained **LaBraM** (`braindecode` `InterpolatedLaBraM`,
`braindecode/labram-pretrained`). The interpolation front-end maps the 129-channel
montage onto LaBraM's canonical channels using the electrode positions; we take
the model's `[CLS]` token (window-level summary), falling back to mean-pooled patch
tokens. Embeddings are extracted with no gradient updates (frozen) and cached.

**Per-subject pooling.** Window embeddings are aggregated to one vector per subject
by **mean+std pooling** (concatenated mean and standard deviation across windows,
400-d), which outperformed mean-only — cross-window variability is informative.

This replaces the original hand-crafted 5-channel EEG features; the downstream
NeuroFormer EEG encoder is otherwise unchanged.

**How the representation is used (training-regime ablation).** Under identical
subject-level CV we compare three regimes (GPU, `src/finetune_labram.py`): (i)
subject-pooled linear probing (ROC-AUC ≈0.66); (ii) **window-level training** of a
classification head on the frozen backbone — each 15 s window is a training
instance, predictions pooled per subject (multiple-instance learning) — reaching
**≈0.80**; and (iii) end-to-end fine-tuning of the full backbone, which **overfits**
at N≈118 (0.69 ± 0.12, vs 0.795 ± 0.04 for the frozen regime; ~3× the fold
variance). The recommended configuration is therefore the **frozen foundation model
with lightweight window-level training**. See `docs/resubmission/Phase5-Finetuning-Results.md`.

---

## D. Eye-tracking and behavioural features (re-derived from raw)

**Eye-tracking** (SMI iView `*Samples.txt`): keep `SMP` sample rows; per block
compute **11 features** — pupil diameter (mean/std of L/R `Dia X`), gaze position
(mean/std of L/R `POR X/Y`), head-position variability (`H POS X/Y/Z`), latency
(mean/std), and gaze-vector (mean/std of L/R `GVEC X/Y/Z`).

**Behaviour** (SAIIT 2AFC task logs, `Behavioral/csv`): **6 features** — mean/std/
median/min/max reaction time (`CD_RESP − CD_TG`) and accuracy (`trialLR == RespLR`).

Both are computed per block and pooled (mean) per subject. Re-derived values were
**validated against the original pipeline's features (per-feature r ≈ 1.0)**.

**Hand-crafted EEG baseline** (for the head-to-head): Hjorth activity/mobility/
complexity + Welch spectral entropy per channel, on the same windows.

---

## E. Multimodal alignment (honest, asynchronous)

The three modalities come from **different paradigms** and are not trial-
synchronized. We therefore align at the **subject level** (one feature vector per
modality per subject, keyed by subject ID) and treat fusion as **asynchronous,
cross-paradigm** integration — explicitly *not* synchronized cross-modal attention.
Subjects missing a modality are tracked with an explicit **availability mask**
(coverage: EEG 118, eye 109, behaviour 113; complete-case ≈ 104; union ≈ 120).

**Why a foundation model for EEG only, and what "alignment" means here.** Two
deliberate design choices, both dictated by data availability and cohort size:

1. *Representation.* We apply a pretrained foundation model only to EEG because
   mature, downloadable EEG foundation models exist (LaBraM/EEGPT/CBraMod, pretrained
   on thousands of hours), whereas no foundation model of comparable maturity exists
   for SMI eye-tracking sample streams, and the concept does not meaningfully apply to
   the low-dimensional behavioural summaries (6 RT/accuracy statistics). EEG is also
   the modality where representation learning buys the most — a raw high-dimensional
   multichannel time series — which is precisely why foundation-model EEG (0.66)
   surpasses hand-crafted EEG (0.43–0.47), while the already-compact eye (11-d) and
   behaviour (6-d) features have little additional structure for a learned encoder to
   exploit (and sit at chance in this cohort regardless).

2. *Alignment.* "Alignment" here means **subject-level keying** (joining modalities by
   participant ID under an availability mask), **not** a learned shared embedding space.
   We do *not* perform representation-space (e.g. contrastive, CLIP-style) alignment:
   it requires many paired cross-modal examples, and our complete-case N ≈ 104 is far
   too small to fit one without overfitting. Trial-level temporal alignment is also
   impossible because the modalities are acquired in **separate, asynchronous
   paradigms** with no shared time origin. Subject-level alignment is therefore the
   honest and appropriate choice for this data, and we adopt permutation-invariant
   set-based fusion consistent with the absence of a shared time origin (cf. R3.6).

*Future work.* A principled extension is to pretrain a gaze-sequence encoder on raw
eye-tracking and learn a contrastive EEG↔gaze representation alignment on a larger
paired cohort (e.g. HBN, which records both) — out of scope for this revision given
N, but the natural next step beyond subject-level keying.

---

## F. NeuroFormer architecture (unchanged from the original)

Three modality-specific transformer encoders (shared design) → a **cross-modal
fusion transformer** with a learnable `[CLS]` aggregation token and modality-type
embeddings → an MLP classifier; optional auxiliary reconstruction decoders. Each
stream is linearly projected to the model dimension with sinusoidal positional
encoding. We additionally report a **compact** variant (~2.9 M parameters) to match
the cohort size and address overparameterization. The EEG stream's *input* is now
LaBraM embeddings (200-d) instead of 128-d hand-crafted features; the architecture
is otherwise identical.

**Optional cross-modal alignment (off by default).** The framework also includes an
optional `CrossModalAlignment` module that projects each modality's pooled encoding
into a shared, L2-normalized space and adds a symmetric InfoNCE contrastive loss
(pulling together the modalities of the same subject, pushing apart different
subjects), respecting the availability mask. This addresses the natural question of
representation-space (not merely subject-level) alignment. It is **disabled for all
headline results**: on the small complete-case cohort (N≈104) it is not expected to
add accuracy and would increase parameter count, so we keep it as a documented
robustness/transfer mechanism and a basis for the future contrastive EEG↔gaze
direction (section E), not an accuracy claim.

---

## G. Missing-modality training

**Modality dropout**: during training each modality is randomly withheld per
sample (never all of them); the modality-type embeddings act as availability
indicators. One trained model therefore serves any subset of {EEG, eye, behaviour}
with graceful degradation, reflecting real screening conditions.

---

## H. Calibration and referral-aware prediction

Post-hoc on leak-free out-of-fold scores: **temperature scaling** (minimising NLL)
with **Expected Calibration Error** before/after; **split-conformal** prediction
sets at error rate α (decisive singleton vs referred `{0,1}`), and a
**risk–coverage** curve — abstaining on low-confidence cases ("refer to clinician").

---

## I. Evaluation protocol (leakage-free)

- **Strict subject-level** partitioning — every recording from a subject stays in
  one fold (no subject in train and test).
- **Repeated stratified k-fold** (5 folds × 10 repeats), class-balanced.
- All fold-dependent steps (standardization, calibration, any feature fitting) are
  fit on **training folds only**.
- **Paired** comparisons across feature sets/models use **identical CV splits**.
- **Metrics:** ROC-AUC (primary; ~20 % positive prevalence makes accuracy
  misleading) + balanced accuracy, with **95 % CIs** across folds; paired AUC deltas.
- **Baselines:** hand-crafted EEG, eye-alone, behaviour-alone, naive concatenation,
  and the NeuroFormer fusion variants (mean / attention / stacking late fusion),
  all under the same features, splits, and protocol.
- No heavy hyperparameter tuning (small N → avoids optimistic bias).

---

## J. External validation (HBN)

Frozen LaBraM embeddings are extracted from HBN RestingState with the *same*
preprocessing/QC. Within-HBN, we predict each CBCL factor from the embeddings
(Ridge for regression Pearson r; logistic regression on the median split for AUC),
under the same subject-level CV. We also run a **transfer** test (MIPDB any-DX model
applied to HBN), reported with the explicit label-construct caveat. All at the
well-powered scale **N ≈ 1,980**.

---

## K. Implementation and compute

Digital Research Alliance of Canada **Nibi** cluster. Python 3.11; MNE-Python;
`braindecode` ≥ 1.5; PyTorch (CPU for frozen extraction and the compact fusion;
GPU env prepared for fine-tuning). Raw data pulled from public S3 to `$SCRATCH`;
frozen embeddings cached per subject; cohort extraction parallelised across CPU
workers via SLURM, with per-batch device transfer to bound memory. Fixed random
seeds; subject-level splits seeded for reproducibility.

---

## L. Reproducibility

All code, preprocessing, the model, the evaluation harness, and per-phase result
writeups are released (`github.com/AyeshaAnzer1610/neuroformer`). Raw data is public
and re-downloadable (`scripts/download_data.sh`); the pipeline is one `sbatch` per
stage. Reporting follows standard ML-in-health practice (subject-level CV, CIs,
honest negative results).

---

## M. Summary of what each step addresses (reviewer mapping)

| Step | Concern addressed |
|---|---|
| Leak-free subject-level CV + compact model (I, F) | R3.3 fold sensitivity, R3.4 overparameterization |
| Foundation-model EEG (C) | R3.8 "is EEG worth including", R2.1/2.4 |
| Asynchronous subject-level fusion + modality dropout (E, G) | R3.6 fusion premise, R3.9, R3.10 missingness |
| Calibration / referral (H) | R3.2 clinical-screening realism |
| External HBN validation (J) | R1.3, R3.2, R3.5 generalization |
| Honest multimodal + external reporting (I, J) | overall rigor / R3.9 softened claims |
