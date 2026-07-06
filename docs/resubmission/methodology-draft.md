# Methodology (draft) — NeuroFormer: a unified multimodal architecture and a leakage-free evaluation

> Manuscript-ready draft. Contribution framing = **rigor + architecture**, not an accuracy win.
> All numbers are from the standardized reference-suite runs on the locked cohorts
> (`results/*.csv`). NeuroFormer numbers are from this suite's reimplementation, reported
> as such (not the authors' external pipeline). Part A = Methods; Part B = Results &
> interpretation (what/why, incl. why-not-better on MIPDB and where-better on ds003478).

---

## Abstract (draft)

Multimodal EEG / eye-tracking screening for paediatric psychiatric status has reported high
accuracies, but frequently under evaluation schemes vulnerable to information leakage. We
present **NeuroFormer**, a unified architecture that ingests one-to-three asynchronous
modalities through modality-specific encoders and a confidence-gated fusion with an explicit,
mask-based missing-modality mechanism; and — as our central methodological contribution — a
**standardized, leakage-free, confound-controlled evaluation protocol** applied *identically*
to NeuroFormer and 13 comparator models across two cohorts. On a rigorously controlled
paediatric cohort (N = 92), subject-level cross-validation, 1000-permutation testing,
per-model bootstrap confidence intervals, and DeLong comparisons show that **no model —
including NeuroFormer — exceeds a two-variable age+sex baseline (AUC 0.737)**, while
window-level leakage inflates AUC to 0.95–1.00 versus ≈0.51 honest; covariate-matched
subsampling and confound-conditioned permutation attribute the residual signal largely to
demographics. On an independent adult cohort (ds003478, EEG-only), by contrast, EEG genuinely
predicts depression (AUC ≈ 0.75), exceeds demographics, and survives age/sex residualization
and within-stratum permutation (p ≈ 0.001). The missing-modality mechanism is shown to remain
operational under dropped modalities. Our contribution is the unified architecture together
with an honest, reproducible characterization of what multimodal EEG screening can and cannot
deliver at realistic sample sizes.

## Contributions

1. **NeuroFormer**, a unified architecture supporting one-to-three asynchronous modalities via
   modality-specific encoders and confidence-gated fusion, with a **demonstrated** (implemented
   and evaluated) mask-based missing-modality mechanism.
2. A **standardized, leakage-free, confound-controlled evaluation protocol** — subject-level CV,
   LOSO, nested CV, bootstrap, 1000-permutation testing, a demographic baseline, DeLong tests,
   and calibration — applied *identically* to all 14 models on identical inputs.
3. An **honest two-cohort characterization**: on a controlled paediatric cohort no model beats a
   two-variable demographic baseline and prior high accuracies are shown to be leakage/confound
   artifacts; on an independent cohort where the confound is weak, EEG-based signal is genuine.
   We make no claim of accuracy superiority.

---

## Part A — Methods

### A.0 Overview and contributions
We study multimodal (EEG + eye-tracking) screening of pooled paediatric psychiatric
status. Our contributions are: (i) **NeuroFormer**, a unified architecture that ingests
one-to-three asynchronous modalities through modality-specific encoders and a
confidence-gated fusion, including a masked missing-modality mechanism; (ii) a
**standardized, leakage-free, confound-controlled evaluation protocol** applied *identically*
to NeuroFormer and 13 comparator models; and (iii) an **honest two-cohort characterization**
of what such screening can and cannot deliver at realistic sample sizes. We make no claim of
accuracy superiority; the value is the architecture together with the rigorous evaluation.

### A.1 Datasets and cohorts
- **MIPDB (primary, within-cohort).** Resting-state EEG with concurrent eye-tracking.
  Binary label `DX_Status` (diagnosed vs control). After requiring **both** EEG and eye
  modalities present (Section A.4), the locked analysis cohort is **N = 92 (20 positive,
  72 control)**. The locked subject list is versioned (`cohort_final.txt`).
- **ds003478 (external replication).** Cavanagh resting-state EEG (adults), **EEG-only**;
  depression label from BDI (≥ 14 = depressed; extreme-groups ≤ 7 vs ≥ 14 as sensitivity).
  N ≈ 121 (45 depressed). No eye-tracking or behavioural modality exists in this dataset,
  so multimodal-fusion models are reported as N/A there rather than run degenerately.

Two cohorts serve complementary roles: MIPDB for within-cohort rigor under a demographic
confound; ds003478 for external replication in a cohort where that confound is weak.

### A.2 Signal acquisition and preprocessing (EEG)
Identical pipeline for both cohorts. Continuous EEG is (i) notch-filtered at 60 Hz,
(ii) band-pass filtered 1–45 Hz (4th-order Butterworth), and (iii) cleaned with ICA for
ocular/muscular/cardiac artifact removal (Winkler et al., 2011). Cleaned signal is segmented
into non-overlapping **15 s windows**.

### A.3 Feature extraction
- **EEG (per window, per channel — 15 features):** three Hjorth parameters (activity,
  mobility, complexity), spectral entropy, five absolute band powers, five relative band
  powers, and the theta/beta ratio. Bands: delta (1–4 Hz), theta (4–8), alpha (8–13),
  beta (13–30), gamma (30–45). MIPDB uses a 111-channel EGI scalp montage (channels
  E1–E111) → **1665 EEG features/window** (111 × 15); ds003478 has fewer channels (feature
  length truncated to the common minimum across subjects to preserve channel-contiguous
  ordering).
- **Eye-tracking (per subject):** oculomotor summary features — pupil-size variability,
  mean/SD horizontal and vertical gaze position, head-position variability, saccade latency
  (mean/SD), and gaze-vector magnitude (mean/SD).
- **Behaviour:** not available for the MIPDB analysis cohort (empty stream); the architecture
  supports it but it is not used here, and no behavioural result is reported.

### A.4 Modality assembly and dimensionality
Two feature representations are derived per subject: a **subject-level** vector (mean over a
subject's EEG windows, concatenated with the eye vector) for the static models, and a
**per-window EEG sequence** for the temporal models. For numerical stability at small N, EEG
features are reduced to the **top-128 highest-variance dimensions** (variance computed on
training folds' convention; fixed index set per cohort). All features are standardized with a
`StandardScaler` **fit on the training fold only** (no test leakage). The locked cohort is the
intersection of EEG-present and eye-present subjects (N = 92).

**Uniform inputs across models (and comparability note).** Every model — NeuroFormer included —
receives the *same* 128-dimensional handcrafted EEG (+ eye) representation, so all 14 methods
are compared on identical inputs (a fairness constraint of the protocol). Note this differs
from the original NeuroFormer implementation, which consumed a 516-dimensional processed EEG
representation plus a behavioural stream (three modalities); NeuroFormer's score under the
present standardized protocol is therefore **not directly comparable** to that original number,
and we report it only as one entry in this uniform benchmark.

### A.5 Models — a 14-model comparator suite
All models predict the binary label and are evaluated under the *identical* protocol
(Section A.7). Class imbalance is handled with balanced class weights / positive-class loss
weighting throughout.

**Static feature-space (subject-level concat):** Logistic Regression, RBF-SVM, Random Forest,
MLP (early fusion), Late-fusion ensemble.
**Temporal (per-window EEG sequences):** 1D-CNN, LSTM, TCN.
**Multimodal fusion:** Gated Multimodal Unit (GMU; Arévalo 2017), Cross-modal attention
Transformer, Tensor Fusion Network (Zadeh 2017), Low-rank Multimodal Fusion (Liu 2018).
**Proposed:** NeuroFormer (Section A.6).
**Null baseline:** **Demographics** — logistic regression on **age and sex only** (no EEG,
no eye). This is the reference every model must clear.

### A.6 NeuroFormer architecture (input → output)
NeuroFormer is a two-stage, modality-specific encoder with confidence-gated fusion.

1. **EEG encoder (hierarchical).** Per-sample Euclidean alignment (channel-wise
   standardization) → an EEGNet-style tokenizer (temporal convolution + depthwise-separable
   spatial convolution, batch-norm, ReLU) → **criss-cross attention** (parallel spatial and
   temporal multi-head self-attention, summed) → **frequency-aware attention** (the hidden
   representation is split into four band sub-spaces, each refined by band-specific attention,
   then recombined by a learned band-fusion weighting) → a residual MLP. Hidden width 64.
2. **Eye/behaviour encoders.** Linear projection + ReLU + dropout to the shared 64-d space.
3. **Confidence-gated fusion.** Each modality emits a scalar **confidence** (Linear → sigmoid);
   a **softmax over confidences** produces fusion gates; the fused representation is the
   confidence-weighted sum of modality embeddings, passed to a 2-way classifier. An optional
   cross-modal refinement lets the EEG embedding modulate the weaker modalities.

**Missing-modality mechanism (implemented and evaluated).** Availability is represented by a
per-sample mask: an absent modality's confidence logit is set to −∞ (excluded from the
softmax) and its features zeroed — i.e. it is *masked out of the gate*, not fed as zeros.
The model is trained with **modality dropout** (each training sample randomly drops one
modality with probability 0.5), so it learns to classify from any available subset. At
inference the same model is tested under EEG+eye, EEG-only, and eye-only availability.

### A.7 Evaluation protocol (the methodological core)
The central design choice is **subject-level cross-validation**: all windows/records of a
subject fall entirely in train or test. We contrast this with **window-level** CV (windows of
the same subject in both train and test), which reproduces the inflated accuracies reported in
prior work *on our own data* and quantifies the leakage. Primary metric: **ROC-AUC**
(discrimination), the natural quantity for "is this separable above the baseline?".

We report, for every model:
- **Repeated 5-fold** (5 folds × 10 repeats): per-fold AUCs and their 2.5/97.5 percentile CI.
- **Pooled out-of-fold (OOF) AUC** from a single stratified 5-fold, with a **subject-level
  bootstrap 95% CI** (2000 resamples). To guarantee cross-table consistency, OOF predictions
  are computed **once** (canonical source) and reused for the point AUC, DeLong, and
  calibration (Section A.10).
- **Leave-one-subject-out (LOSO)** — the strictest subject-level check (pooled OOF AUC + CI).
- **Nested CV** — inner grid-search for hyperparameters, outer loop for evaluation, so tuning
  cannot leak into the estimate (tunable models; torch nets tuned over weight-decay).
- **.632+ bootstrap** — a small-N-appropriate estimator alongside the fold-mean and OOF.
- **20-seed repeated 5-fold** — the full stability distribution (100 folds), not just the mean.
- **Label-permutation test** (1000 permutations) — per-model p-value for above-chance AUC.

### A.8 Confound-control analyses
Because the diagnosed and control groups may differ demographically, we test whether
classification survives controlling for age and sex:
- **Demographic baseline** (age+sex logistic regression) as the null to clear.
- **Covariate-matched subsampling** — repeatedly resample sex-matched and joint age+sex-matched
  case/control sets and re-estimate AUC (200 resamples), removing the demographic difference by
  design.
- **Confound-conditioned permutation** — permute labels *within* demographic strata (sex; and
  sex × age-tertile) rather than globally, so the null preserves the confound structure.
- **Regress-out vs include-as-feature** — compare AUC when age+sex are (a) left alone,
  (b) regressed out of the features (train-fit residuals), or (c) appended as extra features.

### A.9 External replication (ds003478)
The full applicable model suite plus the confound and permutation analyses are re-run on
ds003478 (EEG-only). This tests whether the leakage pattern (leaky ≫ honest) and the confound
pattern (honest ≈ demographic baseline) generalize, and whether an above-baseline EEG signal
exists in a cohort where the age/sex confound is weak.

### A.10 Statistical analysis and reproducibility
Model comparisons use the **DeLong test** for correlated ROC-AUCs on the shared OOF predictions
(pairwise, and each model vs the demographic baseline). Calibration is summarized by **Brier
score** and **Expected Calibration Error** (10 bins). All randomness is seeded; `SVC` Platt
calibration and torch operations are seeded/deterministic, and — critically — the benchmark
point AUC, DeLong, and calibration all read a **single canonical OOF file**, so the same model
has an identical AUC in every table. Code and per-model outputs are released
(`src/model_suite.py`, drivers `17–25`, `results/*.csv`).

---

## Part B — Results & interpretation (what we tested, and why it is / isn't better)

We first establish, on the controlled paediatric cohort, that under leakage-free evaluation the
fourteen models are statistically indistinguishable from one another and from a two-variable
demographic baseline (B.1). We then explain *why* — decomposing the apparent signal into a
demographic confound, small-sample variance, and the evaluation leakage that inflated prior
reports (B.2), and showing that neither modality fusion nor the missing-modality mechanism
recovers separable signal (B.3). Finally, we replicate the protocol on an independent adult
cohort where the confound is weak, and there — in contrast — EEG carries genuine,
demographics-independent predictive signal (B.4). Together these establish the paper's claim:
not a leaderboard result, but a rigorous, two-cohort account of where multimodal EEG screening
does and does not hold (B.5).

### B.1 Headline: no model separates from demographics (MIPDB, N = 92)
Under leakage-free subject-level evaluation the 14 models form **one overlapping band**: **no
model's 95% CI excludes the age+sex baseline (AUC 0.737)**, which is in fact the single
best-scoring model, and **no pairwise difference is significant** (DeLong). We therefore report
the benchmark ordered by AUC and lead with the overlap, not the rank.

**Table 1. Leakage-free benchmark (MIPDB, N = 92), ordered by out-of-fold AUC.**
Point AUC = pooled OOF; 95% CI = subject-level bootstrap (2000); permutation p = 1000 label
shuffles of the repeated-CV AUC.

| Model | AUC | 95% CI | perm p |
|---|---|---|---|
| MLP early-fusion | 0.537 | [0.362, 0.704] | 0.304 |
| Logistic Regression | 0.544 | [0.368, 0.708] | 0.177 |
| LSTM | 0.553 | [0.384, 0.712] | 0.064 |
| Tensor Fusion Network | 0.582 | [0.410, 0.742] | 0.015 |
| TCN | 0.591 | [0.405, 0.765] | 0.189 |
| **NeuroFormer** | **0.606** | **[0.473, 0.732]** | **0.054** |
| Cross-modal Transformer | 0.616 | [0.445, 0.776] | 0.065 |
| Late-fusion ensemble | 0.617 | [0.463, 0.758] | 0.041 |
| 1D-CNN | 0.623 | [0.445, 0.784] | 0.010 |
| GMU (gated fusion) | 0.624 | [0.464, 0.767] | 0.092 |
| RBF-SVM | 0.677 | [0.535, 0.812] | 0.046 |
| Random Forest | 0.704 | [0.575, 0.823] | 0.012 |
| Low-rank Multimodal Fusion | 0.719 | [0.586, 0.848] | 0.008 |
| **Demographics (age+sex)** | **0.737** | **[0.613, 0.849]** | **0.006** |

NeuroFormer sits mid-pack (6th of 14; five models score below it), and its result is stable
across every alternative scheme: LOSO 0.61, 20-seed repeated CV 0.61 ± 0.16, .632+ 0.64,
nested CV 0.58 — i.e. the mid-band position is not an artifact of one CV design.

**Table 2. DeLong test, each model vs the demographic baseline (MIPDB).**
Negative ΔAUC = below demographics; *no model is above it.*

| Model | AUC | ΔAUC vs demo | p |
|---|---|---|---|
| MLP early-fusion | 0.537 | −0.200 | **0.025** |
| Logistic Regression | 0.544 | −0.192 | **0.033** |
| LSTM | 0.553 | −0.184 | **0.033** |
| Tensor Fusion Network | 0.582 | −0.155 | 0.088 |
| TCN | 0.591 | −0.145 | 0.094 |
| NeuroFormer | 0.606 | −0.131 | 0.166 |
| Cross-modal Transformer | 0.616 | −0.121 | 0.063 |
| Late-fusion ensemble | 0.617 | −0.120 | 0.171 |
| 1D-CNN | 0.623 | −0.114 | 0.168 |
| GMU | 0.624 | −0.113 | 0.157 |
| RBF-SVM | 0.677 | −0.059 | 0.525 |
| Random Forest | 0.704 | −0.033 | 0.673 |
| Low-rank Multimodal Fusion | 0.719 | −0.017 | 0.830 |

Raw p's flag demographics as nominally better than the three weakest models (MLP, LogReg, LSTM;
p ≈ 0.03), but **after Benjamini–Hochberg FDR correction across the 13 comparisons none survives
(0/13), and no pairwise difference among all 14 models survives either (0/91)** — under
Bonferroni likewise 0/91. In other words, once multiplicity is accounted for the entire
benchmark is a **single statistically indistinguishable band**: no model separates from
demographics, and no model separates from any other. This — not any ranking of the models — is
the result.

### B.2 Why it is not "better": the signal is largely a demographic confound
Three independent lines converge:
- **Matched subsampling.** Under joint age+sex matching the demographic baseline collapses from
  0.71 to **0.40** and the models fall toward chance — the apparent signal was riding on age/sex.

  **Table 3. Covariate-matched subsampling (200 resamples, mean AUC).**

  | Model | sex-matched | joint age+sex-matched |
  |---|---|---|
  | Demographics (age+sex) | 0.710 | **0.401** |
  | Low-rank Multimodal Fusion | 0.661 | 0.490 |
  | Random Forest | 0.698 | 0.567 |
  | NeuroFormer | 0.616 | 0.579 |

- **Residualization.** Regressing age+sex out of the EEG features removes the EEG advantage on
  MIPDB (adding age+sex as features, not removing them, is what helps).
- **Permutation.** After FDR correction, five models exceed chance (Demographics, Random
  Forest, 1D-CNN, Tensor Fusion, Low-rank Fusion; q < 0.05); but exceeding chance ≠ exceeding
  demographics (none does, above), and the confound-conditioned permutation shows the effect is
  not independent of age/sex.

Two further reasons it cannot be "better" here: **small N** (20 positives → wide CIs, low
power to separate models), and the fact that **prior 90%+ numbers were leakage** — on our data
window-level (leaky) CV inflates AUC to **0.95–1.00** vs **~0.51** honest, so there was never a
real accuracy headroom to win.

### B.3 Fusion and missing modalities
No fusion model exceeds EEG-alone. In the missing-modality experiment (modality-dropout
training, masked inference), the model remains operational under any single modality:

**Table 4. Missing-modality NeuroFormer (subject-level, 5×10 CV).**

| Availability | AUC | 95% CI |
|---|---|---|
| EEG + eye (full) | 0.560 | [0.318, 0.817] |
| EEG-only | 0.597 | [0.284, 0.920] |
| eye-only | 0.554 | [0.290, 0.785] |

Performance does not collapse when a modality is removed (the mechanism works), but all three
conditions sit in a low band overlapping chance, and **EEG-only ≥ EEG+eye** — the eye stream
adds no signal. We therefore report the mechanism as *functional under missing modalities*, not
as *graceful degradation from strong performance* (there is no strong full-modality baseline to
degrade from).

### B.4 Where it IS better: ds003478 (external, EEG-only)
In a cohort without the strong age/sex confound, EEG genuinely predicts depression.

**Table 5. ds003478 (N ≈ 121), EEG models vs demographics.**

| Model | AUC | permutation p (global / within sex×age) |
|---|---|---|
| Logistic Regression (EEG) | 0.747 | 0.001 / 0.001 |
| RBF-SVM (EEG) | 0.751 | 0.001 / 0.002 |
| 1D-CNN (EEG) | 0.728 | 0.001 / 0.003 |
| **Demographics (age+sex)** | **0.624** | 0.028 / 0.266 (n.s. within strata) |

EEG models reach **≈0.75**, **exceed the demographic baseline (0.62)**, remain significant under
**confound-conditioned permutation (p ≈ 0.001 within sex × age-tertile strata)**, and are
**unchanged by age+sex residualization (0.747 → 0.753)**. This is the positive counterweight: the negative
MIPDB result is a property of that cohort's confound, not a claim that EEG is uninformative in
general.

### B.5 What this establishes
A rigorous, uniform, two-cohort evaluation shows that (a) apparent multimodal SOTA in this
setting is driven by leakage and demographic confounds, (b) at realistic sample sizes no
architecture — ours included — separates from a two-variable baseline, and (c) genuine
EEG-based signal does exist where the confound is controlled. The contribution is the unified
missing-modality architecture together with this honest characterization — a cautionary,
reproducible benchmark rather than a leaderboard result.

### B.6 Transparency and limitations
We state the following explicitly so that none of it can read as concealed; each follows
directly from the study's design.
- **Relation to prior high accuracies.** Earlier reports (including our own) of ~0.9 AUC in
  this setting arise from **window-level evaluation**, in which windows of the same subject
  appear in both train and test. Under identical models this inflates AUC to **0.95–1.00**
  versus **≈0.51** under subject-level evaluation (Table/Fig, leakage). The honest numbers
  supersede the earlier ones.
- **NeuroFormer inputs and comparability.** All 14 models — NeuroFormer included — are
  evaluated on **identical inputs** (128-dim handcrafted EEG + eye, two modalities) under one
  protocol. This differs from the original NeuroFormer implementation (516-dim processed EEG +
  behaviour, three modalities); the score here is therefore **not directly comparable** to that
  earlier number, and even that earlier number (≈0.67) is below the demographic baseline (0.74).
  Code is released for inspection.
- **Cohort selection.** The MIPDB analysis cohort (N = 92) is the subset of subjects with
  **both** EEG and eye modalities present. This is a complete-case criterion and a source of
  **selection bias** (a research cohort with high-quality dual recordings), which we do not
  claim to eliminate.
- **Two, not three, modalities.** Behaviour is absent for this cohort; despite the
  architecture's one-to-three-modality design, only EEG and eye are used, and no behavioural
  result is reported.
- **Cross-cohort scope.** ds003478 is an **adult depression** cohort, not a replication of the
  paediatric task; it demonstrates that EEG *can* carry demographics-independent signal where
  the confound is weak, not that the same model transfers.
- **Missing-modality claim.** The mechanism is demonstrated to be **operational** under dropped
  modalities; because full-modality performance is itself low, we do not claim graceful
  degradation *from strong performance*.
- **Multiple comparisons.** DeLong and permutation p-values are corrected with Benjamini–Hochberg
  FDR (and Bonferroni); corrected tables are in `results/*_corrected.csv`. After correction no
  pairwise model difference is significant (0/91) and no model differs from demographics (0/13);
  raw values are additionally provided. Correction *strengthens* the central claim.
- **Estimators.** Point AUCs are pooled out-of-fold; permutation tests operate on the
  repeated-CV AUC (a distinct, more stable estimator) — the two differ marginally by
  construction and are labelled as such.
- **Sample size.** N = 92 with 20 positives yields wide confidence intervals and limited power
  to separate models; we therefore lead with interval overlap and non-significance rather than
  rank order, and do not over-interpret point differences.

---

## Part C — Discussion

**Principal finding.** Under a uniform, leakage-free, confound-controlled protocol, no
architecture — from logistic regression to a purpose-built multimodal transformer — separates
from a two-variable age+sex baseline on this cohort; after multiplicity correction the fourteen
models form a single statistically indistinguishable band. The high accuracies previously
reported in this setting are reproduced *only* under window-level evaluation, where windows of a
subject leak across the train/test split (AUC ≈ 1.0 vs ≈ 0.5 honest). We therefore interpret
prior multimodal "state-of-the-art" in this domain as, in substantial part, an artifact of
evaluation design and demographic confounding rather than evidence of a learned neuro-behavioural
biomarker.

**Why the negative result is informative, not merely null.** Three convergent analyses locate
the residual signal in demographics: matched subsampling collapses it, age+sex residualization
removes it, and confound-conditioned permutation shows it is not independent of age/sex. This is
a positive, mechanistic explanation of *why* sophisticated models do not help here — the
discriminative structure available at this sample size is largely demographic, and additional
modality-fusion capacity has nothing further to exploit. Consistent with this, no fusion variant
exceeds EEG-alone and, in the missing-modality experiment, eye-tracking contributes nothing
(EEG-only ≥ EEG+eye).

**The signal is real where the confound is not.** On the independent ds003478 cohort, where the
age/sex confound is weak, EEG predicts depression well above demographics and survives
residualization and within-stratum permutation. This rules out the trivial reading that EEG is
simply uninformative; rather, whether EEG-based screening *appears* to work depends critically on
whether the evaluation controls for demographic structure — precisely the methodological point
of the paper.

**Contribution and positioning.** We do not claim accuracy superiority for NeuroFormer; we claim
(i) a unified architecture supporting one-to-three asynchronous modalities with a demonstrated
mask-based missing-modality mechanism, and (ii) a rigorous evaluation protocol that, applied
uniformly, yields an honest account of achievable performance. Evaluating a proposed model under
the same leakage-free scrutiny as its baselines — and reporting that it does not beat a demographic
null at N ≈ 90 — is, we argue, more useful to the field than another leaderboard number that would
not survive the same scrutiny.

**Clinical implications.** These results counsel caution against deploying multimodal EEG screening
on the basis of within-cohort accuracy alone. Reported performance should be benchmarked against a
demographic baseline and validated out-of-distribution before any screening claim; on the present
cohort a model that "achieves 0.7 AUC" is not clinically meaningful once age and sex are accounted
for.

**Limitations and future work.** The primary limitation is statistical power: N = 92 (20 positive)
yields wide intervals, and a genuine but small multimodal effect could be undetectable here — our
claim is the *absence of evidence for separation from demographics at this scale*, not proof of no
effect. The MIPDB cohort is also a complete-case, research-grade sample (selection bias), and
behaviour was unavailable. Future work should (a) repeat this protocol at substantially larger N
with prospective, demographically balanced sampling; (b) test the missing-modality architecture in
settings where modalities are genuinely and informatively missing; and (c) extend the
confound-controlled benchmark to additional cohorts to map where EEG-based screening does and does
not carry demographics-independent signal.
