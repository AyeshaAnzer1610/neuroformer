# Phase 2 Results — Missing-modality fusion (no multimodal accuracy gain)

**Question:** does fusing EEG + eye + behaviour beat EEG-alone, and can one model
handle any subset of modalities? (R3.6 fusion premise, R3.9 3-modality claim,
R3.10 real-world missingness.)

## Setup
- Per-subject features, aligned on ID (subject-level; modalities are
  cross-paradigm/asynchronous — no faked temporal sync).
  - EEG: frozen LaBraM, mean+std pooled (400-d).
  - Eye: 11 SMI features, re-derived from raw + **validated** vs the original
    (r≈1.00). Behaviour: 6 RT/accuracy features, re-derived + validated.
- Complete-case N=104 (22 cases); missing-modality union N=120 (eeg=118, eye=109,
  beh=113). Subject-level 5×10 repeated stratified CV, class-balanced, ROC-AUC.

## Result (complete-case ROC-AUC, 95% CI)

| feature set / fusion        | ROC-AUC |
|-----------------------------|---------|
| **EEG-alone (LaBraM)**      | **0.66** |
| eye-alone                   | 0.51 (chance) |
| behaviour-alone             | 0.40 (below chance) |
| naive concatenation         | 0.645 |
| neural attention fusion     | 0.59 |
| stacking late fusion        | 0.47 |

Missing-modality cohort: fusion 0.55–0.57. Degradation (train on all, drop one at
test): **drop EEG → 0.47 (chance)**; drop eye → 0.60; **drop behaviour → 0.61
(removing it helps)**.

## Conclusions
1. **No fusion strategy beat EEG-alone; all degraded it.** Tried early
   concatenation, attention-based set fusion, weighted and stacking late fusion —
   each ≤ 0.66, validated as correct on synthetic signal (they reach ~1.0 there).
2. **Eye and behaviour carry no transdiagnostic signal in MIPDB** (individually
   ≈ or below chance). With only ~22 cases, adding them injects variance that
   outweighs non-existent complementary signal; the degradation sweep confirms
   predictive signal is essentially all in EEG.
3. **The missing-modality *mechanism* works** (one model, any subset, graceful
   degradation) and is deployment-relevant (8/20/16 subjects actually lack
   EEG/eye/behaviour) — but its value here is **operational, not accuracy**.

## Implications for the paper (framing)
- Lead with the **FM-EEG contribution** (Phase 1): foundation-model EEG ≫
  hand-crafted EEG; this is the real, defensible result.
- Present Phase 2 as a **transparent multimodal analysis** that directly answers the
  reviewers' fusion skepticism: in this cohort the extra modalities don't help,
  so we **soften the 3-modality accuracy claim** (R3.9) and report it straight
  (R3.6). The architecture handles real-world missingness (R3.10) even though it
  doesn't raise accuracy here.
- **Do not** claim multimodal fusion improves screening on MIPDB — the data
  doesn't support it. "Proof-of-concept with honest limitations" reads as
  credibility.

## Possible future levers (not pursued now)
- **Richer eye features.** The 11 summary stats (mean/std pupil, gaze, latency)
  are crude. Oculomotor biomarkers from the transdiagnostic literature
  (fixation durations, saccade amplitude/velocity, scanpath entropy, fixation
  maps) may carry signal these averages wash out — a Phase 5 refinement.
- **External eye-tracking cohort** (Phase 4) with a disorder where eye-tracking
  is known-informative (e.g., ADHD), to validate the eye pathway where signal
  exists.
- Larger N (HBN) would let learned fusion overcome the small-sample variance.

## Artifacts
`results/phase2_fusion.csv`; code: `extract_eye_features.py`,
`extract_behaviour_features.py`, `assemble_multimodal.py`, `fusion_model.py`,
`eval_fusion.py`, `validate_features.py`.
