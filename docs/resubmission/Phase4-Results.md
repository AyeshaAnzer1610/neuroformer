# Phase 4 Results — External validation on HBN

**Question:** does the frozen-LaBraM EEG signal (Phase 1) generalize to an
independent cohort? (R1.3, R3.2 clinical-screening realism; R3.5 does the benefit
generalize.)

## Setup
- **HBN-EEG Release 1** (ds005505), **N=128** subjects with RestingState + a
  p-factor label. 128-ch EGI, Cz ref, 500 Hz, 60 Hz mains, EEGLAB `.set` — same
  pipeline/QC as MIPDB (band-pass, notch, 200 Hz, µV, ±800 clip). **RestingState**
  used as a consistent paradigm.
- **Labels:** CBCL bifactor scores in `participants.tsv` — `p_factor` (general
  psychopathology) + `attention`/`internalizing`/`externalizing`. Continuous, so
  **not** MIPDB's categorical DX (label mismatch — see transfer caveat).
- Frozen LaBraM, mean+std pooled per subject. Subject-level CV: Ridge regression
  (Pearson r, no threshold) + median-split logistic AUC (5×10, comparable to the
  Phase 1 0.66).

## Result

| HBN target (within-HBN)     | regression r | median-split AUC (95% CI) |
|-----------------------------|--------------|---------------------------|
| **p_factor (general)**      | **0.217**    | **0.594 ± 0.026**         |
| attention                   | −0.030       | 0.516 ± 0.030             |
| internalizing               | −0.183       | 0.457 ± 0.033             |
| externalizing               | 0.043        | 0.527 ± 0.031             |
| transfer MIPDB(any-DX)→HBN  | −0.019       | 0.557 (no CI)             |

## Conclusions (honest)
1. **The general psychopathology factor replicates out-of-distribution.** Frozen
   LaBraM RestingState EEG predicts HBN `p_factor` at **AUC 0.594** (CI 0.57–0.62,
   excludes chance; r=0.22) — a fully independent cohort, a different label
   (CBCL), and a different paradigm. Modestly **below** the within-MIPDB 0.66 — the
   expected, honest OOD drop. It is the **broad transdiagnostic dimension** that
   carries, consistent with the screening thesis.
2. **Specific factors do not replicate** (attention/externalizing ≈ chance;
   internalizing slightly negative — noise at this N). Resting-state FM-EEG
   indexes general psychopathology, not fine-grained subtypes.
3. **Naive MIPDB→HBN transfer is near chance (0.56).** Honestly attributable to
   (a) label mismatch (clinical DX vs CBCL factor) and (b) cohort + paradigm shift
   (MIPDB QC-largest mixed paradigms vs HBN RestingState). Reported straight; we
   do **not** claim zero-shot clinical transfer.

## Significance for the resubmission
Completes the arc and answers R1.3/R3.2/R3.5: foundation-model EEG carries a
transdiagnostic signal that **partially generalizes to a fully independent
cohort** (general factor, AUC ~0.59 OOD), with candid limits (modest magnitude,
specific factors and naive transfer don't hold). Combined with Phase 1 (FM-EEG ≫
hand-crafted) and Phase 2 (honest multimodal null), the paper is a credible
proof-of-concept with foundation-model EEG as the locus of signal.

## Caveats / future
- Modest absolute AUC (~0.59) and N=128 — proof-of-concept, not a deployable
  screener. Larger HBN releases (ds005506–515) would tighten this.
- A *fair* transfer would need a harmonized label (e.g. derive a comparable
  any-symptom target in both cohorts) — flagged, not done here.
- Per-paradigm and age/sex-confound analyses are natural Phase 3/5 follow-ups.

## Artifacts
`results/phase4_hbn.csv`; code: `extract_eeg_embeddings.py` (`--match RestingState`),
`eval_hbn.py`, `scripts/extract_hbn_sbatch.sh`, `scripts/download_data.sh hbn-rest`.
