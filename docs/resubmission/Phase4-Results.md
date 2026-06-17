# Phase 4 Results — External validation on HBN (well-powered)

**Question (R1.3, R3.2, R3.5):** does the within-MIPDB frozen-LaBraM EEG signal
generalize to an independent cohort?

## Setup
- **HBN-EEG**, OpenNeuro ds005505–ds005515 (Releases 1–11), **RestingState**,
  harmonized to the MIPDB preprocessing (0.1–75 Hz, 60 Hz notch, 200 Hz, µV, ±800
  clip, QC). 128-ch EGI, Cz ref — same net family.
- **N = 1,980** subjects with RestingState + labels (downloaded all releases).
- Labels: CBCL bifactor scores in `participants_all.tsv` (`p_factor` general +
  attention / internalizing / externalizing), **median-split** for AUC.
- Frozen LaBraM, mean+std pooled per subject; subject-level CV.

## Result (well-powered, N=1,980)

| HBN target (within-HBN) | regression r | median-split AUC (±CI) |
|---|---|---|
| **p_factor (general)** | 0.002 | **0.497 ± 0.006** |
| attention | 0.044 | 0.527 ± 0.008 |
| internalizing | 0.093 | 0.551 ± 0.006 |
| externalizing | 0.026 | 0.517 ± 0.006 |
| transfer MIPDB(any-DX)→HBN | 0.025 | 0.508 |

## Conclusion — no out-of-distribution transfer
At well-powered scale, **frozen LaBraM RestingState embeddings do not predict HBN
psychopathology factors**: the general p_factor is at chance (AUC 0.497), and the
others are at most marginally above (internalizing AUC 0.55 / r 0.09 — significant
only because N is large, practically negligible). The within-MIPDB signal (0.66)
**does not generalize** to this cohort.

> **Important correction to the preliminary run.** An earlier N=128 (Release 1)
> result suggested ~0.59 AUC. That was **small-sample noise** — it vanished at
> N=1,980. This is exactly why we scaled, and it is reported transparently.

## Interpretation (candid)
The null is real but the OOD test is stringent and differs from MIPDB on two axes:
1. **Label construct** — clinical DSM diagnosis (MIPDB) vs CBCL questionnaire
   factors (HBN). Different targets.
2. **Paradigm** — MIPDB used the QC-selected (often task) recording; HBN used
   **RestingState only**, which may carry less transdiagnostic signal.
Frozen embeddings may also fail to *linearly* expose signal that end-to-end
fine-tuning could surface (the NeurIPS-2025 HBN-EEG Challenge reports modest
above-chance p-factor prediction with supervised/fine-tuned models). We attempted
LaBraM fine-tuning but did not complete it due to cluster CUDA-environment issues
(see Limitations); it remains the key follow-up.

## What this means for the paper
- **Within-cohort:** foundation-model EEG carries transdiagnostic signal
  (MIPDB 0.66) and beats hand-crafted EEG — the positive contribution.
- **External:** that signal does **not** generalize out-of-distribution under this
  stringent, well-powered test — a rigorous, well-powered negative result that tempers any
  generalization claim and quantifies the within-cohort↔OOD gap.
- Combined with the multimodal null (Phase 2), the paper is a **rigorously
  evaluated, cautionary proof-of-concept**: it answers the reviewers' rigor and
  generalization concerns directly, even though the headline is sober.

## Limitations / future
- Fine-tuning LaBraM end-to-end on HBN (the lever most likely to surface OOD
  signal, per the challenge literature) is the priority follow-up; blocked here by
  an Alliance torch↔CUDA module mismatch, not by method.
- A matched-label / matched-paradigm OOD design would be a fairer transfer test.

## Artifacts
`results/phase4_hbn_all.csv` (N=1,980); `results/phase4_hbn.csv` (N=128, superseded).
