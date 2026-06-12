# Research Repositioning Strategy — NeuroFormer → next contribution

> Purpose: rethink the work "from scratch" to find a current literature gap our assets
> can own, targeting IEEE Access acceptance. Drafted from a 2025–2026 literature scan
> (sources at bottom). This is a thinking aid, not a commitment.

## Reality check on "SOTA" vs. IEEE Access

IEEE Access does **not** require beating everyone's accuracy. Its bar is: technically
sound, genuinely novel, rigorously validated, clearly written. A fragile state-of-the-art
number on N=126 will *lose*; a well-scoped novel method with honest external validation
will *win*. So we optimize for **a clear, current gap + rigor**, not a leaderboard.

## The three open gaps (2025–2026)

1. **Missing-modality fusion is absent in neurophysiological psychiatric screening.**
   Hot in medical imaging/EHR (DrFuse, M3Care, modality-dropout+contrastive, learnable
   modality tokens) — but not in EEG+eye+behaviour mental-health work, which still
   assumes complete, clean recordings.

2. **EEG foundation models have not met multimodal psychiatric fusion.**
   LaBraM, EEGPT, BrainGPT, LEAD, HEAR (2024–2026) give transferable EEG features that
   thrive with limited labels and transfer across datasets — directly countering our
   small-N and weak-hand-crafted-EEG problems. Nobody has fused them with eye+behaviour.

3. **Calibration / uncertainty under dataset shift is nascent.**
   Emerging 2025 work asks models to flag low-confidence cases for clinician referral —
   near-absent in multimodal psychiatric screening.

Supporting theory bonus: a 2025 systematic review on **transdiagnostic eye-tracking
biomarkers of inattention** supplies the neurophysiological rationale reviewers wanted
and validates the eye-tracking-dominance finding.

## Three repositioning options

### Option A — Foundation-model + missing-modality transdiagnostic screen  ⭐ recommended
**One-line novelty:** *First unified model to fuse pretrained EEG foundation-model
representations with eye-tracking and behaviour under a missing-modality regime for
transdiagnostic mental-health screening, with cross-dataset external validation.*
- Fills Gaps 1 + 2 (+ optionally 3).
- Reuses: MIPDB data, the hierarchical-fusion idea, the ablation framework, much of the pipeline.
- New work: integrate an EEG foundation-model encoder; modality-dropout + availability-token
  training; set-based asynchronous fusion; external validation on HBN + an ADHD eye-tracking set.
- **Acceptance odds: high. Effort: high. Risk: medium** (foundation-model integration + HBN logistics).

### Option B — Calibrated, missing-modality benchmark/resource paper
**One-line novelty:** *A reproducible benchmark + calibrated, missingness-robust baselines
for multimodal (EEG/eye/behaviour) transdiagnostic screening across MIPDB + HBN.*
- Fills Gaps 1 + 3; IEEE Access likes resource/benchmark papers.
- Reuses: all baselines already implemented.
- New work: harmonize MIPDB+HBN, define the benchmark, add calibration metrics (ECE,
  selective prediction), missing-modality protocols.
- **Acceptance odds: high. Effort: medium-high. Risk: medium** (data harmonization heavy).

### Option C — Missing-modality methods paper on existing data only
**One-line novelty:** *A modality-dropout transformer with graceful degradation for
incomplete EEG/eye/behaviour psychiatric screening.*
- Fills Gap 1 only.
- Reuses: nearly everything; least new work.
- New work: modality-dropout training + simulated-missingness evaluation.
- **Acceptance odds: medium. Effort: low-medium. Risk: low** — but no external validation
  still leaves R1/R3's biggest ask unanswered, so weaker.

## Recommendation
**Option A**, with the calibration piece (Gap 3) added if time allows. It is the only
option that converts *every* reviewer weakness into a contribution and lands squarely in
two of the field's current gaps. If timeline/compute is tight, **fall back to a focused
Option A** (foundation-model EEG + missing-modality + HBN-only external validation),
deferring the ADHD-eye set and the full clinician study.

## What must be true to choose A (open questions)
- [ ] Compute available to run an EEG foundation model (GPU; or use released frozen embeddings).
- [ ] HBN access + which release(s); confirm EEG+behaviour usable now, eye-tracking status.
- [ ] Timeline vs. effort — is this a 2–3 month effort we can resource?
- [ ] Whether to keep the NeuroFormer name/brand or rename around the new contribution.

## How existing work carries forward (not wasted)
Pipelines, hand-crafted features (now a baseline vs. foundation-model features),
all 9 baselines, the subset-ablation methodology, subject-level CV protocol, and the
honest limitations framing all transfer. The rejected paper becomes the *backbone* of a
stronger one, not a throwaway.

---

### Sources (2025–2026 scan)
- Multimodal ML in mental health — survey of data/algorithms/challenges: arXiv:2407.16804
- Missing/incomplete-modality medical fusion: DrFuse arXiv:2403.06197; M3Care arXiv:2210.17292;
  "Missing-modality enabled multi-modal fusion architecture for medical data" (ScienceDirect S1532046425000255)
- EEG foundation models: LaBraM; EEGPT; BrainGPT arXiv:2410.19779; LEAD (Alzheimer's) arXiv:2502.01678;
  HEAR (heterogeneous electrodes) arXiv:2510.12515
- Transdiagnostic eye-tracking biomarkers of inattention — systematic review: BMC Psychiatry 2025, doi:10.1186/s12888-025-07415-w
- Calibration/uncertainty: "Clinically Calibrated ML Benchmarks for Multi-Disorder EEG" arXiv:2512.22656;
  "Uncertainty in Deep Learning for EEG under Dataset Shifts" bioRxiv 2025.07.09.663220
