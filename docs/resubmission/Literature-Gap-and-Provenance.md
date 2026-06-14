# Literature Gap & Provenance — where each NeuroFormer v2 contribution comes from

For the resubmission: each architectural change is anchored to a recent **top-tier**
paper that established the technique, with the **gap** those papers leave and how
NeuroFormer fills it. Also states the **honest novelty boundary** (what we must NOT
claim, because close prior work exists).

## Closest prior work (the honest baseline for our novelty claim)
- **MTNet** — *Multimodal transformer fusing EEG + eye-tracking for mild depression*,
  Biomedical Signal Processing & Control, 2024 (ScienceDirect S1746809424010541),
  reports ~91.8% — **bimodal, depression-specific, complete data, no foundation
  model, no external validation, no calibration.**
- **MSNet** — *Multimodal Self-attention Network, EEG + eye-tracking depression*,
  ACM ETRA 2025 (10.1145/3715669.3723106) — same limitations.

**=> We must NOT claim "first to fuse EEG + eye-tracking."** They did it (bimodal,
single-disorder). **Our novelty is the *combination* none of them have** (below).

## Contribution → top-tier provenance → gap filled

### 1. Foundation-model EEG encoder (replaces hand-crafted 5-ch EEG)
- **Idea from:** LaBraM (ICLR 2024, spotlight); EEGPT (NeurIPS 2024); CBraMod
  (ICLR 2025) — pretrained EEG transformers giving transferable representations that
  thrive with limited labels.
- **Gap:** these FMs are benchmarked on standard BCI/clinical tasks; **none has been
  embedded in a multimodal (EEG+eye+behaviour) psychiatric-screening transformer.**
  Prior psychiatric multimodal work (incl. MTNet/MSNet) uses hand-crafted EEG.
- **We fill it:** LaBraM embeddings as NeuroFormer's EEG stream → EEG goes from weakest
  to strongest modality. *Answers R3.8, R2.1/2.4.*

### 2. Missing-modality fusion (modality dropout + availability token + set fusion)
- **Idea from:** DrFuse (AAAI 2024, arXiv:2403.06197) and M3Care (KDD 2022) — clinical
  fusion robust to missing modalities, in **EHR + medical imaging.**
- **Gap:** missing-modality fusion is established in EHR/imaging but **absent in
  EEG+eye+behaviour psychiatric screening**, which still assumes complete, clean
  recordings (MTNet/MSNet require both modalities present).
- **We fill it:** modality-dropout training + learned availability token + set-based
  (permutation-invariant) fusion → one model for any subset, graceful degradation.
  *Answers R3.6, R3.9, R3.10 (selection bias / real-world missingness).*

### 3. External / out-of-distribution validation on HBN
- **Idea from:** HBN-EEG FAIR dataset (bioRxiv 2024, 10.1101/2024.10.03.615261) and the
  **NeurIPS 2025 EEG Challenge** (eeg2025.github.io), which establishes predicting the
  four CBCL psychopathology factors (p-factor, internalizing, externalizing, attention)
  from HBN EEG as a recognized task; cf. Neurosense (p-factor prediction, 2025).
- **Gap:** prior EEG+eye psychiatric papers (MTNet/MSNet, and most of Table 1) report
  **single-cohort results with no external validation** — the #1 ask of reviewers R1/R3.
- **We fill it:** train transdiagnostic screen on MIPDB, externally validate the FM-EEG
  signal on HBN (independent cohort, different label/paradigm). Honest OOD drop.
  *Answers R1.3, R3.2, R3.5.* (Note: HBN p-factor prediction itself is an established
  task — we use it as the *external-validation vehicle*, not as a novelty claim.)

### 4. Calibrated, referral-aware prediction (NEW)
- **Idea from:** conformal selective prediction for clinical triage under distribution
  shift (Scientific Reports 2025/26, s41598-026-40637-w); robust conformal prediction
  for EEG classification (arXiv:2602.19483).
- **Gap:** calibration / uncertainty-aware abstention is emerging in single-modality
  clinical EEG but **near-absent in multimodal psychiatric screening.**
- **We fill it:** temperature scaling + conformal sets + risk–coverage/ECE; low-confidence
  cases abstain → "refer to clinician." *Strengthens the screening-tool framing (R3.2).*

## The one-sentence gap NeuroFormer v2 owns
*"Existing multimodal EEG/eye/behaviour psychiatric models (e.g., MTNet, MSNet) are
bimodal, single-disorder, hand-crafted, complete-data, single-cohort, and uncalibrated.
NeuroFormer is the first to unify a pretrained EEG foundation model, missing-modality
fusion, transdiagnostic screening, external (HBN) validation, and calibrated referral in
one framework."* — fills three open gaps (FM-EEG × multimodal; missing-modality in
neurophysiological screening; calibration under shift) that no single top-tier paper
addresses together.

## Architectural changes to NeuroFormer (what we edit in the paper)
1. EEG stream: hand-crafted features → **frozen LaBraM embeddings** (new EEG encoder front-end).
2. Fusion: add **modality-availability token** + **modality-dropout** training +
   **permutation-invariant set fusion** (replaces synchronized cross-attention framing).
3. Output: add a **calibration/selective-prediction head** (temperature/conformal).
4. Protocol: **leak-free subject-level** repeated CV + **external HBN** evaluation.

## Citations to add (top-tier, current)
LaBraM (ICLR'24), EEGPT (NeurIPS'24), CBraMod (ICLR'25), DrFuse (AAAI'24),
M3Care (KDD'22), MTNet (BSPC'24), MSNet (ETRA'25), HBN-EEG (bioRxiv'24 + NeurIPS'25
EEG Challenge), conformal clinical triage (Sci.Rep.'25/26), transdiagnostic
eye-tracking biomarkers review (BMC Psychiatry'25).
