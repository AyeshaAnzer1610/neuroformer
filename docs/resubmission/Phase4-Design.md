# Phase 4 Design — External validation on HBN (the make-or-break)

Goal: show the **frozen LaBraM EEG** pathway (the Phase 1 win) carries
transdiagnostic signal in an **independent cohort**, addressing R1.3 / R3.2
(clinical-screening realism) and R3.5 (does the benefit generalize).

## The label reality (decided the protocol)
HBN-EEG BIDS (OpenNeuro **ds005505**, Release 1; S3
`s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1`, 136 subjects, 5-21 yo,
128-ch EGI, BIDS/EEGLAB `.set`) ships labels in **`participants.tsv`** as
**CBCL-derived continuous psychopathology factors** (a bifactor model: a general
*p-factor* + specific factors) + demographics — **NOT** a categorical
`DX_Status`. So MIPDB (categorical any-DX-vs-none) and HBN (continuous CBCL
factors) are **not the same label**.

**Implication:** a pure "MIPDB classifier → HBN" transfer is confounded by label
mismatch. We therefore make **within-HBN replication** the primary external test.

## Protocol
1. **Primary — within-HBN replication (clean):** extract frozen LaBraM embeddings
   from HBN EEG (same pipeline/QC as MIPDB). Predict the HBN **general
   psychopathology factor** from the embeddings via the *same* protocol as Phase 1
   (per-subject pooled embeddings → CV). Binarize the p-factor (e.g. top tertile =
   "high symptom") for an AUC comparable to Phase 1, and/or regress it (report r).
   *Question:* does FM-EEG carry psychopathology signal OOD, near the MIPDB ~0.66?
2. **Secondary — transfer (honest caveat):** train on MIPDB (any-DX) embeddings,
   test on HBN (binarized p-factor). Report the honest drop; flag the label
   mismatch explicitly. Do **not** over-interpret.
3. **Harmonization:** same EGI ~128-ch net family as MIPDB → minimal. Same
   preprocessing (0.1-75 Hz, notch, 200 Hz, µV, clip, QC). HBN mains = **60 Hz**
   (US, CMI NY) — same notch as MIPDB. EEG files are EEGLAB `.set` (load_raw
   already supports `.set`).

## Considerations / risks
- **Paradigm:** HBN has resting + several tasks; pick a consistent paradigm (e.g.
  resting/RestingState) across subjects rather than QC-largest, to reduce
  cross-subject heterogeneity. Confirm task names from the BIDS filenames.
- **Channel layout:** confirm it's the GSN-HydroCel net InterpolatedLaBraM can map
  (set montage as in MIPDB). HBN may use a 128 vs 129 layout — verify the Cz/ref.
- **`participants.tsv` columns:** confirm the exact factor column name(s) (likely
  `p_factor` + specifics) before writing the label loader.
- **Size:** 103 GB full release. **Validate on a 3-5 subject sample first**
  (`download_data.sh hbn-subjects 3`) before the full sync / Globus.
- **N / balance:** 136 subjects; check the p-factor distribution and how many fall
  in the "high" bin — small-N caveats as in Phase 1/2.

## Build order
1. `download_data.sh hbn-subjects N` (sample) → inspect `participants.tsv` + a
   subject's EEG file paths/format. **(investigate first)**
2. Adapt `load_and_window` if HBN paths differ (BIDS `sub-*/eeg/*_eeg.set`);
   pick the resting paradigm.
3. `extract_eeg_embeddings.py --batch` on HBN → embeddings.
4. `eval_hbn.py` — within-HBN CV on the p-factor + the transfer test.
5. `Phase4-Results.md`.
