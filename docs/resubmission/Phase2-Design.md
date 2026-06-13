# Phase 2 Design — Missing-modality fusion (alignment first)

Goal: one model that fuses **LaBraM EEG + eye-tracking + behaviour** and works on
**any subset** of modalities (modality dropout + availability token + set-based
fusion). Retires R3.6 (fusion premise), R3.9, R3.10 (real-world missingness),
R1.1. This doc pins down **alignment and every methodological consideration**
*before* coding, so we don't rediscover problems mid-run (the Phase 1 lesson).

## 1. Cross-modal alignment — the central decision

The three modalities are **not synchronized**: EEG, eye, and behaviour come from
**different paradigms/recordings** per subject (EEG = QC-selected recording; eye =
up to 10 task blocks; behaviour = task blocks). We do **not** pretend temporal
cross-modal attention. Alignment is at the **subject level**:

- **One feature vector per modality per subject**, keyed by subject ID (`A00…`).
- Within a modality, aggregate across blocks/windows to one vector (mean-pool,
  matching the EEG window-pooling). Eye/behaviour: mean across that subject's
  blocks. (We accept within-modality paradigm heterogeneity — same honesty as
  EEG using one recording; documented, not hidden.)
- Join all modalities + `labels.csv` on subject ID. This is the *only* join key.

## 2. Coverage / missingness (measured, N of 126)

| set | N | positives (any DX) |
|---|---|---|
| EEG | 118 | 23 |
| eye | 107 | 22 |
| behaviour | 111 | 21 |
| **all three (complete-case)** | **~102** | ~21 |

Missing pattern is heterogeneous (19 no-eye, 15 no-beh, 8 no-EEG) → genuine
missing-modality regime. We build an explicit **availability mask** `[n_subj, 3]`.

## 3. Two evaluation regimes (report both)
1. **Complete-case (N≈102):** every subject has all 3. Clean reference / upper
   bound; shows fusion vs EEG-alone when nothing is missing.
2. **Missing-modality (N≈118, union of available):** the contribution. Model sees
   only present modalities (mask); trained with **modality dropout**. Plus a
   **simulated-missingness** sweep at test time → graceful-degradation curves.

## 4. Leakage / CV discipline (carry over from Phase 1)
- **Subject-level** splits only (a subject is never in train and test).
- Per-modality **standardization fit on train folds only** (sklearn Pipeline).
- Same **5×10 repeated stratified CV**, identical splits across model variants
  (paired comparison). Class-balanced; **ROC-AUC primary** (21 positives → plain
  accuracy misleads), balanced-accuracy secondary, 95% CI.
- N≈100 is small → **no heavy hyperparameter tuning** (avoids optimistic bias);
  if any tuning, nested CV. Report CIs honestly.

## 5. Feature scales / dims (must normalize before fusion)
- EEG LaBraM: 200-d (mean) or 400-d (mean+std), embedding-scale.
- Eye: 11-d physical units (pupil, gaze x/y, latency, gaze vector).
- Behaviour: 6-d (RT stats, accuracy).
Wildly different → **standardize each modality separately**, then project each to a
common token dim for set-based fusion.

## 6. Baselines the fusion must beat / be compared to
- **EEG-alone (LaBraM mean+std): AUC 0.660** (Phase 1) — the bar.
- Eye-alone, behaviour-alone (per-modality contribution).
- **Naive concatenation + logreg** — Phase 1 showed this *hurt* (0.48); the foil
  that motivates learned fusion.

## 7. Architecture (kept deliberately small for N≈100)
Per modality: standardize → linear projection to `d` (e.g. 32) → **modality token**
+ a learned **availability embedding** (present/absent). **Masked attention/mean
pool** over present tokens → small MLP head. **Modality dropout** during training
(randomly hide modalities) so one model serves any subset. Start simple; a big
transformer would overfit. Compare against the baselines above.

## 8. Feature source decision (eye + behaviour)
The original Multisource repo has extracted **eye** (`processed_asd_features.csv`,
11 feats × ~10 blocks/subj) and **behaviour** (`processed_features_behavioral_
features.csv`, 6 feats) — *not* re-derivable as raw here yet.
- **Now:** reuse these extracted CSVs to build + validate the fusion/alignment
  machinery (fast, unblocks the real contribution).
- **Rigor follow-up:** re-derive eye/behaviour from raw MIPDB (consistent QC, like
  EEG) before final results, if provenance of the old features is a concern.

## 9. Reproducibility
Per-subject aligned arrays + availability mask cached and committed; results
writeup like `Phase1-Results.md`; embeddings/bundles backed up off-site.

## Build order
1. `src/assemble_multimodal.py` — load eye/beh CSVs (→ per-subject mean) + EEG npz,
   align on ID, emit `{modality: {subj: vec}}` + availability mask. **(alignment core)**
2. `src/fusion_model.py` — masked set-fusion + modality dropout (torch), small.
3. `src/eval_fusion.py` — CV, both regimes, degradation sweep, vs baselines.
4. `Phase2-Results.md`.
