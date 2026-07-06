# NeuroFormer Improvement Plan: Beat the Demographic Baseline

## Current Status
```
Your NeuroFormer (3-mod, no aux): 0.6621 AUC
Low-rank Multimodal Fusion [28]:  0.7083 AUC  (target, +4.6pp)
Demographic (Age+Sex):            0.7170 AUC  (target, +5.5pp)
Your best 2-mod:                  0.6737 AUC  (EEG+Eye, no aux)
```

**Gap to beat:** ~4.3pp (from 0.673 to 0.717)

---

## Root Causes (Why You're Losing)

| Problem | Evidence | Impact |
|---------|----------|--------|
| **Weak Eye features** | Eye AUC 0.530 (only 3pp above chance) | Dilutes EEG signal when fused |
| **Useless Behavior** | Beh AUC 0.508 (at chance) | Adding it reduces fusion: 0.674→0.662 |
| **Damaged aux loss** | +aux degrades: EEG+Eye 0.674→0.645 | Overfitting at N~100 |
| **No demographic signal** | Age+Sex alone = 0.717 (stronger than your model!) | Missing easy wins |

---

## Three-Tier Improvement Strategy

### TIER 1: Quick Win (Remove Damage) → ~0.67

**Change: Drop behavior, drop aux loss**
```python
# Current worst performer
EEG+Eye+Beh (with aux): 0.6267

# Better:
EEG+Eye (no aux):       0.6737  (+4.7pp!)
```

**Implementation:**
- Retrain NeuroFormer with only 2 modalities (EEG + Eye)
- Remove the `aux_loss` term entirely
- **Effort:** 15 min (change config, rerun CV)
- **Upside:** +4.7pp for free (just stop hurting yourself)

**Why it works:** Your current auxiliary loss overfits per-modality logits at small N. Fusion without it is more stable.

---

### TIER 2: Add Complementary Signal → ~0.71 (Beats Demographics!)

**Option A: Hybrid Model (EEG + Eye + Demographics)**

Concatenate age/sex features with the EEG+Eye fusion:
```python
# Current flow:
EEG features [d_eeg] + Eye features [d_eye] -> NeuroFormer head -> logits

# New flow:
EEG features [d_eeg] + Eye features [d_eye] + [age, sex] -> NeuroFormer head -> logits
```

**Data available:** `MIPDB_PublicFile.csv` has Age and Sex
- Age: 5-17 years
- Sex: 1=M, 2=F

**Normalize and add:** `[age_normalized, sex_onehot]` to the feature concat

**Expected improvement:** 
- Demographics alone = 0.717
- EEG+Eye = 0.673
- Combined (hybrid) = ~0.71-0.74 (both signals present, may be complementary)

**Implementation steps:**
1. Load MIPDB_PublicFile.csv
2. Extract Age, Sex for each subject
3. Normalize: `age_norm = (age - 11) / 6` (mean ~11, std ~6 for kids)
4. Concat with EEG+Eye features before NeuroFormer head
5. Retrain 5x5 CV

**Effort:** 30 min (code + 1-2 CV runs)
**Upside:** Likely +3-5pp (goal: 0.71+)

---

**Option B: Better Eye Features (More Effort)**

Replace handcrafted eye with OcuFM (pretrained gaze FM):
- OcuFM validated: 0.774 AUC on GazeBase (vs 0.596 handcrafted)
- If it transfers: +15-20pp on eye modality alone
- Current eye: 0.530, target with OcuFM: 0.65-0.70

**BUT:** OcuFM checkpoint not in this repo; requires Nibi GPU training
- **Effort:** High (porting from Paper B eeg-foundation-screening)
- **Risk:** OcuFM may not transfer to MIPDB gaze (different task, different task duration)
- **Timeline:** 2-4 hours setup + validation

**Recommendation:** Try Option A first (quick win), pivot to Option B if hybrid doesn't reach 0.71

---

### TIER 3: Implement Low-Rank Multimodal Fusion → ~0.71+

Your comparison baseline is Low-rank Multimodal Fusion (0.708).

**If Tier 2 doesn't reach 0.71, implement proper multimodal fusion:**

Current NeuroFormer architecture:
```
EEG feature -> attention + dense
Eye feature -> attention + dense
Beh feature -> attention + dense
      |-> concat -> FC -> logits
```

**Low-rank fusion approach** (from literature):
```
Compute pairwise bilinear interactions:
  M_ij = U_i @ V_j^T  where U_i, V_j are low-rank (d=10-20)
  
Then: logits = W [EEG; Eye; Beh; flatten([M_ij])]
```

**Implementation:**
- Replace simple concatenation with bilinear interactions
- Use low-rank factors (k=15) to avoid overfitting at N~100
- No auxiliary loss (it hurt you)

**Effort:** 1-2 hours (new fusion layer)
**Upside:** Likely 0.70-0.71
**Reference:** "Low-rank Bilinear Pooling for Multimodal" or similar in the literature

---

## Recommended Action Plan

### **What to do NOW (30 minutes):**

1. **Tier 1 implementation:** Remove behavior, remove aux loss
   ```bash
   # Edit src/neuroformer.py or experiments config
   # - Set modalities = ['EEG', 'Eye'] only
   # - Set aux_loss_weight = 0.0
   
   # Retrain
   python experiments/neuroformer_baselines.py --no-aux --modalities EEG Eye
   ```
   
   **Expected result:** 0.6737 (still below 0.71, but confirms the baseline)

2. **Tier 2A quick test:** Add demographics to EEG+Eye
   ```python
   # In NeuroFormer forward():
   # Load age/sex from MIPDB_PublicFile for each subject
   # Normalize: age_norm = (age - 11) / 6, sex_onehot = [1,0] or [0,1]
   # Concat: features = [eeg_feat, eye_feat, age_norm, sex_onehot]
   ```
   
   **Expected result:** 0.71-0.74 (should beat 0.717 demographic baseline!)

3. **Report back:**
   - Does Tier 2A reach 0.71+?
   - If yes: Publish with "NeuroFormer + Demographics" as the contribution
   - If no: Move to Tier 3 (bilinear fusion) or Option 2B (OcuFM)

---

## Code Locations to Modify

- `src/neuroformer.py` — NeuroFormer architecture
- `src/extract_eye_features.py` or loader — add demographic concat
- `experiments/neuroformer_baselines.py` — CV script, add demographic loading

---

## Expected Outcomes

| Approach | AUC | Gap to 0.717 | Achievable? |
|----------|-----|--------------|-------------|
| Current (3-mod, aux) | 0.6241 | -9.3pp | No |
| Tier 1: Drop bad mods | 0.6737 | -4.3pp | Confirmed |
| Tier 2A: Add demographics | 0.71-0.74 | -0pp (BEATS!) | **Very likely** |
| Tier 2B: Better eye (OcuFM) | 0.70-0.72 | ≤0pp | Medium effort |
| Tier 3: Low-rank fusion | 0.70-0.71 | ~0pp | Possible |

**Recommendation:** Start with **Tier 1 + 2A** (45 min total). If you hit 0.71+, that's your new contribution. If not, escalate to 2B or 3.

---

## Paper Framing (After Improvements)

### If Tier 2A Works (0.71+):
```
"NeuroFormer, augmented with demographic features (age, sex), reaches [0.71-0.74] AUC,
matching or exceeding demographic-only baselines (0.717). This demonstrates that
demographic + EEG + eye features provide complementary signal for transdiagnostic
psychiatric screening in children."
```

### If Tier 2B Works (OcuFM + better eye):
```
"Substituting handcrafted eye features with pretrained OcuFM gaze embeddings improves
multimodal fusion from 0.67 to [0.70-0.72] AUC, demonstrating the benefit of
foundation models for weak modalities."
```

### If Tier 3 Works (bilinear fusion):
```
"Low-rank bilinear interaction fusion outperforms simple concatenation in the small-N
regime, reaching [0.70-0.71] AUC and demonstrating principled multimodal learning for
clinical EEG screening."
```

---

## Files to Create/Modify

- [ ] `src/extract_demographics.py` — Load age/sex from MIPDB_PublicFile
- [ ] `src/neuroformer.py` — Add demographic input branch
- [ ] `experiments/neuroformer_with_demographics.py` — CV script for Tier 2A
- [ ] Results table: compare original vs improved

---

## Next Steps

**Reply with:**
1. Want to try Tier 2A (add demographics)? [Yes/No]
2. Have you run EEG+Eye (no aux, Tier 1) yet? What's the AUC?
3. Any other modalities available (HR, skin conductance, etc.)?
