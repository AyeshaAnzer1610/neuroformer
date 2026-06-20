# Reviewer-objection register (TNSRE) — pre-empt rejection

Adversarial self-review of the planned "adaptation × sample-size crossover for EEG
foundation models" paper. Each item: the objection, the mitigation, and honest status
(✅ addressed · 🖥️ needs a cluster run · ⚠️ hard limitation → limitations section).

## CRITICAL (any one can sink the paper)

**C1. Single foundation model (LaBraM only).** "Is 'freeze beats adapt' a property of
LaBraM or of EEG-FMs in general? With one backbone this is an anecdote."
→ **Mitigation:** replicate the crossover on **≥2 backbones** — EEGPT (NeurIPS'24) and/or
CBraMod (ICLR'25), both on HuggingFace. If the pattern holds across FMs, the claim
generalizes; if not, that itself is the finding. **Status: 🖥️ needs a second backbone
wired in (the single biggest acceptance lever).**

**C2. Crossover confounds sample size with dataset/task.** MIPDB (small, any-DX) vs HBN
(large, p_factor) differ in cohort, task, AND label — so "FT wins at large N" could be a
dataset effect, not an N effect.
→ **Mitigation:** the PRIMARY crossover must be **within a single dataset** — subsample
**HBN at N = 150/300/500/1000/2000** (same task, same label, only N varies). MIPDB is then
a *separate* cross-cohort small-N anchor, not a point on the same curve. `crossover_sbatch.sh
hbn --max-subj N` already does this. **Status: 🖥️ run the within-HBN N-sweep (design-critical).**

**C3. "Freeze > fine-tune" is already known (LP-FT literature).** Kumar et al. 2022
("fine-tuning can distort pretrained features"), linear-probing-then-fine-tuning, etc.
→ **Mitigation:** position novelty precisely — *first systematic characterization of the
adaptation×N crossover for **clinical EEG** foundation models*, with the practical N\*
guideline + leak-free + external validation + calibration. Cite the LP-FT work explicitly;
claim the EEG-clinical-specific quantification, not the bare phenomenon. **Status: ✅
framing (must cite prior LP-FT work, do NOT claim the phenomenon is new).**

**C4. Fine-tuning was under-tuned ⇒ unfair comparison.** "Maybe full-FT loses only because
you used lr 3e-5 / 6 epochs / no regularization; tune it and it wins."
→ **Mitigation:** small **LR × epochs sweep for full-FT and LoRA** (e.g. lr ∈ {1e-5,3e-5,
1e-4}, early stopping); report the *best* adaptation config vs frozen. Same for LoRA rank
(ablation built). **Status: 🖥️ run the tuning sweep so the comparison is defensibly fair.**

**C5. Core single-cohort result is non-significant (equivalence).** frozen vs LoRA p=0.06.
→ **Mitigation:** the crossover (C2) must show a **significant** FT>frozen at large N and
frozen≥FT at small N; report per-N significance + Holm. If no significant crossover exists,
the honest paper is an equivalence/efficiency result — weaker for TNSRE (consider EMBC).
**Status: 🖥️ gated on the within-HBN crossover showing significance.**

## MAJOR

**M1. Weak / heterogeneous labels.** "Any-DSM-vs-none" lumps disorders; HBN p_factor
median-split is arbitrary.
→ Report **continuous** p_factor regression (Pearson r, already in `eval_hbn.py`) alongside
the median split; justify transdiagnostic screening; acknowledge in limitations. **Status:
✅ (regression reported) + ⚠️ (label heterogeneity is a real limitation).**

**M2. OcuFM and CIDA are unvalidated (no accuracy/transfer gain).** Including them as
"contributions" invites "why present methods that don't work?"
→ **Cut OcuFM from the core paper** (or one future-work sentence). **Reframe CIDA** not as a
method but as an **analysis** that *localizes the transfer barrier* (identity is removable
yet transfer stays at chance ⇒ barrier is domain/label shift, not identity) — a diagnostic
finding, not a proposed winner. **Status: ✅ scoping decision (de-scope OcuFM; CIDA→analysis).**

**M3. Statistical power.** Crossover cells at 3-fold×1-repeat are underpowered.
→ Use **5×5** for the key crossover points; report CIs + per-N tests. **Status: 🖥️ (bump
repeats on the decisive N points).**

**M4. Small primary cohort (N≈118, ~23 positives).** Variance, generalizability.
→ Subject-level repeated CV + CIs (have); external HBN (have); frame N≈118 as the *small-N
regime* the paper is explicitly about. **Status: ⚠️ inherent — own it in limitations; the
crossover turns "small N" from a weakness into the subject of study.**

## MINOR (address in text / supplement)

- **m1. Preprocessing/montage choices** (60 Hz notch for US data vs LaBraM's 50 Hz; QC gate;
  15 s windows): justify + cite LaBraM protocol; note any sensitivity. ✅ documented in code.
- **m2. Calibration** for a clinical claim: report ECE/Brier pre/post temperature scaling
  (raw ECE=0.138 measured). 🖥️ compute on the frozen headline (re-run with `--save-preds`).
- **m3. Reproducibility:** code public (GitHub), seeds fixed, leak-free protocol. ✅.
- **m4. Class imbalance:** class-weighted loss + balanced metrics (BAcc/AUC). ✅.
- **m5. Compute/fairness of "efficiency" claim:** report trainable-param counts + wall-clock
  per method (frozen head vs LoRA 2.88% vs full). ✅ params; 🖥️ add timing.

## Priority order to maximize acceptance
1. **C2** within-HBN N-sweep (clean crossover) — design-critical, runnable now.
2. **C1** second backbone (EEGPT/CBraMod) — biggest generality win.
3. **C4** full-FT/LoRA tuning sweep — fairness.
4. **C3/M2** framing: cite LP-FT; de-scope OcuFM; CIDA→analysis.
5. **m2** frozen-model calibration; **M3** bump repeats on decisive N.
