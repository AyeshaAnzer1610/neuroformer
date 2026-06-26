# Memo to Prof. Bais — NeuroFormer resubmission (decision needed)

**From:** Ayesha Anzer  **Re:** Access-2026-19478 (NeuroFormer) revision
**Date:** 2026-06-26  **Status:** results final; **framing decision needed before resubmission**

---

## TL;DR
Rebuilding the evaluation rigorously (leakage-free, subject-level, multi-seed) shows the
original **0.91 AUC is not reproducible**, and that **no model on this cohort reliably beats a
simple age+sex predictor**. The honest paper is a **leakage-controlled re-evaluation /
reproducibility study**, not a "NeuroFormer is SOTA" paper. I need your decision on direction.

## What we did (answering the reviewers' rigor concerns)
- One **leak-free, subject-level, repeated 5×5 stratified CV** protocol, identical for every model.
- **Pooled out-of-fold AUC** (the stable small-N estimator), **averaged over 8 random seeds** for
  the proposed model (it is seed-fragile).
- A **confound-adjusted** test: does the model add signal beyond age+sex?
- All numbers produced by released code (`experiments/neuroformer_baselines.py`,
  `nf_seed_stability.csv`, `confound_adjusted.csv`).

## What we found (robust, honest)
| Result | Value |
|---|---|
| Original reported | 0.912 AUC / 85.1 % acc — **not reproducible** |
| NeuroFormer, leak-free, 8-seed pooled-OOF | **0.617 ± 0.019** |
| Low-rank multimodal fusion (baseline) | 0.708 |
| GMU (baseline) | 0.669 |
| **age + sex alone** | **0.717 — best in the table** |
| NeuroFormer increment beyond age+sex | **−0.7 ± 1.7 pp** (not significant) |
| Auxiliary reconstruction loss | no gain (n.s.) |
| Multimodal vs EEG-alone | no gain |

**Root cause of the old 0.91:** window-level information leakage in the prior protocol **plus**
a mis-stated cohort (manuscript said balanced 63/63; the real complete-case sample is N≈102,
~20 % positive, **unbalanced**).

> One experiment still running: a **regularization sweep** (`nf_regularize.csv`) to check whether
> 0.62 is partly an overfitting artifact. Best case it lifts NeuroFormer to ~0.70 — **on par with
> baselines but still not above the 0.72 demographic predictor.** It cannot change the core
> conclusion.

## The decision (yours)
1. **Reframe as a reproducibility / negative-results paper** *(my recommendation).* Contribution =
   exposing the leakage+cohort error, honest confound-controlled benchmarks, and a rigor checklist
   for small-N psychiatric ML. Reuses all our work; honest; publishable (incl. IEEE Access, which
   values reproducibility). Draft already prepared (`Corrected-Manuscript-Content.md`).
2. **Keep a method paper, honestly downgraded** — "a compact multimodal transformer on par with
   simple fusion baselines under rigorous evaluation." Weaker, and reviewers will ask why it
   doesn't beat age+sex. Not recommended.
3. **Withdraw / defer** and redirect effort to the new-direction paper (EEG foundation models).

## Why I'm flagging this rather than just resubmitting
The previous headline does not survive honest evaluation. Resubmitting a "NeuroFormer wins"
narrative would repeat the original error and risks our credibility with the editor. I've kept
everything reproducible so whichever way you choose is defensible. Ready to finalize the manuscript
in your chosen framing today once you decide.

## Attached / ready
- `Corrected-Manuscript-Content.md` — corrected abstract, Table 2, contributions (honest framing).
- `IEEE-Response-to-Reviewers-Formal.md` — all 22 comments answered, aligned to the honest results.
- `Baseline-References.md` — baselines + metric (Forman & Scholz, Airola) citations.
- Results: `table2_baselines.csv`, `nf_seed_stability.csv`, `confound_adjusted.csv`,
  `table3_ablation.csv`, (`nf_regularize.csv` pending).
