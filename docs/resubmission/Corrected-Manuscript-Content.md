# Corrected Manuscript Content — NeuroFormer (honest, reproducible)

> Ready-to-paste replacements for the manuscript, built on numbers produced by
> `experiments/neuroformer_baselines.py` + `experiments/robustness_3mod.py` under ONE
> leak-free, subject-level, repeated (5×5) stratified CV. Every figure here is
> reproducible from committed code + data. `[[ ]]` = fill from the Table 3 / full run.

## The corrected headline (honest — NEGATIVE / reproducibility result)
> ⚠️ Built on the **8-seed** leak-free run (`nf_seed_stability.csv`, `confound_adjusted.csv`).
> The single-seed 0.733 / +6.9 pp figures were **NOT robust** and have been retracted from our
> own framing. Do not use them.

- **NeuroFormer (3-modality), leak-free, pooled out-of-fold over 8 seeds: ROC-AUC = 0.617 ± 0.019**
  (the previously-shown 0.733 was a single lucky seed). A **pre-declared 5-config regularization
  sweep** (`nf_regularize.csv`) confirms this is **not an under-regularization artifact**: the
  best config is the full-capacity baseline (0.656 ± 0.032 over 4 seeds), and *reducing capacity
  or adding regularization only makes it worse* (down to 0.52). NeuroFormer's honest ceiling on
  this cohort is ≈0.62–0.66.
- **NeuroFormer does NOT outperform simpler baselines**: Low-rank Multimodal Fusion = 0.708,
  GMU = 0.669 — both **above** NeuroFormer's 0.617.
- **No model reliably beats demographics.** Age+sex alone = **0.717**, *higher than every model*.
  Confound-adjusted increment of NeuroFormer over age+sex = **−0.7 ± 1.7 pp** (`robust_positive
  = False`) → the model adds **no significant signal beyond age and sex** on this cohort.
- The original **0.912/85.1% was not reproducible** (window-level leakage + an incorrectly-
  balanced 63/63 cohort; true N≈102, ~20 % positive, unbalanced). We report the rigorous numbers
  and explain the discrepancy openly.
- **Therefore the honest contribution is a leakage-controlled re-evaluation / reproducibility
  study, not a method "win."** (Final framing pending corresponding-author decision — see
  `Supervisor-Decision-Memo.md`.)

### Metric note — why pooled out-of-fold AUC (answers R3.3/R3.4 on evaluation rigor)
At N≈102 with ~20 % positives, each held-out fold contains only ~4 positive subjects, so a
**per-fold** AUC is extremely high-variance and downward-biased, and averaging 25 such values
understates performance. We therefore report the **pooled out-of-fold AUC**: within each of the
5 repeats every subject is held out exactly once, so we compute one AUC per repeat over all N
subjects and average across the 5 repeats (mean ± 95 % CI). This is the recommended estimator
for small samples [Forman & Scholz 2010; Airola et al. 2011] and is applied **identically to
every model**, including all baselines. The unstable per-fold averages are reported alongside
in a supplement column (`auc_perfold`) for full transparency. (This single methodological choice
is itself a substantive answer to the reproducibility concerns; the run is also seeded and
`cudnn`-deterministic, so the tables are byte-reproducible.)

---

## Corrected Table 2 — baselines (leak-free 5×5, **pooled out-of-fold** ROC-AUC) — **[[fill from job 16728021]]**
Headline column = pooled-OOF AUC; per-fold column kept for transparency.

| Model | Pooled-OOF ROC-AUC |
|---|---|
| SVM-RBF (concat) | 0.494 |
| RandomForest (concat) | 0.562 |
| MLP early-fusion | 0.543 |
| LogReg (concat) | 0.538 |
| Late-fusion ensemble | 0.575 |
| 1D-CNN / TCN (temporal) | ~0.61 |
| **NeuroFormer (proposed, 8-seed)** | **0.617 ± 0.019** |
| Tensor Fusion Network | 0.623 |
| GMU (gated fusion) | 0.669 |
| Low-rank Multimodal Fusion | **0.708** |
| **age + sex (demographic confound)** | **0.717 — beats every model** |

*Numbers from the deterministic 5×5 run + 8-seed NeuroFormer (`table2_baselines.csv`,
`nf_seed_stability.csv`). Cross-modal Transformer is unstable (0.50–0.69 across runs; report as
"did not converge reliably"). The key, uncomfortable observation: **the age+sex predictor (0.717)
is the strongest model in the table** — on this cohort the recoverable signal is largely
demographic, and no neural model reliably exceeds it.*

## Corrected Table 3 — modality ablation (NeuroFormer, leak-free, **pooled-OOF**)
Seven non-empty modality subsets × {without, with auxiliary reconstruction loss}, leak-free 5×5 CV,
pooled out-of-fold ROC-AUC (±95% CI). `Δaux` = AUC(+aux) − AUC(−aux); p = paired Wilcoxon over folds.

| Modality subset | AUC (−aux) | AUC (+aux) | Δaux (pp) | p |
|---|---|---|---|---|
| EEG             | 0.649 ± 0.042 | 0.641 ± 0.026 | −0.8 | 0.563 |
| Eye             | 0.530 ± 0.025 | 0.521 ± 0.017 | −0.9 | 0.977 |
| Beh             | 0.508 ± 0.007 | 0.504 ± 0.023 | −0.4 | 0.830 |
| EEG + Eye       | 0.674 ± 0.041 | 0.645 ± 0.039 | −2.9 | 0.968 |
| EEG + Beh       | 0.673 ± 0.044 | 0.670 ± 0.055 | −0.3 | 0.444 |
| Eye + Beh       | 0.540 ± 0.013 | 0.520 ± 0.024 | −2.0 | 0.183 |
| **EEG + Eye + Beh (full)** | **0.662 ± 0.050** | 0.627 ± 0.049 | −3.5 | 0.346 |

*Numbers from `results/table3_ablation.csv` (per-fold in `perfold_table3.csv`), single production seed,
same leak-free 5×5 protocol as Table 2.*

**Two honest negative findings, both confirmed by the data:**
1. **The auxiliary reconstruction loss does NOT help.** Every `Δaux` is negative (−0.3 to −3.5 pp) and
   **none is statistically significant** (all p ≥ 0.18). **Drop the "+4.8 pp aux" claim entirely** and
   report the ablation as a negative result.
2. **EEG-alone ≈ full multimodal.** EEG alone reaches 0.649 vs. the full three-modality model 0.662
   (+1.3 pp); the best subset is EEG+Eye / EEG+Beh (~0.67), and adding the third modality does *not*
   improve over the best pair. Eye-alone (0.530) and behaviour-alone (0.508) are at chance, so **no
   single non-EEG modality is informative** and there is **no reliable multimodal synergy** — state this
   plainly rather than overclaiming fusion gains. (Every subset stays in 0.50–0.67, below the age+sex
   demographic baseline of 0.72.)

---

## Corrected Abstract (draft — reproducibility/negative framing)
> Psychiatric disorders are widely underdiagnosed, and machine-learning screening from passive
> signals is an active area — but small clinical cohorts make such results fragile and prone to
> optimistic bias. We revisit a multimodal EEG + eye-tracking + behaviour screening model
> (NeuroFormer) on the public MIPDB cohort (complete-case N≈102, ~20 % positive, unbalanced)
> under a **strict subject-level, leakage-free, repeated stratified cross-validation** protocol
> with a **pooled out-of-fold AUC averaged over multiple seeds**. Under this rigorous evaluation,
> a previously-reported ROC-AUC of 0.91 is **not reproducible**: we trace it to window-level
> information leakage and a mis-stated balanced cohort. Corrected, NeuroFormer reaches **ROC-AUC
> 0.62 ± 0.02**, **below** strong but simpler baselines (low-rank multimodal fusion 0.71), and —
> most importantly — **no model reliably exceeds a plain age+sex demographic predictor (0.72)**;
> a confound-adjusted test shows **no significant signal beyond demographics** (−0.7 ± 1.7 pp).
> Auxiliary reconstruction losses and multimodal fusion provide no reliable gain over EEG alone.
> We present this as a cautionary, fully-reproducible benchmark: it quantifies how leakage and
> cohort mis-specification inflate small-N psychiatric ML, provides honest confound-controlled
> baselines, and offers a rigor checklist for the field. Code and per-seed results are released.

*(All 85.1 % / 0.912 figures removed throughout; claims reframed from method superiority to
leakage-controlled re-evaluation. The regularization sweep is in: NeuroFormer's ceiling is
≈0.62–0.66, **below** baselines and demographics — "below baselines" wording stands, no softening
needed.)*

## Corrected cohort description (IMPORTANT — was misstated)
- The public MIPDB `DX_Status` is **101 control / 22 (=1) / 3 (=2)** → **25 positive of 126**.
- The all-three-modality complete-case used here is **N = 102: 21 positive, 81 control**.
- **The cohort is UNBALANCED (~20% positive)** — *not* the "63 diagnosed / 63 matched controls,
  50:50" stated in the manuscript. Correct this everywhere it appears (§III.A.1, abstract).
- Because of the imbalance, report **class-weighted training + balanced metrics (ROC-AUC,
  balanced accuracy)**, not raw accuracy (raw "85.1%" was tied to the incorrect balanced
  cohort). The corrected pipeline already uses class-weighted loss + ROC-AUC.

## External validation (HBN) — honest handling (recommended)
The **full 3-modality NeuroFormer cannot be externally validated**: no public cohort has
EEG + eye-tracking + behaviour under matching protocols (HBN is **EEG-only**). Therefore:
- **Keep external validation as a limitation** (as the original draft honestly did): *"No
  external validation of the full multimodal model was possible, as no public dataset
  provides all three modalities under consistent acquisition."*
- **Add one Discussion sentence** noting that even a strong EEG representation (a pretrained
  EEG foundation model) **fails to transfer to the large Healthy Brain Network cohort**
  (general-psychopathology ROC-AUC ≈ 0.50, N≈1,980) — reinforcing the caution about
  out-of-distribution generalization. *Attribute this clearly to an EEG-only foundation-model
  analysis, not to NeuroFormer itself.* (`results/phase4_hbn_all.csv`.)
- **Do NOT** write "we externally validated NeuroFormer on HBN."

## Corrected contributions (honest — reproducibility study)
1. **A leakage-controlled re-evaluation that overturns an over-optimistic prior result.** We
   identify window-level leakage + a mis-stated balanced cohort behind a reported 0.91 AUC, and
   show that under strict subject-level, multi-seed, pooled-OOF evaluation the true performance
   is far lower (NeuroFormer 0.62 ± 0.02).
2. **A rigorous, reproducible small-sample protocol** — subject-level repeated stratified CV +
   pooled out-of-fold AUC + multi-seed mean±std, applied identically to every model — that we
   recommend as a template for small-N psychiatric ML.
3. **The decisive confound finding**: on MIPDB a plain **age+sex predictor (0.72) outperforms
   every model**, and the proposed model adds **no significant signal beyond demographics**
   (−0.7 ± 1.7 pp over 8 seeds). This reframes "screening signal" as largely demographic on this
   cohort — an important, honest caution.
4. **Negative findings on architecture choices**: auxiliary reconstruction loss gives no gain;
   multimodal fusion does not beat EEG alone — reported plainly, not buried.
5. **Honest generalization boundary** (EEG-FM representation ≈ chance on HBN, N≈1,980) and
   interpretability caveats.

> If the corresponding author prefers to keep a *method* paper, the only honest version is
> "a compact multimodal transformer that performs **on par with** simple fusion baselines under
> rigorous evaluation," with contributions 2–5 as the substance — but it still cannot claim to
> beat baselines or demographics. Recommend the reproducibility framing above.

## Key edits throughout (checklist)
- [ ] Replace **all** 85.1% / 0.912 with the corrected numbers (abstract, intro, Table 2,
      conclusion).
- [ ] Methods §III.A.3: state the protocol is **repeated (5×5) subject-level** CV (matches code).
- [ ] Cohort: the all-3-modality complete-case is **N≈102 (21 positive)** — and the public
      `DX_Status` is **~20% positive, not 63/63** (correct the cohort description honestly).
- [ ] Reframe the eye-dominant claim — leak-free, **no single modality is informative alone**;
      the gain is the fusion.
- [x] Aux-loss claim: **resolved** — Table 3 shows aux loss does NOT help (all Δaux<0, n.s.); claim dropped.
- [ ] Add a short "Reproducibility & corrected evaluation" paragraph explaining the
      leakage-controlled protocol (this is what makes it defensible and answers R3.3/R3.4).
- [ ] Soften every clinical/deployment claim to proof-of-concept (R3.2).

## How this answers the reviewers (bonus)
The corrected, leak-free reconstruction is *itself* the response to R3.3 (fold sensitivity:
now 5×5 with CIs), R3.4 (overparameterization: compact 2/2-layer config), and R3.5
(generalization: HBN null reported). The honesty converts the hardest concerns into the
paper's rigor.
