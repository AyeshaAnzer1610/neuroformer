# Reviewer-Response Worklog — NeuroFormer (IEEE Access, Access-2026-19478)

> **⚠️ SUPERSEDED FRAMING (2026-06-24).** The project is now split into **two separate
> papers**:
> - **(A) This IEEE Access resubmission** keeps the **original NeuroFormer paper
>   (0.912, eye/behaviour-dominant, hand-crafted 3-modality)** and answers the reviewers
>   *within that framework*. The authoritative point-by-point response is
>   **[IEEE-Response-to-Reviewers-Formal.md](IEEE-Response-to-Reviewers-Formal.md)**
>   (IEEE concern→response→action template). Reviewer asks needing new runs (R1.3
>   external, R3.3 robustness, R3.4 compact, R2.2 baselines) run on the **authors'
>   original pipeline** (not in this repo) and fill `[[ ]]`.
> - **(B) A new paper** carries the **foundation-model EEG** results (0.748 window-level,
>   multi-backbone, HBN external null) — drafted in [paper-draft.md](paper-draft.md).
>
> **The §0 numbers below (0.748 etc.) belong to paper (B), not the Access resubmission.**
> Treat this worklog as the (B)-paper evidence ledger + a cross-reference of which
> reviewer concern each (B) result happens to answer. For the actual resubmission, use the
> formal doc above.

> **Purpose.** A single living document of reviewer comments → evidence → response →
> status. Companion docs: [Response-to-Reviewers.md](Response-to-Reviewers.md);
> [reviewer-objections.md](reviewer-objections.md); [paper-draft.md](paper-draft.md).

**Status legend:** ✅ done · ✍️ text-only (write-up pending, no run needed) ·
🖥️ needs/awaiting a cluster run · ⚠️ honest limitation (own it) · ❓ author decision.

---

## 0. Headline evidence (the numbers every response leans on)

| result | value | source / status |
|---|---|---|
| hand-crafted EEG (Hjorth+spectral) | ROC-AUC **0.43–0.47** | ✅ `results/phase1_comparison.csv` |
| **frozen LaBraM + linear probe** | **0.66** | ✅ Phase 1 |
| **frozen LaBraM + window-level head** (5×10) | **0.748 ± 0.029** | ✅ `results/finetune_labram_frozen_r10.csv` — **the headline** |
| frozen LaBraM reproduced (5×5) | 0.739 ± 0.047 | ✅ `results/frozen_labram*.csv` |
| LoRA (r=8) | 0.701 ± 0.048 | ✅ `results/finetune_labram_lora_5x5.csv` |
| full end-to-end fine-tune | 0.686 ± 0.120 | ✅ overfits at N≈118 (variance 3×) |
| **frozen CBraMod** (5×5, robustness) | **0.602 ± 0.044** | ✅ `results/frozen_cbramod*.csv` |
| **frozen EEGPT** (5×5, robustness) | _pending_ | 🖥️ **job 16616052** running |
| multimodal fusion (best variant) | ≤ 0.66 (no gain over EEG) | ✅ `results/phase2_fusion.csv` |
| NeuroFormer end-to-end transformer | 0.668 | ✅ ties EEG-alone, < 0.739 |
| **external HBN** (general-psychopathology) | **0.497 ± 0.006**, N≈1,980 | ✅ well-powered null `results/phase4_hbn_all.csv` |
| calibration (raw) | ECE 0.138 | ✅ `results/calibration.csv` |

**One-line story:** within-MIPDB, a *frozen* EEG foundation model with a lightweight
window-level head is the sweet spot (**0.748**); adding trainable capacity (full
fine-tune) overfits; adding modalities (fusion) does not help; the result is
**robust across FM families** (does not depend on LaBraM); and it does **not** transfer
out-of-distribution (HBN ≈ chance at N≈2,000). A rigorously-evaluated proof-of-concept,
reported honestly.

---

## Reviewer 1

| # | Comment | Response (paste-ready) | Status |
|---|---|---|---|
| R1.1 | Theoretical justification for explainability modules | New Methods subsection grounding interpretability in attention-attribution literature; state explicitly attention is correlational not causal (Rudin 2019; Jain & Wallace 2019); per-modality `[CLS]` attribution as a faithful summary for an asynchronous cross-paradigm model. | ✍️ write subsection |
| R1.2 | Human-centered interpretability eval (user/expert study) | Add an expert-validation study: present per-modality importance for **[[m]]** cases to **[[k]]** clinicians; collect Likert usefulness + inter-rater agreement. Fallback: structured face-validity review if full study infeasible in window. | ❓ decide study size + feasibility |
| R1.3 | OOD / real-world datasets | Central addition: **well-powered external validation on HBN** (N≈1,980, RestingState, all releases R1–11). Honest null — general-psychopathology AUC **0.497 ± 0.006**; best specific factor (internalizing) 0.55 / r 0.09 (negligible); MIPDB→HBN transfer 0.51. Earlier N=128 ≈0.59 did not survive scaling (small-sample variability). Framed as rigorous quantification of the within-cohort↔OOD gap, not a positive transfer claim. | ✅ |

---

## Reviewer 2

| # | Comment | Response (paste-ready) | Status |
|---|---|---|---|
| R2.1 / R2.4 | Neurocognitive / theoretical basis for modality contributions | New "Neurocognitive basis" subsection: hand-crafted 5-ch features under-represent cortical dynamics (EEG looked weak); a pretrained FM (LaBraM, ~2,500 h) recovers spatial-spectral-temporal structure tied to attentional/inhibitory + arousal networks disrupted transdiagnostically. Leak-free: FM-EEG **0.66** vs hand-crafted **0.43–0.47** (Δ up to +0.23); eye 0.51 / behaviour 0.40 at chance here. Earlier eye-dominant finding = leakage artifact + weak EEG features. | ✍️ write subsection |
| R2.2 | Add 3–5 recent SOTA baselines, unify protocol, report significance | Added single-stage multimodal Transformer (no-hierarchy control, also R2.5) + gated-attention fusion under identical subject-level CV; LaBraM = the SSL baseline. Report paired tests + Bonferroni. **Run on Nibi (mh311 CPU), paste table → fill `[[ ]]` in Results.** | 🖥️ run + fill table |
| R2.3 | Suggested refs (indoor object detection; gesture recognition) | Reviewed; outside scope of multimodal psychiatric screening. Per EIC guidance citation is not obligatory and should reflect genuine relevance — respectfully not added; strengthened genuinely-relevant attention/fusion related-work instead. | ✅ (text stance) |
| R2.5 | Why hierarchical > single-stage fusion (theory) | New paragraph: modality-specific encoding preserves within-modality structure before mixing (information fidelity) and reduces premature cross-modal noise coupling (noise filtering). Reported honestly: in this cohort hierarchical fusion does not beat EEG-alone → framed as the preferable *architecture* + missing-modality robustness, not an accuracy gain. Single-stage Vanilla-Transformer = the control. | ✍️ write paragraph |
| **R2 (generality)** | **Is the result a single-FM (LaBraM) artifact?** | **Robustness across FM families.** Identical frozen-readout protocol on independently-pretrained EEG FMs: LaBraM **0.739**, **CBraMod 0.602** (ICLR'25, TUAB-SOTA, channel-agnostic). A different FM does not exceed frozen LaBraM → the ~0.74 ceiling is **task/cohort-bound, not architecture-specific.** EEGPT [pending: either a third confirming point, or report montage-incompatibility on the 129-ch EGI montage and omit the score]. | 🖥️ **EEGPT job 16616052 → finalize wording** |

---

## Reviewer 3 *(most substantive)*

| # | Comment | Response (paste-ready) | Status |
|---|---|---|---|
| R3.1 | Binary diagnosed-vs-control too broad; may detect cohort diffs | Retain binary screening (legitimate triage, RDoC-consistent); soften claims; add confound control (age/sex/site). **Per-disorder breakdown is impossible** — public MIPDB has only binary `DX_Status`, no DSM categories; documented, not fabricated. | ⚠️ + ✍️ (no per-disorder) |
| R3.2 | Clinical-screening claim overstated (research cohort ≠ primary care) | Soften to "proof-of-concept screening support under controlled conditions"; deployment/PPV framed hypothetically; population-shift limitation made prominent. | ✍️ consistency pass |
| R3.3 | N=126, ~25 test/fold; fold-sensitive | **Repeated** stratified CV (5×10 headline) + bootstrap CIs; report distribution across repeats. Full fine-tuning's 3× fold variance (±0.12) is itself reported as evidence. External HBN further mitigates. | ✅ |
| R3.4 | 18.9M params overparameterized vs ~91 subjects | Converted criticism into a result: frozen backbone + window-level head **0.748 ± 0.029**, whereas end-to-end full fine-tune **overfits** — mean drops to 0.69, fold variance triples (±0.12 vs ±0.04). At N≈118 added trainable capacity hurts; frozen/compact recommended, model size justified empirically. | ✅ |
| R3.5 | Auxiliary reconstruction loss may just regularize MIPDB distribution | Tested on HBN: neither fusion nor any auxiliary-loss config recovers above-chance external prediction (≈0.50 at N≈1,980). MIPDB benefit does not transfer OOD; stated plainly under Limitations. | ✅ |
| R3.6 | Cross-paradigm sourcing weakens cross-modal fusion claim | Reframe as **asynchronous, cross-paradigm participant-level fusion**; remove trial-synchronized language; permutation-invariant set-based fusion; state what attention does/doesn't represent. | ✍️ reframe text |
| R3.7 | May learn task/acquisition artifacts (compliance, fatigue, quality) | Add confound analyses (signal-quality, epoch-rejection rate, calibration error as covariates); test whether classification survives controlling for them. | 🖥️/✍️ confound analysis |
| R3.8 | EEG contribution weak — include it at all? | Reviewer right about *original* features; fix reverses it. Leak-free head-to-head: FM-EEG **0.66** (→ **0.748** window-level) vs hand-crafted **0.43–0.47**; EEG exceeds every other single modality and every fusion variant → EEG is the **core**; missing-modality design lets the model still run when EEG is absent. | ✅ |
| R3.9 | Soften three-modality integration claim | Replace "three-modality integration" with "a unified model supporting one-to-three modalities." Report transparently: no fusion variant exceeds EEG-alone (concat 0.48–0.54, attention 0.59, stacking 0.47, all ≤ 0.66); simulated-missingness sweep shows reliance almost entirely on EEG. Title/abstract lead with FM-EEG; multimodal = robustness/missing-data feature. | ✅ + ✍️ title/abstract |
| R3.10 | Selection bias (complete high-quality recordings only) | The missing-modality model addresses this directly: modality-dropout training + learned availability token; evaluate under simulated missing/degraded modalities; report graceful degradation. Residual selection bias (still a research cohort) owned as a limitation. | ✅ mechanism + ⚠️ |

---

## Reviewer 4 *(minor)*

| # | Comment | Response (paste-ready) | Status |
|---|---|---|---|
| R4.1 | Citation for "≈970 million people worldwide" | Add GBD / WHO mental-health citation. | ✍️ |
| R4.2 | Why no ICA? | ICA **was** used (ocular/muscular/cardiac removal, Winkler 2011) — reviewer missed it. Make the step prominent (pipeline-figure caption + clear sentence in EEG subsection). No method change. | ✍️ visibility fix |
| R4.3 | Add interpretability to Results | New Results subsection: per-modality attention/attribution + R1.2 expert-validation. | ✍️ (gated on R1.2) |
| R4.4 | Condense conclusion | Tighten conclusion; move detailed limitations to a dedicated short paragraph. | ✍️ |

---

## Cross-cutting

- **Title/abstract** — foreground the genuine contribution. Candidate: *"A Unified
  Missing-Modality Transformer for Asynchronous EEG / Eye-Tracking / Behavioural
  Mental-Health Screening, with External Validation."* Lead with frozen FM-EEG 0.748.
- **Tone** — respectful, evidence-led, never defensive. Where a reviewer was right
  (R3.8 EEG features, R4.2 ICA visibility), say so plainly.
- **Word: avoid "honest null/negative"** in figures/manuscript (author pref) — say
  "no OOD transfer" / "no multimodal gain."
- **Operating point / calibration** — `src/operating_point.py` gives sens/spec/PPV/NPV +
  risk-coverage from `--save-preds`; fill calibration `[[ ]]` in Results once
  `results/calibration_riskcoverage.csv` is regenerated on the headline.

## What's actually left (the short list)

1. 🖥️ **EEGPT result** (job 16616052) → finalize the R2-generality wording (3rd
   confirming FM, or montage-incompatibility note) + fold into multi-backbone ensemble.
2. 🖥️ **R2.2 SOTA-baseline table** — run on Nibi, paste, fill `[[ ]]` rows in Results.
3. ❓ **R1.2 expert study** — decide real user study vs structured face-validity review;
   set k (raters) and m (cases).
4. ✍️ Prose subsections: R1.1, R2.1/R2.4, R2.5, R3.6 reframe; R4.1/R4.2/R4.4 cosmetics.
5. ✍️ Title/abstract/cover-letter consistency pass to the 0.748 headline.
