# Implementation Roadmap — Option A

> **Foundation-model + missing-modality transdiagnostic screening**, targeting IEEE Access.
> No hard deadline → do it properly. Each phase lists deliverables, a decision gate, and
> which reviewer concerns it retires. Effort tags are rough (solo researcher, part-time).

## Guiding contribution (the spine)
A unified transformer that fuses **pretrained EEG foundation-model embeddings**,
**eye-tracking**, and **behaviour** under a **missing-modality regime** (modality dropout +
availability token + set-based asynchronous fusion), validated **across datasets (MIPDB → HBN
+ an external eye-tracking cohort)**, with **calibrated, referral-aware predictions**.

---

## Phase 0 — Setup & access  *(1–2 weeks, low effort)*
- [ ] Bring the existing pipeline into this repo as the backbone (keep hand-crafted features as a baseline).
- [ ] Get **HBN** access: pick OpenNeuro/NEMAR release(s); confirm EEG + behavioural usable now,
      check current eye-tracking availability. (Same 128-ch EGI net as MIPDB — minimal harmonization.)
- [ ] Obtain a foundation-model checkpoint: **LaBraM** (default) or **EEGPT**; verify input format
      (sampling rate, patching, channel handling). Consider **HEAR** if any electrode mismatch arises.
- [ ] Compute: confirm a free T4 (Colab/Kaggle) works for frozen extraction; request UofR HPC for optional heavy runs.
- **Gate:** HBN downloaded + foundation-model embeddings extractable on one MIPDB subject.

## Phase 1 — Foundation-model EEG features  *(2–3 weeks, medium)*
- [ ] Re-process MIPDB EEG to the foundation model's expected format; use **high-density** channels
      (not the 5-ch montage) since the FM exploits them.
- [ ] Extract & cache frozen per-window embeddings; optionally add a small trainable adapter.
- [ ] **Head-to-head:** foundation-model EEG features vs. old hand-crafted 5-ch features (EEG-alone task).
- **Gate:** decide whether FM-EEG meaningfully outperforms hand-crafted EEG. *Either outcome is a
  reportable finding* and directly answers R3.8 ("is EEG worth including?").
- **Retires:** R3.4 (overparam — FM + light head is principled), R3.8, R2.1/2.4 (EEG-contribution question, now empirically grounded).

## Phase 2 — Missing-modality architecture  *(3–4 weeks, high)*
- [ ] Replace "pretend-synchronized cross-modal attention" with **set-based / permutation-invariant
      fusion** over modality tokens (honest about asynchronous cross-paradigm sourcing).
- [ ] **Modality dropout** during training + a learned **modality-availability token** so one model
      serves any subset of {EEG, eye, behaviour}.
- [ ] (Optional, Gap 3) Add **calibrated/selective prediction**: temperature scaling or conformal
      prediction → "refer to clinician" abstention; report ECE + risk–coverage.
- **Gate:** single model handles all 7 modality subsets without separate retraining.
- **Retires:** R3.6 (fusion premise), R3.9 (soften 3-modality claim), R3.10 (selection bias / real-world missingness), R1.1 (interpretability grounding via per-modality tokens).

## Phase 3 — Internal experiments (MIPDB)  *(2–3 weeks, medium)*
- [ ] Repeated stratified CV (e.g., 5×5 seeds) + bootstrap CIs; report distribution, not one run.
- [ ] Re-run the **7-subset ablation** with FM-EEG + missing-modality model.
- [ ] **Simulated-missingness** eval: degrade/drop modalities at test time → graceful-degradation curves.
- [ ] **Compact-vs-large** model comparison (parameter count vs. accuracy).
- [ ] **Aux-loss ablation** retained (now also tested externally in Phase 4).
- [ ] **Confound controls:** signal quality, epoch-rejection rate, calibration error, age/sex as covariates.
- **Retires:** R3.3 (fold sensitivity), R3.4 (overparam), R3.7 (task/acquisition artifacts), R3.1 (per-disorder breakdown).

## Phase 4 — External validation  *(3–4 weeks, high — the make-or-break)*
- [ ] **HBN:** evaluate MIPDB-trained model on HBN using EEG + behaviour (eye-tracking dropped via
      missing-modality design). Report honest OOD performance + drop.
- [ ] **ADHD eye-tracking cohort:** validate the eye pathway on an independent psychiatric label.
- [ ] (If available) EEGEyeNet for fusion/missingness mechanism stress-test (no clinical labels).
- [ ] Test whether the **auxiliary-loss benefit transfers** externally (answers R3.5 directly).
- **Gate:** at least HBN external validation complete with honest numbers.
- **Retires:** R1.3, R3.2 (clinical-screening realism), R3.5 (aux-loss generalization).

## Phase 5 — Theory, baselines, interpretability  *(2–3 weeks, medium)*
- [ ] **Neurocognitive theory** section: oculomotor/attentional networks transdiagnostically;
      cite the 2025 transdiagnostic eye-tracking biomarker review (R2.1/2.4).
- [ ] **Representation-learning theory** for hierarchical vs single-stage fusion (R2.5).
- [ ] **SOTA baselines:** add 3–5 recent multimodal/temporal-fusion/self-supervised models under
      identical inputs + protocol + significance tests (R2.2).
- [ ] **Interpretability:** per-modality attribution analysis in Results + a structured **expert
      face-validity review** (clinician ratings); full user study only if feasible (R1.2, R4.3).
- **Retires:** R2.1, R2.2, R2.4, R2.5, R1.1, R1.2, R4.3.

## Phase 6 — Writing & framing  *(2–3 weeks, medium)*
- [ ] New title/abstract/contributions around the FM + missing-modality + external-validation story.
- [ ] Fold minor fixes: 970M citation (R4.1), make ICA prominent (R4.2 — already used), condense conclusion (R4.4).
- [ ] Decline the coercive citations politely (R2.3); strengthen genuinely relevant references.
- [ ] Decide: keep the **NeuroFormer** name or rebrand around the new contribution.
- **Gate:** full internal read-through; every original reviewer concern demonstrably addressed.

---

## Risks & honest caveats
- **FM-EEG may still underperform.** Fine — report it; it answers a real question and you still
  have a strong missing-modality + external-validation contribution.
- **HBN harmonization / eye-tracking availability** could shift the external-validation design.
- **External eye-tracking cohort** may be small or access-restricted — confirm early.
- **Expert study** logistics — have a face-validity fallback.
- Even with all this, **N is still modest**; keep "proof-of-concept" honesty — it reads as
  credibility, not weakness.

## Sequencing note
This is realistically a **2–3 month part-time** program. Phases 1–2 are the core method;
Phase 4 is the make-or-break; Phases 5–6 are addressable in parallel once data flows.
Decision gates let us stop/pivot if a phase result changes the story.
