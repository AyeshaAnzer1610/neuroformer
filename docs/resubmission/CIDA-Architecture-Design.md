# CIDA — Connectivity-aware, Identity-Disentangled Adapter for LaBraM

**Status:** proposed (design + runnable skeleton in [`src/cida.py`](../../src/cida.py)).
**Type:** new *method* contribution for the revision — an improvement to **LaBraM**
(not a new foundation model, and not a generic FM-agnostic wrapper). LaBraM is the
fixed, frozen backbone; CIDA is the trainable contribution on top of it.

---

## 1. The gap (one root cause, three symptoms)

Our own results triangulate a single weakness of frozen/fine-tuned LaBraM. These
are not three separate problems — they share one cause:

| Symptom (our result) | What it actually is |
|---|---|
| Leakage inflated AUC to 0.91; honest = 0.75 | The embedding encodes **subject identity** so well that the same person in train+test is a free score |
| Full fine-tuning **overfits** (0.69, ~3× variance) | 5.8M+ params chasing N≈118 → memorises **people**, not pathology |
| External HBN transfer = **0.50** | New device / montage / site → the representation is entangled with **recording domain**, so it collapses out-of-distribution |

**Root cause.** LaBraM's self-supervised objective is *per-channel, linear-spectral*
reconstruction (a VQ tokenizer predicting Fourier amplitude + phase of 1 s, 200-sample
channel patches). It was never asked to be **subject-** or **site-invariant**, and it
encodes **cross-channel functional connectivity** only weakly and implicitly (channels
interact via attention, but the *learning signal* is per-channel). So the representation
entangles *who you are* and *what machine recorded you* with *what condition you have*.

This is corroborated externally: **LEAD** (an EEG foundation model for Alzheimer's that
reports beating LaBraM) gets its edge from **subject-regularised training + clinical
contrastive learning** — i.e. attacking exactly this entanglement.

## 2. Scope and the honest SOTA framing

Two readings of "new architecture that becomes SOTA":

- **Not** a new foundation model trained from scratch — that needs a TUH/GazeBase-scale
  corpus and thousands of GPU-hours, and would still likely lose to LaBraM-Huge (369M)
  in-distribution. Infeasible for a revision; high risk of no result.
- **Yes**, a new *adaptation architecture* that wraps **frozen LaBraM** and beats full
  fine-tuning on the axis where LaBraM is weak: **small clinical N** and **cross-dataset
  transfer**. Winnable, novel, and this is where SOTA is actually contestable
  (LEAD, CBraMod, EEGPT all compete here).

A SOTA *claim* is only credible on **shared public benchmarks**. The plan (§5) is to
report CIDA-on-LaBraM against published LaBraM/EEGPT/CBraMod numbers on **TUAB**
(abnormal detection) and **TUEV** (event type) — LaBraM's own tables — plus a
**leave-one-dataset-out clinical transfer** protocol. MIPDB/HBN remain our application
case study; they are not a leaderboard, so we do not claim "SOTA on MIPDB".

## 3. The architecture

One-line thesis: *Don't retrain the brain model — wrap it. Freeze LaBraM and learn a
tiny adapter that (a) injects the cross-channel connectivity its tokenizer omits and
(b) explicitly factors subject/site identity out of the clinical signal — turning an
identity-dominated embedding into a transfer-robust clinical one.*

```
   raw EEG windows (any montage, 200 Hz, 200-sample patches)
            │
   ┌────────▼─────────┐   FROZEN LaBraM  (0 trainable params)
   │  LaBraM backbone │   per-channel patch tokens  +  window embedding
   └────────┬─────────┘   (braindecode InterpolatedLaBraM, return_features=True)
            │  channel/patch token embeddings  [B, C, P, d]
   ┌────────▼───────────────────────────────────────┐
   │ (1) CONNECTIVITY GRAPH ADAPTER                  │  injects the inductive bias
   │   electrodes = nodes; edges = learned (+ optional│  LaBraM's per-channel spectral
   │   PLV/coherence prior); GNN over channel tokens │  objective omits
   ├─────────────────────────────────────────────────┤
   │ (2) LoRA on LaBraM attention (≈1% of params)    │  PEFT: adapt without overfitting;
   │     (rank-r updates injected into the backbone) │  directly answers "too big"
   ├─────────────────────────────────────────────────┤
   │ (3) ATTENTION TEMPORAL POOL over windows        │  replaces mean-pool; keeps
   │     (set-transformer; emits μ + uncertainty)    │  temporal structure + uncertainty
   ├─────────────────────────────────────────────────┤
   │ (4) DISENTANGLEMENT HEAD  z → [z_clin ‖ z_nuis] │  THE novel core
   │   • clinical classifier on z_clin               │  subject+site adversary via
   │   • subject+site adversary (GRL) on z_clin      │  gradient reversal → z_clin is
   │   • nuisance classifier on z_nuis (captures id) │  invariant; orthogonality keeps
   │   • orthogonality penalty  z_clin ⟂ z_nuis      │  the two factors disjoint
   └────────┬────────────────────────────────────────┘
            │
   ┌────────▼─────────┐
   │ calibrated head  │  conformal abstention → "refer to clinician" (reuses our triage)
   └──────────────────┘
```

### Block-by-block (what each fixes, and the precedent it reuses)

1. **Connectivity graph adapter.** Treat electrodes as graph nodes; edges are a learned
   adjacency, optionally initialised from a connectivity prior (PLV / coherence) computed
   per window. A light GNN/message-passing layer over the per-channel LaBraM tokens adds
   the connectivity inductive bias the tokenizer lacks. *Reuses:* EEG-GraphAdapter (EGA)
   graph-adapter idea for PEFT of EEG models.
2. **LoRA on LaBraM attention.** Rank-`r` low-rank updates injected into the backbone's
   attention projections; backbone weights stay frozen. ~1% trainable params → adapts
   without the overfitting we saw at full fine-tune, and *answers the "18.9M too big"
   reviewer head-on*. *Reuses:* LoRA; TaKF⁺ (PEFT for EEG FMs).
3. **Attention temporal pool.** A set-transformer / attention pool aggregates a subject's
   window tokens (replacing the current mean-pool), preserving temporal structure and
   emitting an uncertainty estimate that feeds the triage head.
4. **Disentanglement head (the novel core).** Split the pooled representation into
   `z_clin` and `z_nuis`. Train:
   - clinical classifier on `z_clin` → diagnosis (keep clinical info in `z_clin`);
   - **subject + site discriminators on `z_clin` through a gradient-reversal layer**
     → push identity/domain *out* of `z_clin` (invariance);
   - a nuisance classifier on `z_nuis` → subject/site (let `z_nuis` *absorb* identity);
   - an **orthogonality penalty** ‖z_clinᵀz_nuis‖ → keep the two factors disjoint.
   *Reuses:* DANN / gradient-reversal subject-invariance; the novelty is applying it
   **in an adapter on a frozen FM**, not by re-pretraining (LEAD) or full fine-tuning.

### The single novel claim (state it this crisply)

> *The first connectivity-aware, identity-disentangled parameter-efficient adapter for
> EEG foundation models — invariance and connectivity are enforced in a ~1%-parameter
> adapter on a **frozen** backbone, rather than by re-pretraining or full fine-tuning.*

Everything else (LoRA, graph adapter, attention pooling, conformal triage) is
**deliberately reused** and cited as such. The contribution is the combination and
*where the invariance is applied*, motivated directly by our leakage→identity story.

## 4. Loss

```
L = L_clin                              (cross-entropy, class-weighted)
  + λ_adv · (L_subj_adv + L_site_adv)   (via GRL on z_clin — maximised w.r.t. encoder)
  + λ_nui · L_nuis                       (subject/site CE on z_nuis — normal direction)
  + λ_ortho · ‖z_clinᵀ z_nuis‖²_F        (factor disjointness)
  + λ_rec · L_recon                      (optional: reconstruct z from [z_clin‖z_nuis]
                                          to prevent representational collapse)
```

`λ_adv` is ramped (GRL coefficient schedule) so the adversary doesn't dominate early.

## 5. Evaluation plan (how the SOTA claim is earned)

1. **In-distribution PEFT vs full FT (our cohort).** CIDA vs frozen-readout (0.66) vs
   full fine-tune (0.69), subject-level 5×10 CV. Expectation: ≥ both, far fewer trained
   params. *Answers the "too big" reviewer with a result.*
2. **Public benchmarks (the SOTA bar).** CIDA-on-LaBraM on **TUAB** + **TUEV** vs
   published LaBraM / EEGPT / CBraMod / LEAD. Needs TUH/public sets on the cluster.
3. **Cross-dataset transfer (where LaBraM is weakest).** Leave-one-dataset-out; report
   whether CIDA survives the OOD drop that takes plain LaBraM to ~0.50 (our HBN result).
4. **Disentanglement ablations (the falsifiers, §6).**

## 6. What would falsify CIDA (run these, report either way)

- **Subject-decodability probe.** Train a probe to predict subject ID from `z_clin`
  before/after the adversary. If it stays high, disentanglement failed — report it.
- **Connectivity ablation.** Remove block (1); if AUC/transfer is unchanged, the graph
  adapter isn't earning its place — say so.
- **PEFT vs full FT variance.** If full FT matches CIDA *and* generalises, the PEFT
  motivation weakens. We expect the opposite (our 0.69-with-3×-variance result).

## 7. Reviewer mapping

- "18.9M params too big for ~90 subjects" → PEFT adapter, ~1% trainable params (§3.2, §5.1).
- "small N / fold sensitivity" → subject-level 5×10 CV + invariance objective (§3.4).
- "no external validation / doesn't generalise" → cross-dataset transfer is the headline
  axis CIDA targets (§5.3).
- "what's genuinely new?" → the disentangled-invariant *adapter* on a frozen FM (§3).

## 8. References (verified)

- LaBraM — *Large Brain Model…*, ICLR 2024 (spotlight). arXiv:2405.18765;
  code github.com/935963004/LaBraM. (Base 5.8M / Large 46M / Huge 369M params;
  VQ neural-spectrum tokenizer; 200-sample patches; ~2,500 h / ~20 datasets.)
- EEGPT — *Pretrained Transformer for Universal & Reliable EEG Representation*, NeurIPS 2024.
- LEAD — *An EEG Foundation Model for Alzheimer's Disease Detection*, arXiv:2502.01678
  (subject-regularised training + medical contrastive learning; reports beating EEG FMs).
- TaKF⁺ — versatile parameter-efficient tuning for EEG foundation models (OpenReview).
- EEG-GraphAdapter (EGA) — GNN adapter for PEFT of EEG models.
- LoRA — *Low-Rank Adaptation of Large Language Models*.
- DANN / gradient reversal — *Domain-Adversarial Training of Neural Networks*; and
  *Learning Invariant Representations from EEG via Adversarial Inference*.
- braindecode `InterpolatedLaBraM` — montage-interpolating LaBraM wrapper used here.
