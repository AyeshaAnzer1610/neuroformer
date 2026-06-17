# Proposed framework — best-available-representation, missing-modality fusion

> **Update (2-modality direction).** This framework is now *realized concretely*:
> the "self-supervised encoder where no foundation model exists" for eye-tracking is
> built as **OcuFM** ([OcuFM-Architecture-Design.md](OcuFM-Architecture-Design.md)),
> a gaze foundation model on the LaBraM recipe, and the cross-modal/robustness
> mechanism is generalized into **CIDA** ([CIDA-Architecture-Design.md](CIDA-Architecture-Design.md)),
> an identity-disentangling adapter applied to *both* foundation backbones.
> **Behaviour has been dropped** (below chance, ~0.40). The text below is the original
> framing; OcuFM + CIDA are its successors.

A method/positioning contribution for the revision. It is proposed and shown to
**run**; on this small cohort it does **not** improve accuracy (the non-EEG
modalities are at chance), and we report that honestly. Its value is as a
**framework for heterogeneous multimodal psychiatric data** and a concrete future
direction — not a claimed accuracy gain here.

## The idea
Real multimodal psychiatric data mixes modalities that differ enormously in how much
representation learning helps and in whether a pretrained model exists. We propose a
unified scheme that assigns **each modality its strongest available representation**:

1. **Foundation-model representation where one exists.** EEG → frozen **LaBraM**
   embeddings (the path that carries the signal here, ROC-AUC 0.748).
2. **Self-supervised encoder + cross-modal alignment where none exists.** For
   eye-tracking and behaviour — which have no mature foundation model — a modality
   encoder is trained with a **cross-modal contrastive objective** (InfoNCE) that
   projects each modality into a shared space, pulling together the modalities of the
   same subject and pushing apart different subjects, respecting the
   missing-modality availability mask.
3. **Missing-modality regime throughout.** Modality dropout + availability tokens, so
   one model serves any subset {EEG, eye, behaviour} with graceful degradation.

This is implemented as the optional `CrossModalAlignment` module in
[`src/neuroformer.py`](../../src/neuroformer.py) (shared projection + symmetric
InfoNCE, availability-aware), composing with the existing modality-dropout fusion.

## What we claim — and what we do not
- **We claim:** a coherent framework for representing and fusing *heterogeneous*
  modalities (foundation-model where available, self-supervised + aligned otherwise)
  under realistic missingness. It is implemented and runs.
- **We do not claim** an accuracy gain on MIPDB: with eye (0.51) and behaviour (0.40)
  at chance and N≈104 complete-case, the alignment cannot manufacture signal and is
  **off for all headline results**. We report this transparently.
- **We do not claim** a trained "foundation model" for eye/behaviour: that needs a
  large external corpus (GazeBase-scale) and is out of scope — see Future work.

## Why not just train an eye/behaviour foundation model now
Foundation models require large-scale pretraining corpora (LaBraM: thousands of hours
of EEG). MIPDB's ~118 subjects cannot pretrain one without overfitting, and no
off-the-shelf gaze/behaviour foundation model exists. Moreover, the within-cohort
signal in these modalities is at chance, so a richer representation has nothing to
recover here. Pretraining such an encoder is a separate, larger project.

## Future work (the concrete next step)
Pretrain a **gaze-sequence self-supervised encoder** on a large external
eye-movement corpus, then learn the EEG↔gaze contrastive alignment on a cohort that
records both at scale (e.g., HBN-style). This would test whether the framework yields
gains when the non-EEG modalities actually carry signal and when there is enough
paired data to fit a shared representation — the natural successor to this work.

## Reviewer mapping
Strengthens the methodological contribution and the missing-modality story (R3.6,
R3.9, R3.10) by making the *representation* choice principled and explicit, while
keeping every empirical claim honest.
