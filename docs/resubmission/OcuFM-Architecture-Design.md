# OcuFM — a foundation model for eye-tracking (the LaBraM analog for gaze)

**Status:** proposed (design + runnable skeleton in [`src/ocufm.py`](../../src/ocufm.py)).
**Type:** new *method* contribution for the revision — a **gaze/eye-tracking foundation
model** built on exactly the LaBraM recipe, filling the gap that eye-tracking has **no
pretrained foundation model** while EEG has LaBraM. Like LaBraM, OcuFM is a *frozen*
representation; downstream it is adapted with **CIDA** (see
[CIDA-Architecture-Design.md](CIDA-Architecture-Design.md)).

> Honesty up front: OcuFM is **proposed and runnable, not yet pretrained.** Pretraining
> needs a large external gaze corpus (GazeBase-scale) and GPU time; that is the concrete
> next step, not a claimed result here. We do **not** report an accuracy gain from OcuFM
> on our cohort. Our headline empirical signal remains EEG (LaBraM + CIDA). This mirrors
> the honest stance already taken in the project.

---

## 1. Why this fills a real gap

Our pipeline pairs a strong EEG representation (LaBraM, AUC ~0.75) with **11 hand-crafted
eye-tracking features** that sit at chance (~0.51). The asymmetry is the problem the
[Proposed-Framework doc](Proposed-Framework-Multimodal.md) already named: EEG has a
foundation model; eye-tracking does not. Hand-crafted gaze features (mean fixation
duration, saccade counts, …) discard most of the temporal/kinematic structure that
representation learning would capture.

**OcuFM closes the asymmetry**: give gaze the *same* kind of pretrained, transferable
representation LaBraM gives EEG, so the fusion is foundation-model ⊕ foundation-model
rather than foundation-model ⊕ hand-crafted-features.

## 2. The unifying insight (why this is one coherent paper, not two)

Both EEG **and** gaze are **identity-dominated** signals:
- EEG foundation embeddings encode *who you are* (this is exactly what produced our
  leakage 0.91 and full-fine-tune overfit 0.69).
- Eye movements are a **biometric** — gaze dynamics identify individuals so reliably
  that "eye-movement biometrics" is an established field.

So the *same* failure mode threatens both modalities, and the *same* fix — **CIDA's
identity-disentanglement** — applies to both. The paper's spine becomes:

> *Two foundation-model representations (LaBraM for EEG, OcuFM for gaze), each
> identity-dominated, each disentangled by CIDA, fused for transdiagnostic screening —
> leak-free, externally validated, calibrated.*

Behaviour is removed (it was below chance and only added noise).

## 3. The architecture — LaBraM recipe, mapped to gaze

| LaBraM (EEG) | **OcuFM (gaze)** |
|---|---|
| Input: multi-channel EEG @ 200 Hz | Input: gaze channels — `x, y` position, `pupil`, and derived `velocity, acceleration` (resampled to a canonical rate) |
| 1 s = 200-sample per-channel patches | fixed-duration per-channel patches (e.g. 1 s) |
| VQ neural tokenizer → predict **Fourier amplitude + phase** | VQ tokenizer → predict **kinematic + spectral** targets: saccade/fixation dynamics + pupil spectrum |
| Masked-patch pretraining (predict codes of masked patches) | **same** masked-patch objective |
| ~2,500 h, ~20 EEG datasets | **GazeBase** (~322 subjects, multi-round), GazeBaseVR, JuDo1000, … |
| `InterpolatedLaBraM` → arbitrary montages | tracker/sampling-rate-agnostic input layer → arbitrary eye-trackers |
| Frozen → identity-dominated → needs CIDA | Frozen → identity-dominated (biometric) → **needs CIDA** |

### Pipeline blocks (mirrors LaBraM)
```
   raw gaze stream (any tracker, any rate)
        │  resample to canonical rate; derive velocity/acceleration; z-score per channel
   ┌────▼─────────────┐
   │ Patch embedding  │   per-channel 1 s patches → patch tokens + channel + temporal emb
   ├──────────────────┤
   │ VQ tokenizer     │   codebook over gaze patches; target = kinematic + spectral feats
   │ (pretrain stage1)│
   ├──────────────────┤
   │ Transformer enc. │   masked-patch modelling: predict codes of masked patches
   │ (pretrain stage2)│
   └────┬─────────────┘
        │  frozen embeddings  (per-window gaze representation)
   ┌────▼─────────────┐
   │   CIDA adapter   │   identity-disentangled, attention-pooled (shared with EEG path)
   └──────────────────┘
```

The skeleton in [`src/ocufm.py`](../../src/ocufm.py) implements: the gaze patch embedding,
a vector-quantizer (codebook + straight-through estimator) with a kinematic/spectral
reconstruction target, and a masked-patch Transformer pretraining step — runnable on
synthetic data so the design is concrete and unit-tested.

## 4. How it plugs into the rest of the system

- **Frozen representation** → pooled per subject, exactly like LaBraM EEG embeddings, and
  fed to the 2-modality fusion (`MaskedSetFusion`, dims = {eeg, eye}).
- **CIDA on the gaze branch** → the *same* adapter (connectivity-over-gaze-channels +
  identity-disentanglement + attention pool) adapts OcuFM, demonstrating CIDA is not
  EEG-specific. `src/cida.py` already accepts a `[B, W, C, d]` token tensor; OcuFM's
  channel tokens (x/y/pupil/velocity = channels) drop straight in.

## 5. Evaluation plan (and the honest boundary)

1. **Pretraining (the work):** masked-patch pretrain OcuFM on GazeBase-scale data; report
   linear-probe quality on held-out gaze tasks (biometric ID should be *high* pre-CIDA —
   that's the point — and clinical signal recoverable post-CIDA).
2. **On our cohort:** OcuFM-eye vs the current 11 hand-crafted eye features (0.51), and
   OcuFM⊕LaBraM fusion vs LaBraM-alone — reported honestly, gain or not.
3. **Public gaze benchmarks** (if claiming SOTA on gaze): GazeBase biometric / task
   classification leaderboards.

**Boundary we will not cross:** no OcuFM accuracy number is reported until it is actually
pretrained. Until then it is a proposed architecture + runnable skeleton + a concrete
pretraining plan — the natural successor experiment, stated as such.

## 6. Reviewer mapping

- "soften the multimodal claim / why include the weak modality" → we *upgrade* the weak
  modality with a foundation model instead of dropping the multimodal idea, and keep the
  empirical claim honest (no fabricated gain).
- "what's genuinely new?" → first gaze foundation model on the LaBraM recipe **and** the
  demonstration that one disentanglement adapter (CIDA) serves both identity-dominated
  modalities.

## 7. References (verified / to cite)

- LaBraM — ICLR 2024 (the recipe OcuFM mirrors). arXiv:2405.18765.
- GazeBase — large longitudinal eye-movement dataset (Griffith et al., Sci. Data 2021);
  GazeBaseVR; JuDo1000 — candidate pretraining corpora.
- Eye-movement biometrics literature — motivation for the identity-dominance claim.
- VQ-VAE — *Neural Discrete Representation Learning* (van den Oord et al.) — the tokenizer.
- CIDA — [CIDA-Architecture-Design.md](CIDA-Architecture-Design.md) (the shared adapter).
