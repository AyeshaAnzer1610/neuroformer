# Fixing & Improving NeuroFormer (keep the model, fix the pipeline, fill the gaps)

We keep the NeuroFormer architecture (`src/neuroformer.py`: modality encoders +
cross-modal fusion + CLS classifier + optional reconstruction). We fix what was
broken/missing in the **pipeline and inputs**, and add components that fill current
top-tier gaps and answer every reviewer. **No full model rewrite.**

## What was broken/missing → the fix (all keep the model)

| # | Was broken / missing | Fix (in pipeline or as add-on) | Reviewer | Top-tier anchor |
|---|---|---|---|---|
| 1 | **Evaluation likely leaked** (block-level / not strictly subject-level → inflated ~0.91) | **Strict subject-level, leak-free repeated CV + bootstrap CIs** (already built in our `src/`) | R3.3, R3.4 | standard ML rigor |
| 2 | **Weak hand-crafted 5-ch EEG** (the EEG encoder input) | Feed the *same* `ModalityEncoder` **LaBraM foundation-model embeddings** (eeg_input_dim 128→200); EEG becomes the strongest stream | R3.8, R2.1/2.4 | LaBraM ICLR'24, EEGPT NeurIPS'24, CBraMod ICLR'25 |
| 3 | **Assumed synchronized cross-modal fusion** on cross-paradigm data | Keep fusion transformer, reframe as **asynchronous participant-level set fusion**; add **modality-dropout + availability** (model already has modality embeddings + masks) | R3.6, R3.9, R3.10 | DrFuse AAAI'24, M3Care KDD'22 |
| 4 | **No external validation** | Evaluate on **HBN** (independent cohort) | R1.3, R3.2, R3.5 | HBN-EEG bioRxiv'24, NeurIPS'25 EEG Challenge |
| 5 | **No calibration / uncertainty** | Post-hoc **temperature scaling + conformal** referral head (`src/calibrate.py`) | R3.2 | conformal clinical triage Sci.Rep.'25 |
| 6 | **Overparameterized** (18.9M vs ~91 train) | **Compact variant** + the FM front-end (light head on frozen features is principled) | R3.4 | — |
| 7 | Interpretability ungrounded; ICA not prominent; 970M uncited; conclusion long | Ground attention claims; surface ICA; add citation; condense | R1.1, R4.1/4.2/4.4 | Rudin'19 |

## Architecture: what changes vs stays
- **Stays:** `ModalityEncoder` ×3, `CrossModalFusionTransformer` (CLS + modality
  tokens), classifier head, reconstruction decoders. The NeuroFormer name + design.
- **Changes (minimal):** EEG stream input = LaBraM per-window embeddings (200-d)
  instead of 128-d hand-crafted; **add modality-dropout** in training; **post-hoc
  calibration** on outputs. That's it — same model, better inputs + honest training.

## Honest framing of results (non-negotiable)
The fixed, leak-free NeuroFormer reports lower-but-real numbers than the inflated
0.91. The publishable strength is the **contribution set** (FM-EEG + missing-modality
+ external validation + calibration, under leak-free eval), not a leaderboard score.
We report the multimodal-fusion result honestly (EEG carries the signal here).

## The gap NeuroFormer v2 owns (vs MTNet'24 / MSNet'25, which fuse EEG+eye)
Prior EEG+eye psychiatric fusion is bimodal, single-disorder, hand-crafted,
complete-data, single-cohort, uncalibrated. **NeuroFormer is the first to unify a
pretrained EEG foundation model + missing-modality fusion + transdiagnostic screening
+ external validation + calibrated referral in one framework.**

## Build order (keeps model, fixes pipeline)
1. ✅ Port model to `src/neuroformer.py`.
2. **`src/train_neuroformer.py`** — leak-free subject-level trainer feeding LaBraM-EEG
   window sequences + eye/behaviour block sequences into NeuroFormer, with
   modality-dropout; report CV AUC + the fusion ablation honestly. (GPU for end-to-end
   / LaBraM fine-tuning; CPU OK for frozen-feature mode.)
3. External HBN eval of the trained model; post-hoc calibration.
4. Fill `Response-to-Reviewers.md` numbers; edit manuscript methods/results.
