# NeuroFormer — Hierarchical Multimodal Mental-Health Screening (EEG · Eye-Tracking · Behaviour)

Code and data pipeline for the IEEE Access paper *"NeuroFormer: A Hierarchical
Multimodal Framework for Binary Mental Health Screening from EEG, Eye-Tracking, and
Behavioural Signals"* (Anzer & Bais), and its resubmission materials.

NeuroFormer fuses three non-invasive modalities from the Child Mind Institute **MIPDB**
cohort through a two-stage transformer (modality-specific encoders → cross-modal fusion)
for binary diagnosed-vs-control screening.

## Layout

| Path | Contents |
|---|---|
| `src/neuroformer.py` | The 3-modality NeuroFormer architecture (EEG + eye + behaviour encoders, cross-modal fusion transformer, optional auxiliary reconstruction). |
| `src/extract_handcrafted_eeg.py` | Hand-crafted EEG features from raw MIPDB. |
| `src/extract_eye_features.py` | Eye-tracking features from SMI `*Samples.txt`. |
| `src/extract_behaviour.py` | SAIIT_2AFC behaviour: accuracy (`trialLR` vs `RespLR`) + reaction time (event timing `CD_TG`→`CD_BUTTONS`). |
| `data/processed_handcrafted/` | Per-subject hand-crafted EEG features. |
| `data/eye_features.csv` | Per-block eye-tracking features. |
| `data/processed_beh/` | Per-subject behaviour features (RT stats + accuracy). |
| `data/labels.csv`, `data/MIPDB_PublicFile.csv` | Subject labels (`DX_Status`) and metadata. |
| `experiments/robustness_3mod.py` | Leak-free, subject-level repeated-CV robustness evaluation of the 3-modality model. |
| `docs/resubmission/` | Reviewer responses and resubmission planning. |

## Reproduce the robustness evaluation

```bash
python experiments/robustness_3mod.py --beh-dir data/processed_beh --nf-repeats 5
```

Reports per-modality logistic-regression baselines, naive concatenation, and the
3-modality NeuroFormer under strict subject-level repeated stratified cross-validation
(no leakage), with bootstrap confidence intervals.

## Data

Child Mind Institute **MIPDB** — public; raw data downloaded separately and not committed.
The extracted per-subject features (EEG, eye, behaviour) are included under `data/`.

## Related repository

The foundation-model EEG line of work (frozen vs fine-tuned EEG foundation models,
multi-backbone robustness, external-cohort validation) is a separate research direction
maintained in its own repository.

## Citation

A. Anzer and A. Bais, "NeuroFormer: A Hierarchical Multimodal Framework for Binary Mental
Health Screening from EEG, Eye-Tracking, and Behavioural Signals," *IEEE Access* (under
review).

## License

Code: MIT. Data: per the original MIPDB dataset license.
