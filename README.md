# mental-health

Multimodal mental-health screening research — fusing **EEG, eye-tracking, and
behavioural** signals with transformer models. This repo continues the NeuroFormer
work, repositioned around a **foundation-model + missing-modality** contribution for
resubmission to IEEE Access. See [docs/resubmission/](docs/resubmission/) for the
strategy, roadmap, and reviewer-response planning.

## Compute & data architecture

- **Code + docs live here (GitHub).** Small, version-controlled, the durable backup.
- **Raw datasets live on the Digital Research Alliance of Canada cluster** (`$SCRATCH`
  or `/project`), downloaded directly from public S3 — *never committed to git*.
- Raw data is **public and re-downloadable** (see `scripts/download_data.sh`), so it
  needs no backup; only code and processed outputs do.

## Structure

| Folder | What goes here |
|---|---|
| `data/raw/` | Raw signals (git-ignored; on cluster, this points at `$SCRATCH`). |
| `data/processed/` | Cleaned / derived features & cached embeddings (git-ignored). |
| `src/` | Models, preprocessing, training, fusion code. |
| `notebooks/` | Exploratory analysis. |
| `results/` | Figures, tables, metrics (git-ignored). |
| `docs/` | Documentation; `docs/resubmission/` holds the IEEE Access strategy. |
| `scripts/` | Data download + cluster (SLURM) job scripts. |

## Datasets

| Dataset | Use | Source |
|---|---|---|
| CMI **MIPDB** | Primary cohort (N≈126) | `s3://fcp-indi/data/Projects/EEG_Eyetracking_CMI_data` |
| CMI **HBN** (Release 1, `ds005505`) | External validation | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1` |

Both use the 128/129-channel EGI HydroCel net (same montage → clean cross-dataset transfer).

## Getting started (on the Alliance cluster)

```bash
git clone https://github.com/AyeshaAnzer1610/mental-health.git
cd mental-health
bash scripts/download_data.sh        # run on a LOGIN node; reads $SCRATCH
```
