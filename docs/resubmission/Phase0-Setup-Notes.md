# Phase 0 — Setup & Access Notes

> Goal of Phase 0: remove all uncertainty before coding. What to download, where it
> lives, how to run it on the Digital Research Alliance of Canada cluster, and the
> foundation-model input requirements. Sources at the bottom.

## 0.1 Inventory of what we already have (and a problem)

In the original `Multisource` repo, `data/` contains only **extracted feature CSVs**:
- `processed_asd_features.csv` (~217 KB) — **eye-tracking** features despite the name
  (columns are pupil size, gaze x/y, gaze vector — *not* EEG); per-block, ~951 rows.
- `processed_features_behavioral_features.csv` (~22 KB) — behavioural (RT, accuracy).
- `processed_eyetracking_features.csv` — **2 bytes (EMPTY)** ⚠️
- `labels.csv`, `MIPDB_PublicFile.csv` — `ID, Age, Sex, DX_Status` for 126 subjects.
  `DX_Status` is `{0: 101, 1: 22, 2: 3}` → we use **binary** "any DX vs none" (class 2 n=3).
  Copied into this repo's `data/` (tiny public metadata, whitelisted in `.gitignore`).

**Where the hand-crafted EEG features actually live (important):**
- The old EEG baseline (`EEGFinal.ipynb`) computed, after band-pass + normalization,
  **Hjorth (activity/mobility/complexity) + spectral entropy** per channel and saved
  them to a **GCS bucket** (`gs://eegchild/processed_features/*_features.csv` →
  `merged_features.csv`) — **not** in this repo, **not** on Nibi.
- Decision: rather than depend on GCS auth, we **re-implement that exact feature set**
  in [`src/extract_handcrafted_eeg.py`](../../src/extract_handcrafted_eeg.py) on the
  *same* raw EEG / windows we feed LaBraM, so the Phase 1 head-to-head
  ([`src/compare_fm_vs_handcrafted.py`](../../src/compare_fm_vs_handcrafted.py)) is
  apples-to-apples. (Can still cross-check against the GCS `merged_features.csv` later.)

**Implications:**
- There are **no raw EEG signals** in the old repo — for Option A (foundation-model
  EEG over the full high-density montage) we **must download raw MIPDB**.
- The eye-tracking feature file is empty, so re-derive eye-tracking from raw MIPDB.
- Keep the old CSVs only as a reference/sanity check against the new pipeline.

## 0.2 Datasets to download (in order)

1. **CMI MIPDB (primary cohort) — download FIRST.**
   - S3: `s3://fcp-indi/data/Projects/EEG_Eyetracking_CMI_data`
   - Public bucket (no AWS credentials; use `--no-sign-request`).
   - Per-subject `.tar.gz`, ~2.5 GB each; includes EEG (`.raw`/EGI), eye-tracking, and a
     `.sfp` channel-location file. Both raw and preprocessed are available.
   - License: CC BY-NC-SA (non-commercial).

2. **CMI HBN Release 1 (external validation) — download later (Phase 4).**
   - S3: `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1`
   - OpenNeuro `ds005505`: ~103 GB, 129 channels, 136 participants, BIDS format.
   - 11 releases exist (ds005505–ds005515); start with R1, add more if needed.

3. **(External eye-tracking ADHD cohort)** — confirm access in Phase 4.

## 0.3 Foundation model: LaBraM

- Repo: https://github.com/935963004/LaBraM (ICLR 2024). Also available via `braindecode`
  (`braindecode.models.Labram`) for easier integration.
- **Required EEG preprocessing before LaBraM:** bandpass **0.1–75 Hz**, **50 Hz notch**,
  **resample to 200 Hz**, then segment into fixed-length patches.
- Flexible across channel sets (trained on a 128-ch layout — matches the EGI net here).
- **Plan:** use it **frozen** to extract per-window embeddings (defensible for N≈126;
  avoids overfitting). Cache embeddings to `$SCRATCH`, then train the light fusion head.
- Alternatives to keep in mind: EEGPT, HEAR (heterogeneous electrodes), and the EEG-Bench
  clinical benchmark for sanity-checking.

## 0.4 Running on the Alliance cluster

**Storage:**
- `$SCRATCH` — large, fast, **not backed up**, purged (~60 days). Put raw data + cached
  embeddings here.
- `/project` — backed up, quota'd. Put things you can't re-download.
- `$HOME` — small quota; code/venv only.

**Cluster: Nibi** (`anzer@*.nibi`) — has **miniconda3** and **CUDA 11.8** already.
- Use **conda** (not virtualenv) for the env.
- Install a **cu118** PyTorch build to match CUDA 11.8.

```bash
# one-time env (login node)
conda create -n mh python=3.10 -y && conda activate mh
pip install torch --index-url https://download.pytorch.org/whl/cu118   # match CUDA 11.8
pip install mne braindecode awscli numpy pandas scikit-learn
```

**Download (LOGIN node — compute nodes have no internet):**
```bash
conda activate mh                       # awscli installed above

# Explore before syncing (public bucket → --no-sign-request):
aws s3 ls --no-sign-request s3://fcp-indi/data/Projects/EEG_Eyetracking_CMI_data/

# Sync MIPDB into scratch:
mkdir -p $SCRATCH/mh/data/mipdb
aws s3 sync --no-sign-request \
    s3://fcp-indi/data/Projects/EEG_Eyetracking_CMI_data/ \
    $SCRATCH/mh/data/mipdb/
```
For very large transfers (full HBN, multiple releases), prefer **Globus**
(endpoint: "Digital Research Alliance of Canada") over `aws s3 sync`.

**Compute jobs:** submit via SLURM (`sbatch`), request a GPU only for the steps that
need it (FM embedding extraction; fusion-head training is light). Never run heavy work
on login nodes.

## 0.5 Backup discipline (server-down safety)
- **Code/docs/scripts → GitHub** (this repo) — push after every working session.
- **Raw data → re-downloadable from S3**, so not backed up.
- **Processed embeddings/checkpoints** → keep a copy in `/project` (backed up) since
  re-extraction is slow; large ones can also go to Alliance nearline.

## 0.6 Phase 0 checklist
- [x] Alliance account confirmed — cluster is **Nibi** (miniconda3 + CUDA 11.8 present).
- [ ] MIPDB synced to `$SCRATCH`; inspect one subject `.tar.gz` (EEG + `.sfp` present).
- [ ] LaBraM checkpoint obtained; extract embeddings for one MIPDB subject end-to-end.
- [ ] Repo cloned on cluster; GitHub auth working (token/SSH) for push-back.

---
### Sources
- MIPDB EEG portal: http://fcon_1000.projects.nitrc.org/indi/cmi_eeg/eeg.html · FCP-INDI S3: http://fcon_1000.projects.nitrc.org/indi/s3/index.html · AWS registry: https://registry.opendata.aws/fcp-indi/
- HBN R1: https://openneuro.org/datasets/ds005505 · https://github.com/OpenNeuroDatasets/ds005505
- LaBraM: https://github.com/935963004/LaBraM · braindecode: https://braindecode.org/
- Alliance data movement (Globus): https://alliancecan.ca/en/services/advanced-research-computing/national-services/data-movement-globus
