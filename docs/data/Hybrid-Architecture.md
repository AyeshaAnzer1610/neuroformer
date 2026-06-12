# Hybrid data & compute architecture

Confirmed setup: we use **both** Google Cloud Storage (durable/company copy) and the
Digital Research Alliance of Canada **Nibi** cluster (compute). This note pins down what
lives where so nothing is ambiguous.

## What lives where

| Layer | Location | Backed up? | Notes |
|---|---|---|---|
| **Source data** | Public AWS S3 (`s3://fcp-indi/...`) | n/a (public) | MIPDB + HBN; always re-downloadable, free egress (AWS Open Data). |
| **Durable copy** | **GCS** (`gs://awe-hbn-eeg/...`, your GCP project) | yes (GCS) | Company-owned persistent copy. Populate via `docs/data/HBN-GCS-transfer-runbook.md`. |
| **Compute working set** | **Nibi `$SCRATCH`** | ❌ purged ~60d | Raw data + cached FM embeddings during active work. |
| **Persistent results** | Nibi `/project` (+ nearline) | yes | Checkpoints, extracted embeddings (slow to recompute). |
| **Code + docs** | **GitHub** (`AyeshaAnzer1610/mental-health`, private) | yes | The durable backup for everything small. Push every session. |

## How the two clouds fit together (no conflict)

- **S3 is the single source of truth.** Both GCS and Nibi pull from it.
- For **compute on Nibi**, pull data **straight from S3 to `$SCRATCH`** (`scripts/download_data.sh`)
  — it's free and simplest. You do **not** need to route through GCS for training.
- **GCS** is the *durable/company archive* (and useful if non-Alliance collaborators or your
  awedigitalwellness infra need access). Fill it once via the STS runbook; it's not on the
  training hot path.
- Net: **don't pay to stage HBN through GCS for Nibi jobs** — GCS is the archive, Nibi pulls
  from S3 directly. Use GCS only when you need a persistent, org-controlled copy.

## Backup discipline (server-down safety)
- Code/docs → **GitHub** after every working session (already set up).
- Raw data → **re-downloadable from S3**, no backup needed.
- Embeddings/checkpoints → keep in Nibi `/project` (backed up) and/or GCS.
- Scratch can vanish (purge or outage) without losing anything irreplaceable.
