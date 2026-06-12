# HBN-EEG → Google Cloud Storage transfer runbook

> Goal: copy the full **Healthy Brain Network EEG** dataset (all 11 BIDS releases) from the
> public AWS S3 bucket into a GCS bucket in our existing GCP project, using **GCP Storage
> Transfer Service (STS)** — server-side, agentless, no local download.
>
> Why not Globus: HBN publishes no Globus endpoint, and writing to GCS via Globus needs a paid
> subscription + a self-hosted Globus Connect Server. All cost, no benefit here. Why not a local
> `gsutil`/`rclone` pull: multi-TB through this machine; STS does it server-side for free.

## Source facts (verified 2026-06-12)

- **S3 source:** `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/` (region `us-east-1`)
- **Releases under that prefix:** `cmi_bids_R1` … `cmi_bids_R11`, plus `cmi_bids_NC`
- **Access:** public-read; HBN is in the **AWS Open Data** program → **S3 egress is sponsored (free)**
- **License:** mixed CC-BY-SA 4.0 and CC-BY-NC-SA 4.0 (NC subjects are *not* on OpenNeuro; S3 has all)
- **Size:** multi-TB (subject folders ~2 GB each, thousands of subjects)

## Cost expectation

- STS transfer fee: **none** (STS is free to run).
- AWS egress: **free** (Open Data sponsorship).
- GCS storage: ~**$0.020/GB/month** Standard → **~$20/TB/month**. Budget accordingly for full set.

---

## Step 0 — Decide destination layout

- Bucket name (globally unique), e.g. `awe-hbn-eeg`
- Region: a **US region** (e.g. `us-east1`) or **US multi-region** — source is `us-east-1`, keeps it close.
- Storage class: **Standard** (move to Nearline/Coldline later if it's archival).
- Destination prefix inside the bucket: `HBN/BIDS_EEG/`

---

## Step 1 — Create the GCS bucket

### Console
Cloud Console → **Cloud Storage → Buckets → Create**:
- Name: `awe-hbn-eeg`
- Location type: **Region** → `us-east1` (or Multi-region `US`)
- Storage class: **Standard**
- Access control: **Uniform** (recommended)
- Leave public access prevention **on**

### gcloud (if you install the SDK)
```bash
gcloud config set project <YOUR_PROJECT_ID>
gcloud storage buckets create gs://awe-hbn-eeg \
  --location=us-east1 \
  --default-storage-class=STANDARD \
  --uniform-bucket-level-access
```

---

## Step 2 — Get AWS read credentials (STS requires an AWS auth method)

STS has **no anonymous-S3 mode**, even for public buckets. Cheapest path: a throwaway read-only
IAM user. Because `fcp-indi` objects are world-readable, any valid AWS identity can read them.

1. Create / log into a free AWS account.
2. IAM → **Users → Create user** (e.g. `sts-hbn-read`), *no* console access needed.
3. Attach policy **`AmazonS3ReadOnlyAccess`**.
4. **Create access key** → type "Application running outside AWS" → save the
   **Access key ID** + **Secret access key**.

(Keep these secret; you can delete the IAM user once the transfer completes.)

---

## Step 3 — Grant STS permission on the destination bucket

STS acts as a Google-managed service agent. When you create the job in the console it will
**offer to grant** the needed roles automatically — accept that. If doing it manually, grant the
STS service agent `roles/storage.objectAdmin` (or `legacyBucketWriter`) on `awe-hbn-eeg`.

Find the agent: Console STS job wizard shows it, or:
```bash
gcloud storage transfer agents ...   # not needed for agentless; the wizard surfaces the agent email
```

---

## Step 4 — Create the transfer job

### Console (recommended, no SDK needed)
Cloud Console → **Storage Transfer Service → Create transfer job**:
1. **Source type:** Amazon S3
2. **Bucket or folder:** `fcp-indi/data/Projects/HBN/BIDS_EEG`
3. **AWS auth:** Access key → paste the key ID + secret from Step 2
   *(Optionally store the secret in Secret Manager and reference it.)*
4. **Destination:** `awe-hbn-eeg`, path `HBN/BIDS_EEG/`
5. **When/how often:** **Run once** (on demand)
6. **Settings:**
   - Overwrite: **only if different** (`checksum/size`)
   - Delete objects in destination: **never**
   - Keep metadata defaults
7. **Create** → **Run**.

### gcloud equivalent (single job, whole prefix)
```bash
gcloud transfer jobs create \
  s3://fcp-indi/data/Projects/HBN/BIDS_EEG/ \
  gs://awe-hbn-eeg/HBN/BIDS_EEG/ \
  --source-creds-file=aws-creds.json \
  --overwrite-when=different
# aws-creds.json: {"awsAccessKeyId":"...","awsSecretAccessKey":"..."}
```

### Optional: one job per release (finer monitoring/retry)
Repeat the job for each prefix — e.g.:
```
s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1/   → gs://awe-hbn-eeg/HBN/BIDS_EEG/cmi_bids_R1/
... R2 … R11, cmi_bids_NC
```
Single-job-over-parent is simpler; STS already parallelizes and retries internally. Split only if
you want per-release progress or to exclude the NC-licensed subset.

---

## Step 5 — Monitor & verify

- Console STS job page shows objects/bytes transferred, errors, and ETA.
- Spot-check after completion:
  ```bash
  gcloud storage ls gs://awe-hbn-eeg/HBN/BIDS_EEG/
  gcloud storage du -s gs://awe-hbn-eeg/HBN/BIDS_EEG/   # total bytes
  ```
- Re-running the job is safe (idempotent): "overwrite only if different" skips already-copied objects.

## Licensing note for the paper

If we'll redistribute or publish derivatives, the **CC-BY-NC-SA** subjects (`cmi_bids_NC` and any
NC-consented subjects) carry a non-commercial restriction. For internal model training/validation
this is fine; flag it before any release of derived data.

---

*Provenance: source `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/`; obtained via GCP Storage Transfer
Service into `gs://awe-hbn-eeg/HBN/BIDS_EEG/`. Facts verified 2026-06-12 against
neuromechanist.github.io/data/hbn and cloud.google.com/storage-transfer docs.*
