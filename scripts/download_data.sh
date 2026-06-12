#!/usr/bin/env bash
# Download datasets for the mental-health project.
# RUN THIS ON A LOGIN NODE of the Digital Research Alliance of Canada cluster
# (compute nodes have no internet). Data goes to $SCRATCH (not backed up, but
# re-downloadable). Public FCP-INDI bucket → no AWS credentials needed.
#
# Usage:
#   bash scripts/download_data.sh mipdb        # primary cohort (download first)
#   bash scripts/download_data.sh hbn-r1       # external validation (Phase 4)
#   bash scripts/download_data.sh explore      # just list the buckets, download nothing
set -euo pipefail

: "${SCRATCH:?SCRATCH is not set — are you on an Alliance cluster login node?}"
DATA_ROOT="$SCRATCH/mh/data"
mkdir -p "$DATA_ROOT"

# Ensure awscli is available (install into the active venv if missing).
if ! command -v aws >/dev/null 2>&1; then
  echo "awscli not found; installing into the active Python environment..."
  pip install --no-index awscli 2>/dev/null || pip install awscli
fi

MIPDB_S3="s3://fcp-indi/data/Projects/EEG_Eyetracking_CMI_data"
HBN_R1_S3="s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1"

case "${1:-explore}" in
  explore)
    echo "== MIPDB =="
    aws s3 ls --no-sign-request "$MIPDB_S3/"
    echo "== HBN R1 =="
    aws s3 ls --no-sign-request "$HBN_R1_S3/"
    echo "Nothing downloaded. Re-run with 'mipdb' or 'hbn-r1' to sync."
    ;;
  mipdb)
    DEST="$DATA_ROOT/mipdb"
    mkdir -p "$DEST"
    echo "Syncing MIPDB (~per-subject 2.5 GB tarballs) into $DEST ..."
    aws s3 sync --no-sign-request "$MIPDB_S3/" "$DEST/"
    echo "Done. Inspect one subject and confirm the .sfp channel-location file is present."
    ;;
  hbn-r1)
    DEST="$DATA_ROOT/hbn_r1"
    mkdir -p "$DEST"
    echo "Syncing HBN Release 1 (~103 GB) into $DEST ..."
    echo "TIP: for multi-release / very large transfers, use Globus instead."
    aws s3 sync --no-sign-request "$HBN_R1_S3/" "$DEST/"
    echo "Done."
    ;;
  *)
    echo "Unknown target '$1'. Use: explore | mipdb | hbn-r1"
    exit 1
    ;;
esac
