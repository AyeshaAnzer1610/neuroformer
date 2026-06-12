#!/usr/bin/env bash
# Download datasets for the mental-health project.
# RUN THIS ON A LOGIN NODE of the Digital Research Alliance of Canada cluster
# (compute nodes have no internet). Data goes to $SCRATCH (not backed up, but
# re-downloadable). Public FCP-INDI bucket → no AWS credentials needed.
#
# Usage:
#   bash scripts/download_data.sh explore             # list buckets, download nothing
#   bash scripts/download_data.sh mipdb-subjects 3    # first N MIPDB subjects (test set)
#   bash scripts/download_data.sh mipdb               # full MIPDB cohort
#   bash scripts/download_data.sh hbn-r1              # external validation (Phase 4)
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
    echo "Nothing downloaded. Re-run with 'mipdb-subjects N', 'mipdb', or 'hbn-r1'."
    ;;
  mipdb-subjects)
    N="${2:-3}"
    DEST="$DATA_ROOT/mipdb"
    mkdir -p "$DEST"
    echo "Detecting the first $N MIPDB subjects under $MIPDB_S3/ ..."
    LISTING="$(aws s3 ls --no-sign-request "$MIPDB_S3/")"
    # Handle either layout: folders show as 'PRE name/'; tarballs end in .tar.gz
    mapfile -t ITEMS < <(printf '%s\n' "$LISTING" \
        | awk '/PRE/ {print $2} /\.tar\.gz$/ {print $NF}' | head -n "$N")
    if [ "${#ITEMS[@]}" -eq 0 ]; then
      echo "Could not detect subject entries. Raw listing (first lines):"
      printf '%s\n' "$LISTING" | head
      exit 1
    fi
    echo "Downloading ${#ITEMS[@]}: ${ITEMS[*]}"
    for IT in "${ITEMS[@]}"; do
      echo "  -> $IT"
      if [[ "$IT" == */ ]]; then
        aws s3 cp --no-sign-request --recursive "$MIPDB_S3/$IT" "$DEST/$IT"
      else
        aws s3 cp --no-sign-request "$MIPDB_S3/$IT" "$DEST/$IT"
      fi
    done
    echo "Done: ${#ITEMS[@]} subject(s) in $DEST"
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
    echo "Unknown target '$1'. Use: explore | mipdb-subjects N | mipdb | hbn-r1"
    exit 1
    ;;
esac
