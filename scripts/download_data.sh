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
HBN_BASE="s3://fcp-indi/data/Projects/HBN/BIDS_EEG"
HBN_R1_S3="$HBN_BASE/cmi_bids_R1"

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
  hbn-subjects)
    N="${2:-3}"
    DEST="$DATA_ROOT/hbn_r1"
    mkdir -p "$DEST"
    echo "HBN root listing:"
    aws s3 ls --no-sign-request "$HBN_R1_S3/"
    # BIDS root metadata (labels live in participants.tsv).
    for f in participants.tsv participants.json dataset_description.json; do
      aws s3 cp --no-sign-request "$HBN_R1_S3/$f" "$DEST/$f" 2>/dev/null \
        && echo "got $f" || true
    done
    echo "Detecting first $N HBN subjects (sub-*/) ..."
    mapfile -t SUBS < <(aws s3 ls --no-sign-request "$HBN_R1_S3/" \
        | awk '/PRE sub-/ {print $2}' | head -n "$N")
    if [ "${#SUBS[@]}" -eq 0 ]; then
      echo "No sub-*/ entries detected. Raw listing above."; exit 1
    fi
    for S in "${SUBS[@]}"; do
      echo "  -> $S"
      aws s3 cp --no-sign-request --recursive "$HBN_R1_S3/$S" "$DEST/$S"
    done
    echo "Done: ${#SUBS[@]} HBN subject(s) + participants.tsv in $DEST"
    ;;
  hbn-rest)
    # Only RestingState files (the paradigm we use) + participants.tsv. Much
    # smaller/faster than the full release (we never touch the other tasks).
    DEST="$DATA_ROOT/hbn_r1"
    mkdir -p "$DEST"
    aws s3 cp --no-sign-request "$HBN_R1_S3/participants.tsv" "$DEST/participants.tsv"
    echo "Syncing only *RestingState* files into $DEST ..."
    aws s3 sync --no-sign-request "$HBN_R1_S3/" "$DEST/" \
        --exclude "*" --include "*task-RestingState*"
    echo "Done (RestingState-only)."
    ;;
  hbn-releases-explore)
    echo "Probing HBN releases cmi_bids_R1..R11 for participants.tsv ..."
    for R in $(seq 1 11); do
      if aws s3 ls --no-sign-request "$HBN_BASE/cmi_bids_R$R/participants.tsv" >/dev/null 2>&1; then
        N=$(aws s3 cp --no-sign-request "$HBN_BASE/cmi_bids_R$R/participants.tsv" - 2>/dev/null | tail -n +2 | wc -l)
        echo "  R$R: present (~$N subjects)"
      else
        echo "  R$R: not found"
      fi
    done
    ;;
  hbn-rest-releases)
    # Download RestingState-only across a range of releases into ONE dir, and
    # build a merged participants_all.tsv (label table for the whole cohort).
    A="${2:-1}"; B="${3:-11}"
    DEST="$DATA_ROOT/hbn_all"
    mkdir -p "$DEST"
    MERGED="$DEST/participants_all.tsv"; : > "$MERGED"; first=1
    for R in $(seq "$A" "$B"); do
      BUCKET="$HBN_BASE/cmi_bids_R$R"
      aws s3 ls --no-sign-request "$BUCKET/participants.tsv" >/dev/null 2>&1 \
        || { echo "R$R: not found, skipping"; continue; }
      echo "== Release R$R =="
      TMP="$DEST/.part_R$R.tsv"
      aws s3 cp --no-sign-request "$BUCKET/participants.tsv" "$TMP"
      if [ "$first" -eq 1 ]; then cat "$TMP" >> "$MERGED"; first=0
      else tail -n +2 "$TMP" >> "$MERGED"; fi
      rm -f "$TMP"
      aws s3 sync --no-sign-request "$BUCKET/" "$DEST/" \
          --exclude "*" --include "*task-RestingState*"
    done
    echo "Done. Merged labels: $MERGED ($(tail -n +2 "$MERGED" | wc -l) subjects)"
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
    echo "Unknown target '$1'. Use: explore | mipdb-subjects N | mipdb | hbn-subjects N | hbn-rest | hbn-r1 | hbn-releases-explore | hbn-rest-releases A B"
    exit 1
    ;;
esac
