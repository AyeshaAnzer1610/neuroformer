#!/usr/bin/env bash
# Back up Phase 1 outputs (embeddings + comparison results) OFF-SITE to GitHub,
# so a Nibi outage or $SCRATCH purge can't lose them. Code is already on GitHub;
# this covers the only Nibi-only artifacts. Run AFTER the extraction job:
#
#   bash scripts/backup_results.sh
#
# Embeddings are bundled into one .tar.gz (small, ~10-20 MB) and force-added
# past .gitignore. Re-running just refreshes the bundle.
set -euo pipefail
cd ~/mental-health
git pull --rebase --autostash || true

PROC=$SCRATCH/mh/data
BUNDLE=results/phase1_artifacts.tar.gz
mkdir -p results

# Bundle whatever embedding dirs exist (don't fail if one is missing).
DIRS=()
for d in processed processed_handcrafted processed_eye processed_behaviour; do
    [ -d "$PROC/$d" ] && DIRS+=("$d")
done
if [ ${#DIRS[@]} -eq 0 ]; then
    echo "No embedding dirs under $PROC yet — run after the extraction jobs."; exit 1
fi
echo "Bundling: ${DIRS[*]}"
tar czf "$BUNDLE" -C "$PROC" "${DIRS[@]}"
echo "Bundle size: $(du -h "$BUNDLE" | cut -f1)"

# Force-add past .gitignore (small data files worth versioning) + all results CSVs.
git add -f "$BUNDLE"
for c in results/*.csv; do [ -f "$c" ] && git add -f "$c"; done

git commit -m "Back up embeddings (EEG/eye/behaviour) + Phase 1-2 results" \
  || { echo "Nothing new to back up."; exit 0; }
git push origin main
echo "Backed up to GitHub. Restore later with: tar xzf $BUNDLE -C \$SCRATCH/mh/data"
