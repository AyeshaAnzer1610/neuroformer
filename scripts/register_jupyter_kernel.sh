#!/usr/bin/env bash
# Register the mh311 virtualenv as a Jupyter kernel so it appears in the
# Nibi Open OnDemand JupyterLab kernel list as "Python (mh311)".
# Run ONCE on a Nibi LOGIN node, after the mh311 venv already exists.
set -euo pipefail

ENVDIR="$HOME/envs/mh311"
[ -d "$ENVDIR" ] || { echo "Venv $ENVDIR not found — create it first."; exit 1; }

module load StdEnv/2023 python/3.11
source "$ENVDIR/bin/activate"
pip install --no-index ipykernel 2>/dev/null || pip install ipykernel

# Launcher: reloads the module + activates the venv every time the kernel starts
# (Alliance venvs depend on the python module being loaded at runtime).
LAUNCH="$ENVDIR/kernel-launcher.sh"
cat > "$LAUNCH" <<'EOF'
#!/bin/bash
module load StdEnv/2023 python/3.11
source "$HOME/envs/mh311/bin/activate"
exec python -m ipykernel_launcher "$@"
EOF
chmod +x "$LAUNCH"

KDIR="$HOME/.local/share/jupyter/kernels/mh311"
mkdir -p "$KDIR"
cat > "$KDIR/kernel.json" <<EOF
{
 "argv": ["$LAUNCH", "-f", "{connection_file}"],
 "display_name": "Python (mh311)",
 "language": "python"
}
EOF

echo "OK: registered kernel 'Python (mh311)' at $KDIR"
echo "In OOD JupyterLab, choose the 'Python (mh311)' kernel."
