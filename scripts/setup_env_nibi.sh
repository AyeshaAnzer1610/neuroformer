#!/usr/bin/env bash
# Reproducible Nibi (Digital Research Alliance) environment for this project.
# This is the recipe that actually works around Alliance's pinned wheelhouse.
# Run ONCE on a Nibi LOGIN node:  bash scripts/setup_env_nibi.sh
# Activate in future sessions with:
#   module load StdEnv/2023 python/3.11 && source ~/envs/mh311/bin/activate
set -euo pipefail

# 1. Stop conda auto-activating — its CUDA libs clash with torch.
conda config --set auto_activate_base false 2>/dev/null || true
conda deactivate 2>/dev/null || true

# 2. Python 3.11 (braindecode>=1.5 / InterpolatedLaBraM needs >=3.11; Alliance
#    pins braindecode to 1.2.0, which lacks InterpolatedLaBraM).
module load StdEnv/2023 python/3.11

# 3. Fresh venv seeded from Alliance wheels.
virtualenv --no-download ~/envs/mh311
source ~/envs/mh311/bin/activate
pip install --no-index --upgrade pip

# 4. braindecode>=1.5 + HF loaders from PyPI (PIP_CONSTRAINT= blanks the pin).
PIP_CONSTRAINT= pip install --index-url https://pypi.org/simple \
    "braindecode>=1.5" huggingface_hub safetensors

# 5. Matched CPU torch trio from PyTorch's index, IGNORING Alliance pip config
#    (PIP_CONFIG_FILE=/dev/null) — the wheelhouse has torch 2.12 but no matching
#    torchaudio, which caused the libtorchaudio ABI crash. Pinned + force.
PIP_CONFIG_FILE=/dev/null pip install --force-reinstall --no-deps \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.6.0" "torchaudio==2.6.0" "torchvision==0.21.0"

python -c "import torch,torchaudio,braindecode,mne; \
print('torch',torch.__version__,'audio',torchaudio.__version__,\
'braindecode',braindecode.__version__,'mne',mne.__version__)"
echo "Env ready. (CPU torch — for GPU jobs build a separate CUDA env inside the SLURM job.)"
