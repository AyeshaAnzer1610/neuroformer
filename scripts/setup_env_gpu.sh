#!/usr/bin/env bash
# One-time GPU environment for fine-tuning LaBraM on Nibi (H100 / MIG).
# The CPU `mh311` env can't fine-tune (CPU-only torch); this builds a CUDA env.
# Run ONCE on a LOGIN node:  bash scripts/setup_env_gpu.sh
# Then, inside a GPU SLURM job:
#   module load StdEnv/2023 python/3.11 && source ~/envs/mh311gpu/bin/activate
set -euo pipefail

module load StdEnv/2023 python/3.11
virtualenv --no-download ~/envs/mh311gpu
source ~/envs/mh311gpu/bin/activate
pip install --no-index --upgrade pip

# braindecode>=1.5 (InterpolatedLaBraM) + HF loaders from PyPI
PIP_CONSTRAINT= pip install --index-url https://pypi.org/simple \
    "braindecode>=1.5" huggingface_hub safetensors mne scikit-learn pandas

# CUDA build of torch matching Nibi's H100s (CUDA 12.x). Ignore Alliance pins.
PIP_CONFIG_FILE=/dev/null pip install --no-deps --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.4.1" "torchaudio==2.4.1" "torchvision==0.19.1"

python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
echo "GPU env ready at ~/envs/mh311gpu (request a GPU in the SLURM job to use it)."
