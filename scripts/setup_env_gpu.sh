#!/usr/bin/env bash
# One-time GPU environment for fine-tuning LaBraM on Nibi (H100 / MIG).
# Uses Alliance's OWN CUDA-enabled torch (built for the cluster) -- do NOT swap it
# for a pip cu121 wheel (that can't find the cluster's libcudnn).
# Run ONCE on a LOGIN node:  bash scripts/setup_env_gpu.sh
# Then, inside a GPU SLURM job, load the CUDA stack:
#   module load StdEnv/2023 gcc python/3.11 cuda cudnn && source ~/envs/mh311gpu/bin/activate
set -euo pipefail

module load StdEnv/2023 python/3.11
rm -rf ~/envs/mh311gpu
virtualenv --no-download ~/envs/mh311gpu
source ~/envs/mh311gpu/bin/activate
pip install --no-index --upgrade pip

# Step 1: CUDA-enabled torch from the Alliance wheelhouse ONLY (--no-index).
# This is the torch that matches the cluster's cuda/cudnn modules -- do NOT let a
# PyPI cu121 wheel replace it (that caused the libcudnn / cusparse / nvJitLink
# "undefined symbol" failures before).
pip install --no-index torch torchvision torchaudio

# Step 2: braindecode is NOT in the Alliance wheelhouse, so install it (and the
# remaining deps) from PyPI -- but PIN torch to the version just installed via a
# constraints file, so pip cannot upgrade/replace the Alliance CUDA torch.
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo "torch==${TORCH_VER}" > ~/mh_gpu_constraints.txt
echo "Pinning torch==${TORCH_VER} while pulling braindecode from PyPI."
pip install -c ~/mh_gpu_constraints.txt \
    "braindecode>=1.5" huggingface_hub safetensors mne scikit-learn pandas scipy

# Verify torch was not swapped out and is still the CUDA build.
python -c "import torch, braindecode; print('torch', torch.__version__, \
    '| braindecode', braindecode.__version__, '(CUDA check happens in the GPU job)')"
echo "GPU env ready at ~/envs/mh311gpu. In the SLURM GPU job, load: cuda cudnn."
