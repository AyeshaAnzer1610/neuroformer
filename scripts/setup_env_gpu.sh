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
# Pin the MATCHING trio (torch 2.11.0 + torchvision + torchaudio 2.11.0). Do NOT
# install the latest torch (2.12): the wheelhouse has no torchaudio 2.12, so pip
# backtracks torchaudio down to 2.1.2, and a torch-2.12/torchaudio-2.1.2 mismatch
# crashes with the libtorchaudio "undefined symbol" error at import. The 2.11.0
# trio is internally consistent and works with the DEFAULT cuda/cudnn modules
# loaded in the SLURM job (do NOT load a specific cuda/12.x -- that mismatch is
# what caused the cusparse / nvJitLink errors before).
pip install --no-index torch==2.11.0 torchvision torchaudio==2.11.0

# Step 2: braindecode is NOT in the Alliance wheelhouse, so install it (and the
# remaining deps) from PyPI -- PIN torch AND torchaudio to the versions just
# installed so pip cannot upgrade/replace the Alliance CUDA build or re-backtrack
# torchaudio.
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
{ echo "torch==${TORCH_VER}"; echo "torchaudio==${TORCH_VER}"; } > ~/mh_gpu_constraints.txt
echo "Pinning torch==${TORCH_VER} (+torchaudio) while pulling braindecode from PyPI."
pip install -c ~/mh_gpu_constraints.txt \
    "braindecode>=1.5" huggingface_hub safetensors mne scikit-learn pandas scipy

# Verify torch was not swapped out and is still the CUDA build.
python -c "import torch, braindecode; print('torch', torch.__version__, \
    '| braindecode', braindecode.__version__, '(CUDA check happens in the GPU job)')"
echo "GPU env ready at ~/envs/mh311gpu. In the SLURM GPU job, load: cuda cudnn."
