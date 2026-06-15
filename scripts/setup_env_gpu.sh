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

# braindecode + deps + Alliance's CUDA-enabled torch, all from the cluster
# wheelhouse (--no-index). This torch works with the cuda/cudnn modules loaded
# in the GPU job; no external cu121 wheel needed.
pip install --no-index "braindecode>=1.5" huggingface_hub safetensors mne \
    scikit-learn pandas torch torchvision torchaudio

python -c "import torch; print('torch', torch.__version__, '(CUDA check happens in the GPU job)')"
echo "GPU env ready at ~/envs/mh311gpu. In the SLURM GPU job, load: cuda cudnn."
