#!/usr/bin/env bash
# One-time setup of a conda env for local CALVIN + GR-MG rollouts (Gate 0).
# Mirrors the Modal image in policy/modal_calvin_eval.py. No sudo needed if the
# server already has NVIDIA drivers (PyBullet renders through the driver's EGL).
#
# Usage:  bash variant_a_alignment/setup_local_env.sh ~/calvin
set -euo pipefail

CALVIN_DIR="${1:-$HOME/calvin}"
ENV_NAME="${ENV_NAME:-gate0}"

eval "$(conda shell.bash hook)"
export PYTHONNOUSERSITE=1   # keep ~/.local packages out of the env
conda create -y -n "$ENV_NAME" python=3.10
conda activate "$ENV_NAME"

pip install numpy==1.23.5
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install scipy pillow pyarrow pandas tqdm termcolor matplotlib "imageio[ffmpeg]" \
    opencv-python-headless omegaconf hydra-core==1.1.1 pytorch-lightning==1.8.6 \
    transformers==4.36.2 diffusers==0.27.2 accelerate huggingface_hub datasets \
    sentence-transformers ftfy regex pybullet open3d moviepy imageio-ffmpeg requests \
    flamingo-pytorch einops timm "git+https://github.com/openai/CLIP.git" modal

pip install "setuptools==65.5.0" "wheel==0.38.4"
pip install "pip<24.1"   # gym 0.21 metadata is rejected by pip>=24.1
pip install --no-build-isolation "gym==0.21.0"

if [ ! -d "$CALVIN_DIR" ]; then
  git clone --recurse-submodules https://github.com/mees/calvin.git "$CALVIN_DIR"
fi
pip install "setuptools<58" wheel
(cd "$CALVIN_DIR/calvin_env" && pip install -e .) || true
(cd "$CALVIN_DIR/calvin_env/tacto" && pip install -e .) || true
(cd "$CALVIN_DIR/calvin_models" && pip install --no-deps -e .)
pip install pyhash || echo "WARN: pyhash failed (needs clang); CALVIN rollouts usually work without it"

pip install --force-reinstall numpy==1.23.5
python -c "import torch, pybullet, calvin_env, clip; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "env '$ENV_NAME' ready; CALVIN at $CALVIN_DIR"
