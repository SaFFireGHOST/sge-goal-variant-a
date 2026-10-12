#!/usr/bin/env bash
# Helper script to sync model checkpoints with Modal persistent volume or Hugging Face
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure modal executable is found
if command -v modal &> /dev/null; then
    MODAL_CMD="modal"
elif [ -f "$HOME/.local/bin/modal" ]; then
    MODAL_CMD="$HOME/.local/bin/modal"
else
    MODAL_CMD="python3 -m modal"
fi

DEFAULT_HF_REPO="SaFFire/sge-goal"

usage() {
    echo "Usage: $0 [--push-modal | --pull-modal | --push-hf <repo_id> | --pull-hf [repo_id]]"
    echo ""
    echo "Options:"
    echo "  --push-modal           Upload local checkpoints to Modal volume 'rm-policy-ckpts'"
    echo "  --pull-modal           Download checkpoints from Modal volume 'rm-policy-ckpts'"
    echo "  --push-hf <repo_id>    Upload checkpoints to Hugging Face Model Hub"
    echo "  --pull-hf [repo_id]    Download checkpoints from Hugging Face Model Hub (default: $DEFAULT_HF_REPO)"
    exit 1
}

if [ "$1" == "--push-modal" ]; then
    echo "==> Ensuring Modal volume 'rm-policy-ckpts' exists..."
    $MODAL_CMD volume create rm-policy-ckpts 2>/dev/null || true

    echo "==> Uploading checkpoints to Modal volume 'rm-policy-ckpts' using $MODAL_CMD..."
    $MODAL_CMD volume put rm-policy-ckpts calvin-fer-mask-prediction/pytorch_lora_weights.safetensors /ckpts/calvin-fer-mask-prediction.safetensors
    $MODAL_CMD volume put rm-policy-ckpts calvin-image-editing/pytorch_lora_weights.safetensors /ckpts/calvin-image-editing.safetensors
    echo "==> Done uploading to Modal volume."

elif [ "$1" == "--pull-modal" ]; then
    echo "==> Pulling checkpoints from Modal volume 'rm-policy-ckpts' using $MODAL_CMD..."
    mkdir -p calvin-fer-mask-prediction calvin-image-editing
    $MODAL_CMD volume get rm-policy-ckpts /ckpts/calvin-fer-mask-prediction.safetensors calvin-fer-mask-prediction/pytorch_lora_weights.safetensors
    $MODAL_CMD volume get rm-policy-ckpts /ckpts/calvin-image-editing.safetensors calvin-image-editing/pytorch_lora_weights.safetensors
    echo "==> Done pulling from Modal volume."

elif [ "$1" == "--push-hf" ]; then
    TARGET_REPO="${2:-$DEFAULT_HF_REPO}"
    echo "==> Uploading to Hugging Face repository $TARGET_REPO..."
    python3 upload_to_hf.py --repo-id "$TARGET_REPO"
    echo "==> Done uploading to Hugging Face."

elif [ "$1" == "--pull-hf" ]; then
    TARGET_REPO="${2:-$DEFAULT_HF_REPO}"
    echo "==> Pulling checkpoints from Hugging Face model repository $TARGET_REPO..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$TARGET_REPO', local_dir='.')"
    echo "==> Done pulling from Hugging Face."

else
    usage
fi
