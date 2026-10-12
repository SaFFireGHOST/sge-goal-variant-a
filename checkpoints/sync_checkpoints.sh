#!/usr/bin/env bash
# Helper script to sync model checkpoints with Modal persistent volume or Hugging Face
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    echo "Usage: $0 [--push-modal | --pull-modal | --push-hf <repo_id>]"
    echo ""
    echo "Options:"
    echo "  --push-modal           Upload local checkpoints to Modal volume 'rm-policy-ckpts'"
    echo "  --pull-modal           Download checkpoints from Modal volume 'rm-policy-ckpts'"
    echo "  --push-hf <repo_id>    Upload checkpoints to Hugging Face Model Hub"
    exit 1
}

if [ "$1" == "--push-modal" ]; then
    echo "==> Uploading checkpoints to Modal volume 'rm-policy-ckpts'..."
    modal volume put rm-policy-ckpts calvin-fer-mask-prediction/pytorch_lora_weights.safetensors /ckpts/calvin-fer-mask-prediction.safetensors
    modal volume put rm-policy-ckpts calvin-image-editing/pytorch_lora_weights.safetensors /ckpts/calvin-image-editing.safetensors
    echo "==> Done uploading to Modal volume."

elif [ "$1" == "--pull-modal" ]; then
    echo "==> Pulling checkpoints from Modal volume 'rm-policy-ckpts'..."
    mkdir -p calvin-fer-mask-prediction calvin-image-editing
    modal volume get rm-policy-ckpts /ckpts/calvin-fer-mask-prediction.safetensors calvin-fer-mask-prediction/pytorch_lora_weights.safetensors
    modal volume get rm-policy-ckpts /ckpts/calvin-image-editing.safetensors calvin-image-editing/pytorch_lora_weights.safetensors
    echo "==> Done pulling from Modal volume."

elif [ "$1" == "--push-hf" ]; then
    if [ -z "$2" ]; then
        echo "Error: please provide a Hugging Face repository ID, e.g., username/sge-calvin-lora"
        exit 1
    fi
    echo "==> Uploading to Hugging Face repository $2..."
    huggingface-cli upload "$2" .
    echo "==> Done uploading to Hugging Face."

else
    usage
fi
