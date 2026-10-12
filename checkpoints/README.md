# Model Checkpoints & Cloud Volume Sync

This directory contains configuration metadata and scripts for synchronizing pre-trained model weights.

In accordance with modern cloud machine learning best practices, **large model binaries (`.safetensors`, `.pt`) are decoupled from Git** to keep the repository lightweight (< 500 KB) and prevent Git LFS bandwidth quotas from breaking automated cloud deployments.

---

## Pre-trained LoRA Checkpoints

1. **Step 2 (Mask Generation LoRA):**
   - Path: `calvin-fer-mask-prediction/pytorch_lora_weights.safetensors`
   - Trained on CALVIN FER Mask Prediction (Epoch 22)
   - Size: 104 MB
2. **Step 3 (Appearance Completion LoRA):**
   - Path: `calvin-image-editing/pytorch_lora_weights.safetensors`
   - Trained on CALVIN Image Editing (Epoch 18)
   - Size: 104 MB

---

## 1. Push Checkpoints to Modal Persistent Volume

To make these checkpoints instantly accessible to all serverless Modal jobs:

```bash
# Upload Step 2 mask LoRA
modal volume put rm-policy-ckpts \
    calvin-fer-mask-prediction/pytorch_lora_weights.safetensors \
    /ckpts/calvin-fer-mask-prediction.safetensors

# Upload Step 3 appearance LoRA
modal volume put rm-policy-ckpts \
    calvin-image-editing/pytorch_lora_weights.safetensors \
    /ckpts/calvin-image-editing.safetensors
```

Alternatively, run the automated helper:
```bash
bash sync_checkpoints.sh --push-modal
```

---

## 2. Push Checkpoints to Hugging Face Hub (Free & Unlimited Bandwidth)

```bash
# Authenticate
huggingface-cli login

# Upload checkpoints folder to your Hugging Face model repository
huggingface-cli upload <YOUR_HF_USERNAME>/sge-goal-calvin-lora .
```
