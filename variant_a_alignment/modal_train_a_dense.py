"""Modal launcher for Stage A1: Dense Behavioral Matching Training (Variant A).

Deploys the Stage A1 training loop to a remote cloud GPU (B200 / H100) using Modal.
Connects with persistent Modal volumes for CALVIN data, policy checkpoints, and LoRA weights.

Usage:
    modal run modal_train_a_dense.py
"""

from __future__ import annotations

import os
from pathlib import Path
import modal

APP_NAME = "sge-variant-a-dense"
GPU_TYPE = "B200"  # Or "h100", "a100-80gb"

# Volume mounts
data_vol = modal.Volume.from_name("rm-policy-calvin-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("rm-policy-ckpts", create_if_missing=True)
runs_vol = modal.Volume.from_name("rm-policy-runs", create_if_missing=True)

VOLUMES = {
    "/data": data_vol,
    "/ckpts": ckpt_vol,
    "/runs": runs_vol,
}

app = modal.App(name=APP_NAME)
current_dir = Path(__file__).parent.resolve()

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.2.0",
        "torchvision",
        "accelerate>=0.28.0",
        "diffusers>=0.27.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "datasets",
        "tqdm",
        "pillow",
        "numpy",
    )
    .add_local_dir(
        str(current_dir),
        remote_path="/workspace",
        ignore=["*.git*", "__pycache__", "*.pyc", "*.safetensors", "*.pt", "*.pth", "*.bin"],
    )
)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=86400,
    volumes=VOLUMES,
)
def run_stage_a_dense_training():
    import subprocess
    import sys

    print("=== Starting Stage A1 Cloud Behavioral Matching Training on Modal ===")
    
    # Check GPU availability
    import torch
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    output_dir = "/runs/variant_a_dense_checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "/workspace/train_a_dense.py",
        "--pretrained_model_name_or_path", "black-forest-labs/FLUX.1-Kontext-dev",
        "--policy_checkpoint_path", "/ckpts/gr_mg/epoch=47-step=83712.ckpt",
        "--dataset_name", "SuyashAjitChavan2002/Calvin-Dataset-Final",
        "--output_dir", output_dir,
        "--lambda_beh", "1.5",
        "--train_batch_size", "8",
        "--gradient_accumulation_steps", "2",
        "--num_train_epochs", "15",
        "--mixed_precision", "bf16",
    ]

    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    # Flush output volume
    runs_vol.commit()
    print(f"Stage A1 Training Finished with exit code: {result.returncode}")
    return result.returncode


@app.local_entrypoint()
def main():
    print(f"Launching Stage A1 Training on Modal ({GPU_TYPE})...")
    run_stage_a_dense_training.remote()
