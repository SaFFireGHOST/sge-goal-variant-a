"""Modal cloud runner for Stage A2 Flow-DPO alignment.

Launches factorial rollout advantage estimation and Flow-DPO LoRA training on Modal.

Usage:
    modal run modal_train_a_flow_dpo.py --action rollouts
    modal run modal_train_a_flow_dpo.py --action train --stage step3
"""

from __future__ import annotations

import os
import subprocess
import sys
import modal

APP_NAME = "sge-variant-a-flow-dpo"
GPU_TYPE = "B200"

data_vol = modal.Volume.from_name("rm-policy-calvin-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("rm-policy-ckpts", create_if_missing=True)
runs_vol = modal.Volume.from_name("rm-policy-runs", create_if_missing=True)

VOLUMES = {
    "/data": data_vol,
    "/ckpts": ckpt_vol,
    "/runs": runs_vol,
}

app = modal.App(name=APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.2.0",
        "accelerate>=0.28.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "numpy",
        "pillow",
        "tqdm",
    )
)


@app.function(image=image, gpu=GPU_TYPE, timeout=86400, volumes=VOLUMES)
def run_rollouts_remote(num_tasks: int = 200):
    print("=== Launching Factorial Rollout Advantage Collection on Modal ===")
    cmd = [
        sys.executable,
        "/workspace/factorial_rollouts.py",
        "--num_tasks", str(num_tasks),
        "--output_dir", "/runs/variant_a_rollouts",
    ]
    res = subprocess.run(cmd)
    runs_vol.commit()
    return res.returncode


@app.function(image=image, gpu=GPU_TYPE, timeout=86400, volumes=VOLUMES)
def run_flow_dpo_remote(stage: str = "step3"):
    print(f"=== Launching Flow-DPO Training on Modal for {stage.upper()} ===")
    pref_file = f"/runs/variant_a_rollouts/{stage}_preferences.json"
    if not os.path.exists(pref_file):
        # Fallback to local script default
        pref_file = f"/runs/variant_a_rollouts/{'step3_appearance' if stage == 'step3' else 'step2_mask'}_preferences.json"

    cmd = [
        sys.executable,
        "/workspace/train_a_flow_dpo.py",
        "--preference_data_path", pref_file,
        "--target_stage", stage,
        "--output_dir", f"/runs/variant_a_flow_dpo_{stage}",
        "--train_batch_size", "8",
        "--num_train_epochs", "5",
        "--learning_rate", "2e-5",
    ]
    res = subprocess.run(cmd)
    runs_vol.commit()
    return res.returncode


@app.local_entrypoint()
def main(action: str = "rollouts", stage: str = "step3"):
    if action == "rollouts":
        print("Triggering remote factorial rollouts...")
        run_rollouts_remote.remote()
    elif action == "train":
        print(f"Triggering remote Flow-DPO training for {stage}...")
        run_flow_dpo_remote.remote(stage=stage)
    else:
        print(f"Unknown action: {action}. Use --action rollouts or --action train")
