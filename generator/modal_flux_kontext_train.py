"""
Modal script to launch Flux Kontext LoRA finetuning with multi-GPU support using accelerate.

This script uses Modal's serverless infrastructure to train the Flux Kontext model
across multiple GPUs using accelerate launch for distributed training.

Usage:
    # Run training
    modal run modal_flux_kontext_train.py --action train

    # Run inference only (after training)
    modal run modal_flux_kontext_train.py --action inference

    # Download dataset first (optional, for caching)
    modal run modal_flux_kontext_train.py --action download
"""

import modal

# ============================================================================
# GPU Configuration - Adjust these variables for your needs
# ============================================================================
NUM_GPUS = 1  # Number of GPUs to use (1, 2, 4, or 8)
GPU_TYPE = "B200"  # GPU type: "a100", "a100-80gb", "h100", etc.
GPU_CONFIG = f"{GPU_TYPE}:{NUM_GPUS}"

# GPU Monitoring Configuration
ENABLE_GPU_MONITORING = False  # Set to False to disable GPU utilization logging
GPU_MONITORING_INTERVAL = 60  # Seconds between GPU utilization logs

# ============================================================================
# Training Configuration - Defaults match the command line setup
# ============================================================================
TRAINING_CONFIG = {
    # Model
    "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-Kontext-dev",
    "revision": None,
    "variant": None,
    # Dataset
    "dataset_name": "SuyashAjitChavan2002/Calvin-Dataset-Final",
    "dataset_config_name": None,
    "edited_image_column": "edited_image",
    "original_image_column": "original_image",
    "edit_instruction_column": "edit_instruction",
    "initial_edit_region_mask_column": "initial_edit_region_mask",
    "final_edit_region_mask_column": "final_edit_region_mask",
    # Output paths (on Modal volume)
    "output_dir": "/data/models/kontext-i2i",
    "logging_dir": "/data/logs/flux-kontext",
    "cache_dir": "/data/datasets/.cache",
    # Training hyperparameters
    "resolution": 256,
    "train_batch_size": 160,
    "gradient_accumulation_steps": 1,
    "num_train_epochs": 75,
    # max_train_steps is NOT set — epochs take precedence
    "learning_rate": 1e-4,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 200,
    "optimizer": "adamw",
    # LoRA configuration
    "rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    # Memory optimization
    "gradient_checkpointing": True,
    "mixed_precision": "bf16",
    "use_8bit_adam": True,
    # Checkpointing (epoch-based)
    "checkpointing_epochs": 1,   # Save checkpoint after every 1 epoch
    "checkpoints_total_limit": 75,
    "validation_epochs": 1,      # Validate after every epoch
    "resume_from_checkpoint": None,  # Set to "latest" to auto-resume from last checkpoint
    # Other settings
    "seed": 0,
    "max_sequence_length": 512,
    "guidance_scale": 1.0,
    "weighting_scheme": "none",
    "dataloader_num_workers": 4,
    "train_text_encoder": False,
    # Static loss hyperparameters
    "t_rec_max": 0.35,
    "p_rec_fraction": 1.0,
    "lambda_static": 7.0,
    # Validation/Testing configuration
    "max_val_samples": 15,    # 15 samples for validation metrics inference (10 visualized)
    "max_test_samples": None, # None = all test samples for metrics inference (10 visualized)
    "eval_viz_num": 10,       # Number of samples to visualize during val/test
    "val_batch_size": 150,    # Batch size for validation/testing loss computation
}

# ============================================================================
# Modal App Setup
# ============================================================================
app = modal.App("flux-kontext-dev-calvin-image-editing")

# Create a volume for persistent storage
VOLUME_NAME = "flux-kontext-dev-calvin-image-editing-volume"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# CUDA Configuration
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# ============================================================================
# Define the Modal image with all required dependencies
# ============================================================================
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    # Install PyTorch first
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision",
        "torchaudio",
    )
    # Install main dependencies
    .uv_pip_install(
        "accelerate",
        "diffusers",
        "transformers",
        "peft",
        "datasets",
        "pillow",
        "numpy",
        "scipy",
        "huggingface-hub",
        "hf_transfer",
        "wandb",
        "bitsandbytes",
        "prodigyopt",
        "scikit-image",
        "lpips",
        "tqdm",
        "packaging",
        "wheel",
        "sentencepiece",
    )
    # Install flash-attn (needs torch present)
    .run_commands("pip install flash-attn --no-build-isolation")
    # Set environment variables
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": "/data/datasets/.cache",
        }
    )
    # Add the flux_finetuning directory
    .add_local_dir(
        ".",  # Current directory (flux_finetuning)
        remote_path="/root/flux_finetuning",
        ignore=[
            "*.git*",
            "__pycache__",
            "*.pyc",
            "*.pyo",
            ".env",
            ".venv",
            "*.log",
            ".DS_Store",
            "*.egg-info",
            ".pytest_cache",
            "models",
            "models/**",
            "*.safetensors",
            "*.bin",
            "*.pkl",
        ],
    )
)

# HuggingFace and WandB secrets
huggingface_secret = modal.Secret.from_name(
    "huggingface-secret"
)  # Should contain HF_TOKEN

# ============================================================================
# Volume-relative path for models (mirrors output_dir structure)
# ============================================================================
# TRAINING_CONFIG["output_dir"] = "/data/models/kontext-i2i"
# So models dir on the volume is "models" (relative to /data mount)
VOLUME_MODELS_REMOTE_DIR = "models"  # Relative to volume root (/data/)
LOCAL_MODELS_DIR = "models"  # Relative to CWD (flux_finetuning/)

# Signal log for polling-based download (written by remote, read by local)
SIGNAL_LOG_PATH = "/data/.training_signals.jsonl"  # Absolute path on volume mount
SIGNAL_LOG_VOLUME_PATH = ".training_signals.jsonl"  # Volume-relative path

# Polling configuration
POLL_INTERVAL_SECONDS = 60          # How often to check for new signals (seconds)
MAX_CONSECUTIVE_POLL_FAILURES = 10  # Max consecutive poll errors before giving up
STALE_POLL_THRESHOLD = 30           # Stale polls before checking if function is alive


# ============================================================================
# GPU Monitoring Utility Function
# ============================================================================
def start_gpu_monitoring(interval: int = 60):
    """
    Start GPU utilization monitoring in a background thread.

    Args:
        interval: Seconds between GPU utilization logs

    Returns:
        tuple: (stop_event, monitor_thread) for controlling the monitoring
    """
    import subprocess
    import threading
    import time

    stop_monitoring = threading.Event()

    def monitor_gpu_utilization():
        """Monitor and log GPU utilization at regular intervals"""
        while not stop_monitoring.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] GPU Utilization:")

                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 8:
                        (
                            gpu_idx,
                            name,
                            gpu_util,
                            mem_util,
                            mem_used,
                            mem_total,
                            temp,
                            power,
                        ) = parts
                        print(
                            f"  GPU{gpu_idx} ({name}): "
                            f"GPU={gpu_util}% | Mem={mem_util}% ({mem_used}/{mem_total}MB) | "
                            f"Temp={temp}°C | Power={power}W"
                        )

            except Exception as e:
                print(f"Error monitoring GPU: {e}")

            stop_monitoring.wait(interval)

    monitor_thread = threading.Thread(target=monitor_gpu_utilization, daemon=True)
    monitor_thread.start()

    return stop_monitoring, monitor_thread


def build_training_command(config: dict, num_gpus: int) -> list:
    """
    Build the CLI command for running the training script.

    Args:
        config: Training configuration dictionary
        num_gpus: Number of GPUs to use

    Returns:
        List of command arguments
    """
    # Build command based on number of GPUs
    if num_gpus > 1:
        # Multi-GPU: Use accelerate launch
        cmd = [
            "accelerate",
            "launch",
            "--multi_gpu",
            f"--num_processes={num_gpus}",
            "--num_machines=1",
            f"--mixed_precision={config.get('mixed_precision', 'bf16')}",
            "--dynamo_backend=no",
            "/root/flux_finetuning/flux_kontext_lora_finetuning.py",
        ]
    else:
        # Single GPU: Use direct python call (like nohup command)
        cmd = [
            "python",
            "-u",  # Unbuffered output for real-time signal detection
            "/root/flux_finetuning/flux_kontext_lora_finetuning.py",
        ]

    # Add training arguments
    arg_mapping = {
        "pretrained_model_name_or_path": "--pretrained_model_name_or_path",
        "revision": "--revision",
        "variant": "--variant",
        "dataset_name": "--dataset_name",
        "dataset_config_name": "--dataset_config_name",
        "edited_image_column": "--edited_image_column",
        "original_image_column": "--original_image_column",
        "edit_instruction_column": "--edit_instruction_column",
        "initial_edit_region_mask_column": "--initial_edit_region_mask_column",
        "final_edit_region_mask_column": "--final_edit_region_mask_column",
        "output_dir": "--output_dir",
        "logging_dir": "--logging_dir",
        "cache_dir": "--cache_dir",
        "resolution": "--resolution",
        "train_batch_size": "--train_batch_size",
        "gradient_accumulation_steps": "--gradient_accumulation_steps",
        "num_train_epochs": "--num_train_epochs",
        "learning_rate": "--learning_rate",
        "lr_scheduler": "--lr_scheduler",
        "lr_warmup_steps": "--lr_warmup_steps",
        "optimizer": "--optimizer",
        "rank": "--rank",
        "lora_alpha": "--lora_alpha",
        "lora_dropout": "--lora_dropout",
        "checkpointing_epochs": "--checkpointing_epochs",
        "checkpoints_total_limit": "--checkpoints_total_limit",
        "validation_epochs": "--validation_epochs",
        "seed": "--seed",
        "max_sequence_length": "--max_sequence_length",
        "guidance_scale": "--guidance_scale",
        "weighting_scheme": "--weighting_scheme",
        "dataloader_num_workers": "--dataloader_num_workers",
        "max_val_samples": "--max_val_samples",
        "max_test_samples": "--max_test_samples",
        "eval_viz_num": "--eval_viz_num",
        "val_batch_size": "--val_batch_size",
        "mixed_precision": "--mixed_precision",
    }

    for key, arg in arg_mapping.items():
        value = config.get(key)
        if value is not None:
            cmd.extend([arg, str(value)])

    # Boolean flags
    if config.get("gradient_checkpointing"):
        cmd.append("--gradient_checkpointing")
    if config.get("cache_latents"):
        cmd.append("--cache_latents")
    if config.get("use_8bit_adam"):
        cmd.append("--use_8bit_adam")
    if config.get("train_text_encoder"):
        cmd.append("--train_text_encoder")

    # Resume from checkpoint if specified
    resume_from = config.get("resume_from_checkpoint")
    if resume_from is not None:
        cmd.extend(["--resume_from_checkpoint", str(resume_from)])

    # Always enable TF32 for performance on Ampere+ GPUs
    cmd.append("--allow_tf32")

    # Add report_to for logging (defaults to tensorboard, can be wandb)
    report_to = config.get("report_to", "tensorboard")
    cmd.extend(["--report_to", report_to])

    return cmd


# ============================================================================
# Helper: Find latest checkpoint in a directory
# ============================================================================
def find_latest_checkpoint(output_dir: str) -> str | None:
    """
    Find the latest checkpoint directory in output_dir.
    Supports epoch-based naming (checkpoint-epoch-N) and legacy step-based (checkpoint-N).
    Returns the checkpoint directory name (e.g., 'checkpoint-epoch-8') or None.
    """
    import os
    if not os.path.exists(output_dir):
        return None
    dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint") and os.path.isdir(os.path.join(output_dir, d))]
    if not dirs:
        return None
    def _sort_key(name):
        parts = name.split("-")
        for p in reversed(parts):
            if p.isdigit():
                return int(p)
        return 0
    dirs = sorted(dirs, key=_sort_key)
    return dirs[-1]


# ============================================================================
# Helper: Upload local checkpoints to Modal volume
# ============================================================================
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600,
)
def upload_checkpoints_to_volume(checkpoint_data: list[dict]):
    """
    Upload checkpoint files to the Modal volume.
    checkpoint_data is a list of dicts: [{"remote_path": str, "content": bytes}, ...]
    """
    import os
    for item in checkpoint_data:
        remote_path = item["remote_path"]
        content = item["content"]
        os.makedirs(os.path.dirname(remote_path), exist_ok=True)
        with open(remote_path, "wb") as f:
            f.write(content)
        print(f"  Uploaded: {remote_path} ({len(content)} bytes)")
    volume.commit()
    print(f"Volume committed with {len(checkpoint_data)} files.")
    return {"status": "success", "files_uploaded": len(checkpoint_data)}


# ============================================================================
# Helper: Check what checkpoints exist on the volume
# ============================================================================
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=300,
)
def get_volume_checkpoint_info():
    """
    Check the Modal volume for existing checkpoints and return info.
    Returns dict with latest checkpoint info.
    """
    import os
    output_dir = TRAINING_CONFIG["output_dir"]
    latest = find_latest_checkpoint(output_dir)
    
    all_checkpoints = []
    if os.path.exists(output_dir):
        all_checkpoints = sorted([
            d for d in os.listdir(output_dir)
            if d.startswith("checkpoint") and os.path.isdir(os.path.join(output_dir, d))
        ])
    
    return {
        "latest_checkpoint": latest,
        "all_checkpoints": all_checkpoints,
        "output_dir": output_dir,
    }


# ============================================================================
# Dataset Download Function (Optional - run once to cache dataset)
# ============================================================================
@app.function(
    image=image,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=3600,  # 1 hour timeout
)
def download_dataset():
    """Download and cache the training dataset"""
    import os

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    from datasets import load_dataset

    print(f"Downloading dataset: {TRAINING_CONFIG['dataset_name']}")

    dataset = load_dataset(
        TRAINING_CONFIG["dataset_name"],
        TRAINING_CONFIG.get("dataset_config_name"),
        cache_dir=TRAINING_CONFIG["cache_dir"],
    )

    print("Dataset downloaded successfully!")
    print(f"  Train size: {len(dataset.get('train', []))}")
    print(f"  Validation size: {len(dataset.get('validation', []))}")
    print(f"  Test size: {len(dataset.get('test', []))}")

    # Commit the volume to persist the downloaded dataset
    volume.commit()

    return {
        "status": "success",
        "dataset_name": TRAINING_CONFIG["dataset_name"],
        "message": "Dataset downloaded and cached successfully",
    }


# ============================================================================
# Helper: Append a signal event to the JSONL log on the volume
# Called from within remote training functions (runs on the remote container).
# ============================================================================
def _append_signal_to_volume(signal_dict: dict):
    """Append a signal event to the JSONL signal log on the volume mount.
    The caller is responsible for calling volume.commit() afterward to
    make the signal visible to local polling.
    """
    import json
    from datetime import datetime, timezone

    signal_dict = dict(signal_dict)  # Don't mutate caller's dict
    signal_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(SIGNAL_LOG_PATH, "a") as f:
        f.write(json.dumps(signal_dict) + "\n")


# ============================================================================
# Multi-GPU Training Function (commits signals to volume for local polling)
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=86400,  # 24 hour timeout for long training runs
    memory=65536,  # 64 GB RAM
)
def train_flux_kontext():
    """Train Flux Kontext LoRA with multi-GPU support using accelerate launch.
    Writes signal events to a JSONL log on the volume so the local entrypoint
    can poll for new checkpoints, validation, and test completions.
    """
    import os
    import subprocess
    import sys

    # Set environment variables
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    # Create output directories
    os.makedirs(TRAINING_CONFIG["output_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["logging_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["cache_dir"], exist_ok=True)

    # Clear signal log for this training run and write initial signal
    with open(SIGNAL_LOG_PATH, "w") as f:
        pass
    _append_signal_to_volume({"signal": "training_started", "detail": f"{NUM_GPUS}x {GPU_TYPE}"})
    volume.commit()

    print(f"Starting Flux Kontext LoRA training with {NUM_GPUS} {GPU_TYPE} GPUs...")
    print(f"Output directory: {TRAINING_CONFIG['output_dir']}")

    # ================================================================
    # Auto-detect latest checkpoint for resume (preemption recovery)
    # ================================================================
    latest_ckpt = find_latest_checkpoint(TRAINING_CONFIG["output_dir"])
    config = dict(TRAINING_CONFIG)
    if latest_ckpt:
        print(f"\n>>> Auto-detected checkpoint on volume: {latest_ckpt}")
        print(f">>> Will resume training from this checkpoint.")
        config["resume_from_checkpoint"] = "latest"
    else:
        print("\n>>> No existing checkpoints found on volume. Starting training from scratch.")
        config["resume_from_checkpoint"] = None

    # Start GPU monitoring if enabled
    stop_monitoring = None
    monitor_thread = None

    if ENABLE_GPU_MONITORING:
        stop_monitoring, monitor_thread = start_gpu_monitoring(
            interval=GPU_MONITORING_INTERVAL
        )
        print(
            f"GPU monitoring started (logging every {GPU_MONITORING_INTERVAL} seconds)\n"
        )

    # Build the training command
    cmd = build_training_command(config, NUM_GPUS)

    print(f"Running command:\n{' '.join(cmd)}\n")
    print("=" * 80)

    try:
        # Run training with output streaming to detect checkpoint/validation signals
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        process = subprocess.Popen(
            cmd,
            cwd="/root/flux_finetuning",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            env=env,
        )
        
        for line in process.stdout:
            # Print all output in real-time
            sys.stdout.write(line)
            sys.stdout.flush()
            
            # Detect checkpoint save signal — log + commit volume
            if "###CHECKPOINT_SAVED###" in line:
                print("\n>>> [MODAL] Checkpoint saved detected. Committing volume...")
                _append_signal_to_volume({"signal": "checkpoint_saved", "detail": line.strip()})
                try:
                    volume.commit()
                    print(">>> [MODAL] Volume committed successfully after checkpoint save.")
                except Exception as e:
                    print(f">>> [MODAL] Warning: Volume commit failed: {e}")
            
            # Detect validation complete signal — log + commit volume
            elif "###VALIDATION_COMPLETE###" in line:
                print("\n>>> [MODAL] Validation complete detected. Committing volume...")
                _append_signal_to_volume({"signal": "validation_complete", "detail": line.strip()})
                try:
                    volume.commit()
                    print(">>> [MODAL] Volume committed successfully after validation.")
                except Exception as e:
                    print(f">>> [MODAL] Warning: Volume commit failed: {e}")

            # Detect final test complete signal — log + commit volume
            elif "###TEST_COMPLETE###" in line:
                print("\n>>> [MODAL] Final test complete detected. Committing volume...")
                _append_signal_to_volume({"signal": "test_complete", "detail": line.strip()})
                try:
                    volume.commit()
                    print(">>> [MODAL] Volume committed successfully after final test.")
                except Exception as e:
                    print(f">>> [MODAL] Warning: Volume commit failed: {e}")
        
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
            
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        # Log error signal and commit to preserve partial progress
        _append_signal_to_volume({"signal": "training_error", "detail": f"Exit code {e.returncode}"})
        print(">>> [MODAL] Committing volume to preserve partial progress...")
        try:
            volume.commit()
        except Exception:
            pass
        raise
    finally:
        # Stop GPU monitoring
        if ENABLE_GPU_MONITORING and stop_monitoring is not None:
            print("\nStopping GPU monitoring...")
            stop_monitoring.set()
            if monitor_thread is not None:
                monitor_thread.join(timeout=5)

    # Final commit to persist everything
    print("\nCommitting volume to persist trained model...")
    result = {
        "status": "success",
        "output_dir": TRAINING_CONFIG["output_dir"],
        "num_gpus": NUM_GPUS,
        "gpu_type": GPU_TYPE,
        "resumed_from": latest_ckpt,
        "message": f"Flux Kontext LoRA training completed on {NUM_GPUS} {GPU_TYPE} GPUs",
    }
    _append_signal_to_volume({"signal": "training_complete", "detail": result})
    volume.commit()

    return result


# ============================================================================
# Single-GPU Training Function (commits signals to volume for local polling)
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=86400,  # 24 hour timeout
    memory=65536,  # 64 GB RAM
)
def train_flux_kontext_single_gpu():
    """Train Flux Kontext LoRA on a single GPU with auto-resume.
    Writes signal events to a JSONL log on the volume so the local entrypoint
    can poll for new checkpoints, validation, and test completions.
    """
    import os
    import subprocess
    import sys

    # Set environment variables
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    # Create output directories
    os.makedirs(TRAINING_CONFIG["output_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["logging_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["cache_dir"], exist_ok=True)

    # Clear signal log for this training run and write initial signal
    with open(SIGNAL_LOG_PATH, "w") as f:
        pass
    _append_signal_to_volume({"signal": "training_started", "detail": "single GPU"})
    volume.commit()

    print("Starting Flux Kontext LoRA training on single GPU...")

    # ================================================================
    # Auto-detect latest checkpoint for resume (preemption recovery)
    # ================================================================
    latest_ckpt = find_latest_checkpoint(TRAINING_CONFIG["output_dir"])
    config = dict(TRAINING_CONFIG)
    if latest_ckpt:
        print(f"\n>>> Auto-detected checkpoint on volume: {latest_ckpt}")
        print(f">>> Will resume training from this checkpoint.")
        config["resume_from_checkpoint"] = "latest"
    else:
        print("\n>>> No existing checkpoints found on volume. Starting training from scratch.")
        config["resume_from_checkpoint"] = None

    # Build command for single GPU
    cmd = build_training_command(config, 1)

    print(f"Running command:\n{' '.join(cmd)}\n")

    try:
        # Run training with output streaming to detect checkpoint/validation signals
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        process = subprocess.Popen(
            cmd,
            cwd="/root/flux_finetuning",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            env=env,
        )

        for line in process.stdout:
            # Print all output in real-time
            sys.stdout.write(line)
            sys.stdout.flush()

            # Detect checkpoint save signal — log + commit volume
            if "###CHECKPOINT_SAVED###" in line:
                print("\n>>> [MODAL] Checkpoint saved detected. Committing volume...")
                _append_signal_to_volume({"signal": "checkpoint_saved", "detail": line.strip()})
                try:
                    volume.commit()
                    print(">>> [MODAL] Volume committed successfully after checkpoint save.")
                except Exception as e:
                    print(f">>> [MODAL] Warning: Volume commit failed: {e}")

            # Detect validation complete signal — log + commit volume
            elif "###VALIDATION_COMPLETE###" in line:
                print("\n>>> [MODAL] Validation complete detected. Committing volume...")
                _append_signal_to_volume({"signal": "validation_complete", "detail": line.strip()})
                try:
                    volume.commit()
                    print(">>> [MODAL] Volume committed successfully after validation.")
                except Exception as e:
                    print(f">>> [MODAL] Warning: Volume commit failed: {e}")

            # Detect final test complete signal — log + commit volume
            elif "###TEST_COMPLETE###" in line:
                print("\n>>> [MODAL] Final test complete detected. Committing volume...")
                _append_signal_to_volume({"signal": "test_complete", "detail": line.strip()})
                try:
                    volume.commit()
                    print(">>> [MODAL] Volume committed successfully after final test.")
                except Exception as e:
                    print(f">>> [MODAL] Warning: Volume commit failed: {e}")

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        # Log error signal and commit to preserve partial progress
        _append_signal_to_volume({"signal": "training_error", "detail": f"Exit code {e.returncode}"})
        print(">>> [MODAL] Committing volume to preserve partial progress...")
        try:
            volume.commit()
        except Exception:
            pass
        raise

    # Final commit
    result = {
        "status": "success",
        "output_dir": TRAINING_CONFIG["output_dir"],
        "resumed_from": latest_ckpt,
        "message": "Flux Kontext LoRA training completed on single GPU",
    }
    _append_signal_to_volume({"signal": "training_complete", "detail": result})
    volume.commit()

    return result


# ============================================================================
# Inference Function (test the trained model)
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=3600,  # 1 hour timeout
)
def run_inference(
    lora_path: str = None,
    prompt: str = "Move the cat to the right side of the image",
    input_image_url: str = None,
):
    """
    Run inference with the trained Flux Kontext LoRA model.

    Args:
        lora_path: Path to the LoRA weights (defaults to training output_dir)
        prompt: Edit instruction prompt
        input_image_url: URL to the input image
    """
    import os
    import sys

    sys.path.insert(0, "/root/flux_finetuning")

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

    import torch
    from diffusers.utils import load_image
    from pipeline_flux_kontext_multi_imgs import FluxKontextPipeline

    lora_path = lora_path or TRAINING_CONFIG["output_dir"]

    print("Loading Flux Kontext pipeline...")
    pipeline = FluxKontextPipeline.from_pretrained(
        TRAINING_CONFIG["pretrained_model_name_or_path"],
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    # Load LoRA weights
    print(f"Loading LoRA weights from {lora_path}...")
    pipeline.load_lora_weights(lora_path)

    # Load input image
    if input_image_url:
        print(f"Loading input image from {input_image_url}...")
        input_image = load_image(input_image_url)
    else:
        print("No input image provided, creating a test image...")
        from PIL import Image

        input_image = Image.new("RGB", (512, 512), color="white")

    # Run inference
    print(f"Running inference with prompt: {prompt}")
    output = pipeline(
        prompt=prompt,
        image=input_image,
        guidance_scale=TRAINING_CONFIG.get("guidance_scale", 3.5),
        num_inference_steps=28,
    )

    # Save output
    output_path = "/data/inference_output.png"
    output.images[0].save(output_path)
    print(f"Output saved to {output_path}")

    volume.commit()

    return {
        "status": "success",
        "output_path": output_path,
        "prompt": prompt,
    }


# ============================================================================
# Resume Training Function
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=86400,
)
def resume_training(checkpoint_path: str = "latest"):
    """
    Resume training from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint or "latest" to auto-detect
    """
    import os
    import subprocess

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    # Create output directories
    os.makedirs(TRAINING_CONFIG["output_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["logging_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["cache_dir"], exist_ok=True)

    print(f"Resuming Flux Kontext LoRA training from checkpoint: {checkpoint_path}")

    # Start GPU monitoring
    stop_monitoring = None
    if ENABLE_GPU_MONITORING:
        stop_monitoring, _ = start_gpu_monitoring(interval=GPU_MONITORING_INTERVAL)

    # Build command with resume flag (override any config-level resume_from_checkpoint)
    resume_config = dict(TRAINING_CONFIG)
    resume_config["resume_from_checkpoint"] = checkpoint_path
    cmd = build_training_command(resume_config, NUM_GPUS)

    print(f"Running command:\n{' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True, cwd="/root/flux_finetuning")
    finally:
        if stop_monitoring:
            stop_monitoring.set()

    volume.commit()

    return {
        "status": "success",
        "output_dir": TRAINING_CONFIG["output_dir"],
        "message": f"Training resumed from {checkpoint_path}",
    }


# ============================================================================
# List Checkpoints Function
# ============================================================================
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=300,
)
def list_checkpoints():
    """List all available checkpoints in the output directory"""
    import os

    output_dir = TRAINING_CONFIG["output_dir"]

    if not os.path.exists(output_dir):
        return {
            "status": "no_checkpoints",
            "message": f"Output directory {output_dir} does not exist",
        }

    checkpoints = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint"):
            checkpoints.append(item)

    checkpoints = sorted(
        checkpoints, key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0
    )

    # Also check for final LoRA weights
    has_final_weights = os.path.exists(
        os.path.join(output_dir, "pytorch_lora_weights.safetensors")
    )

    return {
        "status": "success",
        "output_dir": output_dir,
        "checkpoints": checkpoints,
        "has_final_weights": has_final_weights,
        "total_checkpoints": len(checkpoints),
    }


# ============================================================================
# Local Helper: Upload local checkpoints to Modal volume
# ============================================================================
def _upload_local_checkpoints_to_volume():
    """
    Check if local 'models/' directory contains checkpoints.
    If so, upload them to the Modal volume preserving directory structure.
    The local models/ dir maps to /data/models/ on the volume.
    """
    import os

    local_models_path = os.path.join(os.path.dirname(__file__), LOCAL_MODELS_DIR)
    if not os.path.exists(local_models_path):
        print(f"  No local '{LOCAL_MODELS_DIR}/' directory found. Skipping upload.")
        return False

    # Collect all files to upload
    files_to_upload = []
    for root, dirs, files in os.walk(local_models_path):
        for fname in files:
            local_path = os.path.join(root, fname)
            # Compute remote path: /data/models/... (maps local models/X to /data/models/X)
            rel_path = os.path.relpath(local_path, local_models_path)
            remote_path = f"/data/{VOLUME_MODELS_REMOTE_DIR}/{rel_path}"
            files_to_upload.append((local_path, remote_path))

    if not files_to_upload:
        print(f"  Local '{LOCAL_MODELS_DIR}/' directory is empty. Skipping upload.")
        return False

    # Check what already exists on the volume
    print("  Checking existing checkpoints on Modal volume...")
    vol_info = get_volume_checkpoint_info.remote()
    existing_checkpoints = vol_info.get("all_checkpoints", [])

    # Determine which local checkpoints are new
    local_checkpoints = set()
    for _, remote_path in files_to_upload:
        # Extract checkpoint dir name from path like /data/models/kontext-i2i/checkpoint-epoch-0/file
        parts = remote_path.split("/")
        for i, p in enumerate(parts):
            if p.startswith("checkpoint"):
                local_checkpoints.add(p)
                break

    new_checkpoints = local_checkpoints - set(existing_checkpoints)
    if not new_checkpoints and existing_checkpoints:
        print(f"  All local checkpoints already exist on volume: {sorted(existing_checkpoints)}")
        print(f"  Skipping upload.")
        return False

    print(f"  Found {len(files_to_upload)} files to upload from local '{LOCAL_MODELS_DIR}/'")
    if new_checkpoints:
        print(f"  New checkpoints to upload: {sorted(new_checkpoints)}")
    print(f"  Existing checkpoints on volume: {sorted(existing_checkpoints) if existing_checkpoints else 'none'}")

    # Upload in batches (to avoid huge memory usage for large files)
    BATCH_SIZE = 10  # Files per batch
    total_uploaded = 0
    for i in range(0, len(files_to_upload), BATCH_SIZE):
        batch = files_to_upload[i : i + BATCH_SIZE]
        batch_data = []
        for local_path, remote_path in batch:
            with open(local_path, "rb") as f:
                content = f.read()
            batch_data.append({"remote_path": remote_path, "content": content})
            total_uploaded += 1

        print(f"  Uploading batch {i // BATCH_SIZE + 1} ({len(batch_data)} files)...")
        upload_checkpoints_to_volume.remote(batch_data)

    print(f"  Successfully uploaded {total_uploaded} files to Modal volume.")
    return True


# ============================================================================
# Local Helper: Download models from Modal volume (incremental)
# ============================================================================
# Module-level set that tracks which checkpoint dirs have already been
# downloaded in this session.  Persists across calls within a single
# `modal run` invocation so we never re-download the same checkpoint twice.
_downloaded_checkpoints: set[str] = set()
_download_tracking_initialized = False


def _download_models_from_volume():
    """
    Incrementally download the 'models' directory from Modal volume to the
    local machine.  Checkpoint directories (`checkpoint-epoch-N`) that already
    exist locally are skipped.  Non-checkpoint content (e.g.,
    `validation_visualization/`) is always re-synced since it gets updated
    every epoch.

    Uses the Modal Python SDK directly to avoid CLI profile/workspace
    mismatches.
    """
    import os
    import re
    from modal.volume import FileEntryType

    global _downloaded_checkpoints, _download_tracking_initialized

    local_dir = os.path.dirname(__file__)
    local_models_path = os.path.join(local_dir, LOCAL_MODELS_DIR)

    print(f"\n>>> Downloading models from Modal volume to {local_models_path}...")

    # ------------------------------------------------------------------
    # One-time: seed the tracking set from checkpoint dirs already on disk
    # (handles resume after crash / restart of the local process)
    # ------------------------------------------------------------------
    if not _download_tracking_initialized:
        if os.path.isdir(local_models_path):
            for root, dirs, _files in os.walk(local_models_path):
                for d in dirs:
                    if re.match(r"checkpoint-epoch-\d+$", d) or re.match(r"checkpoint-\d+$", d):
                        _downloaded_checkpoints.add(d)
                break  # only need top-level + one level down; walk gives us dirs here
            # Also walk one level deeper (e.g., models/kontext-i2i/)
            for sub in os.listdir(local_models_path):
                sub_path = os.path.join(local_models_path, sub)
                if os.path.isdir(sub_path):
                    for d in os.listdir(sub_path):
                        if re.match(r"checkpoint-epoch-\d+$", d) or re.match(r"checkpoint-\d+$", d):
                            _downloaded_checkpoints.add(d)
        if _downloaded_checkpoints:
            print(f">>> Seeded download tracker with {len(_downloaded_checkpoints)} existing local checkpoints.")
        _download_tracking_initialized = True

    try:
        # Ensure the volume reference is hydrated (resolved) for API access.
        if not volume.is_hydrated:
            volume.hydrate()

        # List all files in the models directory recursively
        try:
            entries = volume.listdir(VOLUME_MODELS_REMOTE_DIR, recursive=True)
        except Exception as e:
            print(f">>> Warning: Could not list volume directory '{VOLUME_MODELS_REMOTE_DIR}': {e}")
            return

        file_entries = [e for e in entries if e.type == FileEntryType.FILE]

        if not file_entries:
            print(">>> Warning: No files found in models directory on volume.")
            return

        # ------------------------------------------------------------------
        # Partition files: skip files belonging to already-downloaded
        # checkpoint dirs; always re-download non-checkpoint files.
        # ------------------------------------------------------------------
        checkpoint_re = re.compile(r"(?:checkpoint-epoch-\d+|checkpoint-\d+)")
        files_to_download = []
        skipped = 0
        new_checkpoint_dirs: set[str] = set()

        for entry in file_entries:
            # entry.path e.g. "models/kontext-i2i/checkpoint-epoch-9/pytorch_lora_weights.safetensors"
            parts = entry.path.split("/")
            ckpt_dir = None
            for p in parts:
                if checkpoint_re.fullmatch(p):
                    ckpt_dir = p
                    break

            if ckpt_dir and ckpt_dir in _downloaded_checkpoints:
                skipped += 1
                continue

            files_to_download.append(entry)
            if ckpt_dir:
                new_checkpoint_dirs.add(ckpt_dir)

        if skipped:
            print(f">>> Skipping {skipped} files from {len(_downloaded_checkpoints)} already-downloaded checkpoints.")

        if not files_to_download:
            print(">>> Everything is up to date — nothing new to download.")
            return

        print(f">>> Downloading {len(files_to_download)} files ({len(new_checkpoint_dirs)} new checkpoints)...")

        file_count = 0
        total_bytes = 0
        for entry in files_to_download:
            local_file_path = os.path.join(local_dir, entry.path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            with open(local_file_path, "wb") as f:
                bytes_written = volume.read_file_into_fileobj(entry.path, f)
                total_bytes += bytes_written

            file_count += 1

        # Update the tracking set with newly downloaded checkpoints
        _downloaded_checkpoints.update(new_checkpoint_dirs)

        total_mb = total_bytes / (1024 * 1024)
        print(f">>> Successfully downloaded {file_count} files ({total_mb:.1f} MB) from volume.")

    except Exception as e:
        print(f">>> Warning: Download failed: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# Local Helper: Read signal log from volume (for polling-based downloads)
# ============================================================================
def _read_signal_log_from_volume() -> list:
    """Read the signal log JSONL file from the Modal volume.

    Returns a list of signal event dicts.  Each call is a short-lived API call
    that is resilient to network blips — failures are caught by the caller.
    """
    import io
    import json

    # Ensure the volume reference is hydrated
    if not volume.is_hydrated:
        volume.hydrate()

    buf = io.BytesIO()
    try:
        volume.read_file_into_fileobj(SIGNAL_LOG_VOLUME_PATH, buf)
    except Exception as e:
        # File may not exist yet (training hasn't started writing).
        # Log the reason so we can debug signal-read issues.
        import sys
        print(f">>> [LOCAL/signal_read] Could not read signal log: {type(e).__name__}: {e}", flush=True, file=sys.stderr)
        return []

    buf.seek(0)
    signals = []
    for line in buf.read().decode("utf-8").strip().split("\n"):
        line = line.strip()
        if line:
            try:
                signals.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return signals


# ============================================================================
# Local Entrypoint
# ============================================================================
@app.local_entrypoint()
def main(
    action: str = "train",
    checkpoint: str = "latest",
    prompt: str = "Move the object to the right",
    image_url: str = None,
    single_gpu: bool = True,
):
    """
    Local entrypoint to run training, inference, or utilities.

    Args:
        action: One of "train", "resume", "inference", "download", "list_checkpoints"
        checkpoint: Checkpoint path for resume (default: "latest")
        prompt: Edit prompt for inference
        image_url: Input image URL for inference
        single_gpu: Use single GPU instead of multi-GPU
    """
    print("=" * 60)
    print(f"Flux Kontext LoRA Finetuning - Action: {action}")
    print("=" * 60)

    if action == "train":
        import sys
        import time

        def _fprint(msg: str):
            """Print with immediate flush so messages appear in log files."""
            print(msg, flush=True)

        # =============================================================
        # Step 1: Upload local checkpoints to volume (if any exist)
        # =============================================================
        _fprint("\n[Step 1/3] Checking for local checkpoints to upload to Modal volume...")
        uploaded = _upload_local_checkpoints_to_volume()
        if uploaded:
            _fprint("  Local checkpoints uploaded successfully.")
        else:
            _fprint("  No new local checkpoints to upload.")

        # =============================================================
        # Step 2: Spawn training (non-blocking)
        # =============================================================
        _fprint(f"\n[Step 2/3] Spawning training function...")
        if single_gpu:
            _fprint("  Mode: single-GPU training")
            fc = train_flux_kontext_single_gpu.spawn()
        else:
            _fprint(f"  Mode: multi-GPU training on {NUM_GPUS} {GPU_TYPE} GPUs")
            fc = train_flux_kontext.spawn()

        # =============================================================
        # Step 3: Poll volume for signals and download new checkpoints
        # Each poll is an independent short-lived API call — no long-
        # lived gRPC stream that can die after hours.
        # =============================================================
        _fprint(f"\n[Step 3/3] Polling volume every {POLL_INTERVAL_SECONDS}s for new checkpoints...")
        processed_signal_count = 0
        download_count = 0
        consecutive_poll_failures = 0
        stale_polls = 0
        poll_iteration = 0
        training_done = False
        result = {"status": "unknown"}

        while not training_done:
            time.sleep(POLL_INTERVAL_SECONDS)
            poll_iteration += 1

            try:
                signals = _read_signal_log_from_volume()
                consecutive_poll_failures = 0  # Reset on success

                new_signals = signals[processed_signal_count:]

                # Heartbeat every 5 polls (~5 min) so we know the loop is alive
                if poll_iteration % 5 == 0:
                    _fprint(
                        f">>> [LOCAL] Heartbeat: poll #{poll_iteration}, "
                        f"total_signals={len(signals)}, processed={processed_signal_count}, "
                        f"stale={stale_polls}, downloads={download_count}"
                    )

                if not new_signals:
                    stale_polls += 1
                    # After many stale polls, check if function is still alive
                    if stale_polls >= STALE_POLL_THRESHOLD:
                        _fprint(f"\n>>> [LOCAL] No new signals for {stale_polls * POLL_INTERVAL_SECONDS}s. Checking remote...")
                        try:
                            result = fc.get(timeout=0)
                            _fprint(">>> [LOCAL] Training function has completed (missed final signal).")
                            # Do a final download in case we missed signal-driven downloads
                            download_count += 1
                            _fprint(f"\n>>> [LOCAL] Final download (#{download_count})...")
                            _download_models_from_volume()
                            training_done = True
                        except TimeoutError:
                            _fprint(">>> [LOCAL] Training function still running. Continuing to poll...")
                            stale_polls = 0
                        except Exception as e:
                            # Could be a different timeout exception from Modal SDK
                            err_name = type(e).__name__
                            if "timeout" in err_name.lower() or "timeout" in str(e).lower():
                                _fprint(f">>> [LOCAL] Training function still running ({err_name}). Continuing to poll...")
                                stale_polls = 0
                            else:
                                _fprint(f">>> [LOCAL] Training function may have crashed: {err_name}: {e}")
                                download_count += 1
                                _fprint(f"\n>>> [LOCAL] Downloading partial results (#{download_count})...")
                                try:
                                    _download_models_from_volume()
                                except Exception:
                                    pass
                                training_done = True
                    continue

                stale_polls = 0

                for event in new_signals:
                    signal = event.get("signal", "unknown")
                    ts = event.get("timestamp", "")

                    if signal == "training_started":
                        _fprint(f"\n>>> [LOCAL] [{ts}] Training started on remote.")

                    elif signal == "checkpoint_saved":
                        download_count += 1
                        _fprint(f"\n>>> [LOCAL] [{ts}] Checkpoint saved. Downloading (#{download_count})...")
                        _download_models_from_volume()
                        _fprint(f">>> [LOCAL] Download #{download_count} complete (checkpoint).")

                    elif signal == "validation_complete":
                        download_count += 1
                        _fprint(f"\n>>> [LOCAL] [{ts}] Validation complete. Downloading (#{download_count})...")
                        _download_models_from_volume()
                        _fprint(f">>> [LOCAL] Download #{download_count} complete (validation).")

                    elif signal == "test_complete":
                        download_count += 1
                        _fprint(f"\n>>> [LOCAL] [{ts}] Test complete. Downloading (#{download_count})...")
                        _download_models_from_volume()
                        _fprint(f">>> [LOCAL] Download #{download_count} complete (test).")

                    elif signal == "training_complete":
                        detail = event.get("detail", {})
                        _fprint(f"\n>>> [LOCAL] [{ts}] Training complete!")
                        _fprint(f"  Status: {detail.get('status', 'success')}")
                        _fprint(f"  Message: {detail.get('message', '')}")
                        if detail.get('resumed_from'):
                            _fprint(f"  Resumed from: {detail['resumed_from']}")

                        download_count += 1
                        _fprint(f"\n>>> [LOCAL] Final download (#{download_count})...")
                        _download_models_from_volume()
                        _fprint(f">>> [LOCAL] Final download complete.")
                        training_done = True

                    elif signal == "training_error":
                        _fprint(f"\n>>> [LOCAL] [{ts}] Training error: {event.get('detail', 'unknown')}")
                        download_count += 1
                        _fprint(f"\n>>> [LOCAL] Downloading partial results (#{download_count})...")
                        _download_models_from_volume()
                        training_done = True

                    else:
                        _fprint(f"\n>>> [LOCAL] [{ts}] Unknown signal: {signal}")

                processed_signal_count = len(signals)

            except Exception as e:
                consecutive_poll_failures += 1
                _fprint(f"\n>>> [LOCAL] Poll failed ({consecutive_poll_failures}/{MAX_CONSECUTIVE_POLL_FAILURES}): {type(e).__name__}: {e}")
                if consecutive_poll_failures >= MAX_CONSECUTIVE_POLL_FAILURES:
                    _fprint(">>> [LOCAL] Too many consecutive poll failures. Attempting final download...")
                    try:
                        _download_models_from_volume()
                    except Exception as dl_err:
                        _fprint(f">>> [LOCAL] Final download also failed: {dl_err}")
                    training_done = True

        # Try to collect the function result (may surface exceptions from remote)
        if result.get("status") == "unknown":
            try:
                result = fc.get(timeout=30)
                _fprint(f"\n>>> [LOCAL] Training function returned: {result.get('status', 'unknown')}")
            except TimeoutError:
                _fprint("\n>>> [LOCAL] Training function still running remotely (this is normal for long training runs).")
                result = {"status": "ok", "message": "Polling ended; remote function still running."}
            except Exception as e:
                err_name = type(e).__name__
                if "timeout" in err_name.lower() or "timeout" in str(e).lower():
                    _fprint(f"\n>>> [LOCAL] Training function still running remotely ({err_name}).")
                    result = {"status": "ok", "message": "Polling ended; remote function still running."}
                else:
                    _fprint(f"\n>>> [LOCAL] Error getting training result: {err_name}: {e}")
                    result = {"status": "error", "message": str(e)}

        _fprint(f"\n>>> [LOCAL] Total downloads performed: {download_count}")

    elif action == "resume":
        print(f"Resuming training from checkpoint: {checkpoint}")
        result = resume_training.remote(checkpoint_path=checkpoint)
        print("\nResume Result:")
        print(f"  Status: {result['status']}")
        print(f"  Message: {result['message']}")

        # Download models after resume completes
        print("\nDownloading models from Modal volume...")
        _download_models_from_volume()

    elif action == "inference":
        print("Running inference...")
        result = run_inference.remote(prompt=prompt, input_image_url=image_url)
        print("\nInference Result:")
        print(f"  Status: {result['status']}")
        print(f"  Output: {result['output_path']}")

    elif action == "download":
        print("Downloading and caching dataset...")
        result = download_dataset.remote()
        print("\nDownload Result:")
        print(f"  Status: {result['status']}")
        print(f"  Message: {result['message']}")

    elif action == "list_checkpoints":
        print("Listing checkpoints...")
        result = list_checkpoints.remote()
        print("\nCheckpoints:")
        print(f"  Output dir: {result['output_dir']}")
        print(f"  Total: {result.get('total_checkpoints', 0)}")
        for cp in result.get("checkpoints", []):
            print(f"    - {cp}")
        print(f"  Final weights saved: {result.get('has_final_weights', False)}")

    elif action == "download_models":
        print("Downloading models from Modal volume...")
        _download_models_from_volume()
        result = {"status": "success", "message": "Models downloaded"}

    else:
        raise ValueError(
            f"Invalid action: {action}. "
            "Use 'train', 'resume', 'inference', 'download', 'download_models', or 'list_checkpoints'"
        )

    print("=" * 60)
    return result
