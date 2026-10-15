"""
Modal script to launch Flux Kontext LoRA finetuning for Edit Region Mask (ERM) prediction.

This script uses Modal's serverless infrastructure to train the Flux Kontext model
for predicting edit region masks across multiple GPUs using accelerate launch.

The ERM task predicts binary masks for regions that will be edited based on an
edit instruction, enabling better control over image editing operations.

Usage:
    # Run training
    modal run modal_flux_kontext_train.py --action train

    # Run inference only (after training)
    modal run modal_flux_kontext_train.py --action inference

    # Download dataset first (optional, for caching)
    modal run modal_flux_kontext_train.py --action download

    # List checkpoints
    modal run modal_flux_kontext_train.py --action list_checkpoints

    # Resume training from checkpoint
    modal run modal_flux_kontext_train.py --action resume --checkpoint latest
"""

import modal
import os
import pathlib

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
# Training Configuration - Matches the provided command line setup
# ============================================================================
TRAINING_CONFIG = {
    # Model
    "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-Kontext-dev",
    "revision": None,
    "variant": None,
    "vae_encode_mode": "mode",
    # Dataset
    "dataset_name": "SuyashAjitChavan2002/LGRM_BridgeDataV2_Teleoperated",
    "dataset_config_name": None,
    # Dataset columns for ERM task
    "edited_image_column": "final_edit_region_mask",  # Target: final edit region mask
    "original_image_column": "original_image",
    "edit_instruction_column": "edit_instruction",
    "initial_edit_region_mask_column": "initial_edit_region_mask",
    # Output paths (on Modal volume)
    "output_dir": "/data/models/kontext-i2i-erm",
    "logging_dir": "/data/logs/flux-kontext-erm",
    "cache_dir": "/data/datasets/.cache",
    # Training hyperparameters
    "resolution": 256,
    "train_batch_size": 160,
    "gradient_accumulation_steps": 1,
    "num_train_epochs": 75,
    # max_train_steps is intentionally omitted (None) — epochs take precedence
    "learning_rate": 1e-4,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 200,
    "optimizer": "adamw",
    # LoRA configuration
    "rank": 16,
    "lora_alpha": 16,  # Typically same as rank
    "lora_dropout": 0.0,
    # Memory optimization
    "gradient_checkpointing": True,
    "mixed_precision": "bf16",
    "use_8bit_adam": True,
    # Checkpointing (epoch-based)
    "checkpointing_epochs": 1,
    "checkpoints_total_limit": 75,
    "validation_epochs": 1,
    # Other settings
    "seed": 0,
    "max_sequence_length": 512,
    "guidance_scale": 1.0,
    "weighting_scheme": "none",
    "dataloader_num_workers": 4,
    "train_text_encoder": False,
    # "report_to": "wandb",
    # Validation/eval settings
    "eval_viz_num": 10,
    "sample_batch_size": 150,  # Batch size for validation/test loss computation
    "validation_samples": 15,  # Samples for validation metrics computation (10 among these for viz)
    "test_samples": -1,  # Run test metrics on all samples (-1 for all)
    "skip_validation_loss": False,  # Compute loss during validation (uses all available samples)
    "skip_test_loss": False,  # Compute loss during testing (uses all available samples)
    # Data augmentation
    "center_crop": False,
    "random_flip": False,
    "repeats": 1,
}

# ============================================================================
# Modal App Setup
# ============================================================================
app = modal.App("flux-kontext-dev-bridge-fer-mask-prediction")

# Create a volume for persistent storage
volume = modal.Volume.from_name("flux-kontext-dev-bridge-fer-mask-prediction-volume", create_if_missing=True)

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
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    # Add the flux_finetuning_erm directory
    .add_local_dir(
        ".",  # Current directory (flux_finetuning_erm)
        remote_path="/root/flux_finetuning_erm",
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
            "models",  # Don't bundle models into image; they go to the volume
        ],
    )
)

# HuggingFace and WandB secrets
huggingface_secret = modal.Secret.from_name(
    "huggingface-secret"
)  # Should contain HF_TOKEN


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
            "/root/flux_finetuning_erm/flux_kontext_lora_finetuning.py",
        ]
    else:
        # Single GPU: Use direct python call
        cmd = [
            "python",
            "/root/flux_finetuning_erm/flux_kontext_lora_finetuning.py",
        ]

    # Add training arguments - map config keys to CLI arguments
    arg_mapping = {
        "pretrained_model_name_or_path": "--pretrained_model_name_or_path",
        "revision": "--revision",
        "variant": "--variant",
        "vae_encode_mode": "--vae_encode_mode",
        "dataset_name": "--dataset_name",
        "dataset_config_name": "--dataset_config_name",
        "edited_image_column": "--edited_image_column",
        "original_image_column": "--original_image_column",
        "edit_instruction_column": "--edit_instruction_column",
        "initial_edit_region_mask_column": "--initial_edit_region_mask_column",
        "output_dir": "--output_dir",
        "logging_dir": "--logging_dir",
        "cache_dir": "--cache_dir",
        "resolution": "--resolution",
        "train_batch_size": "--train_batch_size",
        "sample_batch_size": "--sample_batch_size",
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
        "eval_viz_num": "--eval_viz_num",
        "repeats": "--repeats",
        "mixed_precision": "--mixed_precision",
        "validation_samples": "--validation_samples",
        "test_samples": "--test_samples",
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
    if config.get("center_crop"):
        cmd.append("--center_crop")
    if config.get("random_flip"):
        cmd.append("--random_flip")
    # Handle skip_validation_loss - default in argparse is True, so we need --no_skip_validation_loss to disable
    if config.get("skip_validation_loss", True):
        cmd.append("--skip_validation_loss")
    else:
        cmd.append("--no_skip_validation_loss")
    if config.get("skip_test_loss"):
        cmd.append("--skip_test_loss")

    # Always enable TF32 for performance on Ampere+ GPUs
    cmd.append("--allow_tf32")

    # Add report_to for logging (defaults to tensorboard, can be wandb)
    report_to = config.get("report_to", "tensorboard")
    cmd.extend(["--report_to", report_to])

    return cmd


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
# Upload Local Checkpoints to Volume
# ============================================================================
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600,  # 1 hour timeout for uploads
)
def upload_checkpoints_to_volume(file_map: dict):
    """
    Receive a dict of {remote_path: file_bytes} and write them to the volume.
    This is called from the local entrypoint to seed the volume with local checkpoints.
    """
    import os

    total_files = len(file_map)
    print(f"Uploading {total_files} files to Modal volume...")

    for remote_path, file_bytes in file_map.items():
        os.makedirs(os.path.dirname(remote_path), exist_ok=True)
        with open(remote_path, "wb") as f:
            f.write(file_bytes)
        print(f"  Uploaded: {remote_path} ({len(file_bytes)} bytes)")

    volume.commit()
    print(f"Volume committed. {total_files} files uploaded successfully.")
    return {"status": "success", "files_uploaded": total_files}


# ============================================================================
# Download Models Directory from Volume
# ============================================================================
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600,
)
def download_models_from_volume(already_downloaded_checkpoints: list = None):
    """
    Walk the models directory on the volume and return a dict of
    {relative_path: file_bytes} — but only for NEW/changed content.

    Incremental download logic:
    - Checkpoint directories listed in `already_downloaded_checkpoints` are skipped.
    - Non-checkpoint content (e.g. validation_visualization/) is always included.
    - Signal/hidden files are always skipped.

    This prevents OOM and redundant transfers as checkpoints accumulate.
    """
    import os

    if already_downloaded_checkpoints is None:
        already_downloaded_checkpoints = []
    skip_set = set(already_downloaded_checkpoints)

    volume.reload()  # Ensure we see the latest state

    models_dir = os.path.dirname(TRAINING_CONFIG["output_dir"])  # /data/models
    output_subdir = os.path.basename(TRAINING_CONFIG["output_dir"])  # kontext-i2i-erm
    output_dir_full = TRAINING_CONFIG["output_dir"]  # /data/models/kontext-i2i-erm

    if not os.path.exists(models_dir):
        print(f"Models directory {models_dir} does not exist on volume.")
        return {"status": "empty", "files": {}, "all_checkpoints": []}

    # Enumerate all checkpoint dirs on volume (for the caller to track)
    all_vol_checkpoints = []
    if os.path.isdir(output_dir_full):
        all_vol_checkpoints = sorted(
            [d for d in os.listdir(output_dir_full)
             if os.path.isdir(os.path.join(output_dir_full, d)) and d.startswith("checkpoint")],
            key=lambda x: int(x.rsplit("-", 1)[-1])
        )

    skipped = 0
    file_map = {}
    for root, dirs, files in os.walk(models_dir):
        # Determine which top-level subdir of the output dir this belongs to
        if root.startswith(output_dir_full):
            rel_to_output = os.path.relpath(root, output_dir_full)
            top_level = rel_to_output.split(os.sep)[0] if rel_to_output != "." else None
            # Skip checkpoint directories we've already downloaded
            if top_level and top_level.startswith("checkpoint") and top_level in skip_set:
                skipped += 1
                continue

        for fname in files:
            # Skip signal/hidden files
            if fname.startswith("."):
                continue
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, models_dir)
            try:
                with open(full_path, "rb") as f:
                    file_map[rel_path] = f.read()
            except Exception as e:
                print(f"  Warning: could not read {full_path}: {e}")

    if skip_set:
        print(f"Skipped {skipped} already-downloaded checkpoint dirs. "
              f"Returning {len(file_map)} new/updated files.")
    else:
        print(f"Full download: collected {len(file_map)} files from {models_dir}")

    return {"status": "success", "files": file_map, "all_checkpoints": all_vol_checkpoints}


# ============================================================================
# Poll Training Signals (lightweight, short-lived RPC for polling architecture)
# ============================================================================
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=120,
)
def check_training_signals():
    """
    Read signal files from the volume and return the current set.
    
    This is a lightweight, short-lived RPC (~seconds) that the local entrypoint
    calls periodically instead of maintaining a long-lived gRPC stream.
    Each call is independent — if one fails, the next retry will succeed.
    
    Returns:
        dict with:
        - signals: sorted list of signal file names (e.g. [".checkpoint_saved_epoch_0", ...])
        - training_complete: bool — whether .training_complete marker exists
        - num_checkpoints: int — number of checkpoint directories on volume
    """
    import glob
    import os

    volume.reload()

    output_dir = TRAINING_CONFIG["output_dir"]
    if not os.path.exists(output_dir):
        return {"signals": [], "training_complete": False, "num_checkpoints": 0}

    # Gather all signal files
    signal_patterns = [
        ".checkpoint_saved_epoch_*",
        ".validation_done_epoch_*",
        ".test_done",
    ]
    signals = set()
    for pattern in signal_patterns:
        for f in glob.glob(os.path.join(output_dir, pattern)):
            signals.add(os.path.basename(f))

    # Check for training-complete marker
    training_complete = os.path.exists(os.path.join(output_dir, ".training_complete"))

    # Count checkpoint directories
    num_checkpoints = len([
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("checkpoint")
    ])

    return {
        "signals": sorted(signals),
        "training_complete": training_complete,
        "num_checkpoints": num_checkpoints,
    }


# ============================================================================
# List Volume Checkpoints (for auto-resume detection)
# ============================================================================
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=300,
)
def detect_latest_checkpoint_on_volume():
    """
    Check the Modal volume for existing checkpoints and return info about the latest one.
    Used to decide whether to resume training automatically.
    """
    import os

    volume.reload()

    output_dir = TRAINING_CONFIG["output_dir"]

    if not os.path.exists(output_dir):
        return {"has_checkpoints": False, "latest_checkpoint": None, "all_checkpoints": []}

    checkpoints = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint"):
            checkpoints.append(item)

    if not checkpoints:
        return {"has_checkpoints": False, "latest_checkpoint": None, "all_checkpoints": []}

    # Sort by epoch number (works for "checkpoint-epoch-5" naming)
    checkpoints = sorted(checkpoints, key=lambda x: int(x.rsplit("-", 1)[-1]))
    latest = checkpoints[-1]

    # Read metadata from latest checkpoint if available
    metadata = None
    metadata_path = os.path.join(output_dir, latest, "training_metadata.json")
    if os.path.exists(metadata_path):
        import json
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    print(f"Found {len(checkpoints)} checkpoints on volume. Latest: {latest}")
    if metadata:
        print(f"  Latest checkpoint metadata: epoch={metadata.get('epoch')}, "
              f"global_step={metadata.get('global_step')}")

    return {
        "has_checkpoints": True,
        "latest_checkpoint": latest,
        "all_checkpoints": checkpoints,
        "metadata": metadata,
    }


# ============================================================================
# Multi-GPU Training Function
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=86400,  # 24 hour timeout for long training runs
    memory=8192,  # 8 GB RAM
)
def train_flux_kontext():
    """Train Flux Kontext LoRA for ERM with multi-GPU support using accelerate launch.
    
    Runs training as a subprocess, monitors signal files, and commits the volume
    after each checkpoint/validation/test event.
    
    Automatically detects existing checkpoints on the volume and resumes training
    from the latest one.
    
    NOTE: This is a regular function (not a generator). The local entrypoint
    discovers new signals by polling check_training_signals() independently,
    which avoids long-lived gRPC streams that can die on multi-hour runs.
    """
    import glob
    import os
    import subprocess
    import time

    # Set environment variables
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    # Reload volume to see latest state (important for preemption recovery)
    volume.reload()

    # Clear stale .training_complete from a previous run so the local poller
    # doesn't immediately think we're done.
    _complete_marker = os.path.join(TRAINING_CONFIG["output_dir"], ".training_complete")
    if os.path.exists(_complete_marker):
        os.remove(_complete_marker)
        volume.commit()
        print("[CLEANUP] Removed stale .training_complete from previous run.")

    # Create output directories
    os.makedirs(TRAINING_CONFIG["output_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["logging_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["cache_dir"], exist_ok=True)

    print(
        f"Starting Flux Kontext LoRA (ERM) training with {NUM_GPUS} {GPU_TYPE} GPUs..."
    )
    print(f"Output directory: {TRAINING_CONFIG['output_dir']}")
    print("Task: Edit Region Mask (ERM) Prediction")

    # ---- Auto-detect latest checkpoint for resume ----
    resume_checkpoint = None
    dirs = [d for d in os.listdir(TRAINING_CONFIG["output_dir"])
            if os.path.isdir(os.path.join(TRAINING_CONFIG["output_dir"], d))
            and d.startswith("checkpoint")]
    if dirs:
        dirs = sorted(dirs, key=lambda x: int(x.rsplit("-", 1)[-1]))
        resume_checkpoint = "latest"
        print(f"\n[AUTO-RESUME] Found {len(dirs)} checkpoints on volume: {dirs}")
        print(f"[AUTO-RESUME] Will resume from latest checkpoint: {dirs[-1]}")
    else:
        print("\n[AUTO-RESUME] No checkpoints found on volume. Starting training from scratch.")

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
    cmd = build_training_command(TRAINING_CONFIG, NUM_GPUS)

    # Add resume flag if checkpoints exist
    if resume_checkpoint:
        cmd.extend(["--resume_from_checkpoint", resume_checkpoint])

    print(f"Running command:\n{' '.join(cmd)}\n")
    print("=" * 80)

    # Track which signal files we've already processed
    def _gather_signal_files():
        return set(
            glob.glob(os.path.join(TRAINING_CONFIG["output_dir"], ".checkpoint_saved_epoch_*"))
            + glob.glob(os.path.join(TRAINING_CONFIG["output_dir"], ".validation_done_epoch_*"))
            + glob.glob(os.path.join(TRAINING_CONFIG["output_dir"], ".test_done"))
        )

    processed_signals = _gather_signal_files()

    try:
        # Run training as a subprocess (Popen) so we can monitor signal files
        process = subprocess.Popen(
            cmd,
            cwd="/root/flux_finetuning_erm",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Monitor the process output and signal files
        import select
        while process.poll() is None:
            # Print subprocess output in real-time
            if process.stdout:
                # Read available lines (non-blocking via select)
                ready, _, _ = select.select([process.stdout], [], [], 2.0)
                if ready:
                    line = process.stdout.readline()
                    if line:
                        print(line, end="", flush=True)

            # Check for new signal files
            current_signals = _gather_signal_files()
            new_signals = current_signals - processed_signals
            if new_signals:
                signal_names = []
                for sig in sorted(new_signals):
                    sig_name = os.path.basename(sig)
                    signal_names.append(sig_name)
                    print(f"\n[VOLUME COMMIT] Detected signal: {sig_name}")
                processed_signals = current_signals
                print("[VOLUME COMMIT] Committing volume to persist latest data...")
                volume.commit()
                print("[VOLUME COMMIT] Volume committed successfully.")

        # Drain remaining output
        if process.stdout:
            for line in process.stdout:
                print(line, end="", flush=True)

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        raise
    finally:
        # Stop GPU monitoring
        if ENABLE_GPU_MONITORING and stop_monitoring is not None:
            print("\nStopping GPU monitoring...")
            stop_monitoring.set()
            if monitor_thread is not None:
                monitor_thread.join(timeout=5)

    # Write training-complete signal so the local poller knows we're done
    import pathlib
    pathlib.Path(os.path.join(TRAINING_CONFIG["output_dir"], ".training_complete")).touch()

    # Final volume commit
    print("\nFinal volume commit to persist all trained model data...")
    volume.commit()

    return {
        "status": "success",
        "output_dir": TRAINING_CONFIG["output_dir"],
        "num_gpus": NUM_GPUS,
        "gpu_type": GPU_TYPE,
        "task": "Edit Region Mask (ERM) Prediction",
        "message": f"Flux Kontext LoRA (ERM) training completed on {NUM_GPUS} {GPU_TYPE} GPUs",
    }


# ============================================================================
# Single-GPU Training Function (for smaller runs or debugging)
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=86400,  # 24 hour timeout
    memory=8192,  # 8 GB RAM
)
def train_flux_kontext_single_gpu():
    """Train Flux Kontext LoRA for ERM on a single GPU.
    
    Runs training as a subprocess, monitors signal files, and commits the volume
    after each checkpoint/validation/test event.
    
    Automatically detects existing checkpoints on the volume and resumes training
    from the latest one.
    
    NOTE: This is a regular function (not a generator). The local entrypoint
    discovers new signals by polling check_training_signals() independently,
    which avoids long-lived gRPC streams that can die on multi-hour runs.
    """
    import glob
    import os
    import subprocess
    import time

    # Set environment variables
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    # Reload volume to see latest state (important for preemption recovery)
    volume.reload()

    # Clear stale .training_complete from a previous run
    _complete_marker = os.path.join(TRAINING_CONFIG["output_dir"], ".training_complete")
    if os.path.exists(_complete_marker):
        os.remove(_complete_marker)
        volume.commit()
        print("[CLEANUP] Removed stale .training_complete from previous run.")

    # Create output directories
    os.makedirs(TRAINING_CONFIG["output_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["logging_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["cache_dir"], exist_ok=True)

    print("Starting Flux Kontext LoRA (ERM) training on single GPU...")
    print("Task: Edit Region Mask (ERM) Prediction")

    # ---- Auto-detect latest checkpoint for resume ----
    resume_checkpoint = None
    dirs = [d for d in os.listdir(TRAINING_CONFIG["output_dir"])
            if os.path.isdir(os.path.join(TRAINING_CONFIG["output_dir"], d))
            and d.startswith("checkpoint")]
    if dirs:
        dirs = sorted(dirs, key=lambda x: int(x.rsplit("-", 1)[-1]))
        resume_checkpoint = "latest"
        print(f"\n[AUTO-RESUME] Found {len(dirs)} checkpoints on volume: {dirs}")
        print(f"[AUTO-RESUME] Will resume from latest checkpoint: {dirs[-1]}")
    else:
        print("\n[AUTO-RESUME] No checkpoints found on volume. Starting training from scratch.")

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

    # Build command for single GPU
    cmd = build_training_command(TRAINING_CONFIG, 1)

    # Add resume flag if checkpoints exist
    if resume_checkpoint:
        cmd.extend(["--resume_from_checkpoint", resume_checkpoint])

    print(f"Running command:\n{' '.join(cmd)}\n")
    print("=" * 80)

    # Track which signal files we've already processed
    def _gather_signal_files():
        return set(
            glob.glob(os.path.join(TRAINING_CONFIG["output_dir"], ".checkpoint_saved_epoch_*"))
            + glob.glob(os.path.join(TRAINING_CONFIG["output_dir"], ".validation_done_epoch_*"))
            + glob.glob(os.path.join(TRAINING_CONFIG["output_dir"], ".test_done"))
        )

    processed_signals = _gather_signal_files()

    try:
        # Run training as a subprocess (Popen) so we can monitor signal files
        process = subprocess.Popen(
            cmd,
            cwd="/root/flux_finetuning_erm",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Monitor the process output and signal files
        import select
        while process.poll() is None:
            # Print subprocess output in real-time
            if process.stdout:
                ready, _, _ = select.select([process.stdout], [], [], 2.0)
                if ready:
                    line = process.stdout.readline()
                    if line:
                        print(line, end="", flush=True)

            # Check for new signal files
            current_signals = _gather_signal_files()
            new_signals = current_signals - processed_signals
            if new_signals:
                signal_names = []
                for sig in sorted(new_signals):
                    sig_name = os.path.basename(sig)
                    signal_names.append(sig_name)
                    print(f"\n[VOLUME COMMIT] Detected signal: {sig_name}")
                processed_signals = current_signals
                print("[VOLUME COMMIT] Committing volume to persist latest data...")
                volume.commit()
                print("[VOLUME COMMIT] Volume committed successfully.")

        # Drain remaining output
        if process.stdout:
            for line in process.stdout:
                print(line, end="", flush=True)

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        raise
    finally:
        # Stop GPU monitoring
        if ENABLE_GPU_MONITORING and stop_monitoring is not None:
            print("\nStopping GPU monitoring...")
            stop_monitoring.set()
            if monitor_thread is not None:
                monitor_thread.join(timeout=5)

    # Write training-complete signal so the local poller knows we're done
    import pathlib
    pathlib.Path(os.path.join(TRAINING_CONFIG["output_dir"], ".training_complete")).touch()

    # Final volume commit
    print("\nFinal volume commit to persist all trained model data...")
    volume.commit()

    return {
        "status": "success",
        "output_dir": TRAINING_CONFIG["output_dir"],
        "task": "Edit Region Mask (ERM) Prediction",
        "message": "Flux Kontext LoRA (ERM) training completed on single GPU",
    }


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
    prompt: str = "Move the robotic arm to the right",
    input_image_url: str = None,
):
    """
    Run inference with the trained Flux Kontext LoRA model for ERM prediction.

    Args:
        lora_path: Path to the LoRA weights (defaults to training output_dir)
        prompt: Edit instruction prompt
        input_image_url: URL to the input image
    """
    import os
    import sys

    sys.path.insert(0, "/root/flux_finetuning_erm")

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

    # Append mask instruction to prompt (as done in training)
    mask_instruction = "Predict a binary mask for given image: 1 for regions to be edited per the instruction above, 0 for unchanged regions, consistent across all channels"
    full_prompt = f"{prompt}\n\n{mask_instruction}"

    # Run inference
    print(f"Running inference with prompt: {prompt}")
    print("(Mask instruction appended automatically)")
    output = pipeline(
        prompt=full_prompt,
        image=input_image,
        guidance_scale=TRAINING_CONFIG.get("guidance_scale", 1.0),
        num_inference_steps=28,
    )

    # Save output
    output_path = "/data/inference_output_erm.png"
    output.images[0].save(output_path)
    print(f"Output saved to {output_path}")

    volume.commit()

    return {
        "status": "success",
        "output_path": output_path,
        "prompt": prompt,
        "task": "Edit Region Mask (ERM) Prediction",
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
    
    Regular function that monitors signal files and commits the volume after
    each event. The local entrypoint discovers new signals by polling
    check_training_signals() independently.

    Args:
        checkpoint_path: Path to checkpoint or "latest" to auto-detect
    """
    import glob
    import os
    import subprocess

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    # Reload volume to see latest state
    volume.reload()

    # Clear stale .training_complete from a previous run
    _complete_marker = os.path.join(TRAINING_CONFIG["output_dir"], ".training_complete")
    if os.path.exists(_complete_marker):
        os.remove(_complete_marker)
        volume.commit()
        print("[CLEANUP] Removed stale .training_complete from previous run.")

    print(
        f"Resuming Flux Kontext LoRA (ERM) training from checkpoint: {checkpoint_path}"
    )

    # Create output directories (needed for new runs)
    os.makedirs(TRAINING_CONFIG["output_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["logging_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["cache_dir"], exist_ok=True)

    # Start GPU monitoring
    stop_monitoring = None
    if ENABLE_GPU_MONITORING:
        stop_monitoring, _ = start_gpu_monitoring(interval=GPU_MONITORING_INTERVAL)

    # Build command with resume flag
    cmd = build_training_command(TRAINING_CONFIG, NUM_GPUS)
    cmd.extend(["--resume_from_checkpoint", checkpoint_path])

    print(f"Running command:\n{' '.join(cmd)}\n")

    # Track which signal files we've already processed
    def _gather_signal_files():
        return set(
            glob.glob(os.path.join(TRAINING_CONFIG["output_dir"], ".checkpoint_saved_epoch_*"))
            + glob.glob(os.path.join(TRAINING_CONFIG["output_dir"], ".validation_done_epoch_*"))
            + glob.glob(os.path.join(TRAINING_CONFIG["output_dir"], ".test_done"))
        )

    processed_signals = _gather_signal_files()

    try:
        process = subprocess.Popen(
            cmd,
            cwd="/root/flux_finetuning_erm",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        import select
        while process.poll() is None:
            if process.stdout:
                ready, _, _ = select.select([process.stdout], [], [], 2.0)
                if ready:
                    line = process.stdout.readline()
                    if line:
                        print(line, end="", flush=True)

            current_signals = _gather_signal_files()
            new_signals = current_signals - processed_signals
            if new_signals:
                signal_names = []
                for sig in sorted(new_signals):
                    sig_name = os.path.basename(sig)
                    signal_names.append(sig_name)
                    print(f"\n[VOLUME COMMIT] Detected signal: {sig_name}")
                processed_signals = current_signals
                print("[VOLUME COMMIT] Committing volume...")
                volume.commit()
                print("[VOLUME COMMIT] Volume committed successfully.")

        if process.stdout:
            for line in process.stdout:
                print(line, end="", flush=True)

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
    finally:
        if stop_monitoring:
            stop_monitoring.set()

    # Write training-complete signal so the local poller knows we're done
    import pathlib
    pathlib.Path(os.path.join(TRAINING_CONFIG["output_dir"], ".training_complete")).touch()

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
        checkpoints, key=lambda x: int(x.rsplit("-", 1)[-1])
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
# Helper: Upload local models directory to volume
# ============================================================================
def _upload_local_checkpoints_if_present(script_dir: str):
    """
    Check if a 'models' directory with checkpoints exists next to the modal runner
    script. If so, upload **only the checkpoint directories that are missing from the
    Modal volume** (per-checkpoint deduplication). Non-checkpoint files/dirs (e.g.
    validation_visualization/) are always uploaded.
    """
    local_models_dir = os.path.join(script_dir, "models")
    if not os.path.isdir(local_models_dir):
        print("[UPLOAD] No local 'models' directory found. Skipping upload.")
        return False

    # Local output subdir: e.g. "kontext-i2i-erm"
    output_basename = os.path.basename(TRAINING_CONFIG["output_dir"])
    local_ckpt_dir = os.path.join(local_models_dir, output_basename)

    if not os.path.isdir(local_ckpt_dir):
        print(f"[UPLOAD] Local output directory {local_ckpt_dir} not found. Skipping upload.")
        return False

    # Enumerate local checkpoint dirs and other (non-checkpoint) dirs/files
    local_checkpoint_names = set()
    for item in os.listdir(local_ckpt_dir):
        if os.path.isdir(os.path.join(local_ckpt_dir, item)) and item.startswith("checkpoint"):
            local_checkpoint_names.add(item)

    if not local_checkpoint_names:
        print("[UPLOAD] Local output directory contains no checkpoints. Skipping upload.")
        return False

    print(f"[UPLOAD] Found {len(local_checkpoint_names)} local checkpoints in {local_ckpt_dir}")

    # ---- Per-checkpoint deduplication: query volume for existing checkpoints ----
    print("[UPLOAD] Checking volume for existing checkpoints...")
    vol_info = detect_latest_checkpoint_on_volume.remote()

    volume_checkpoint_names = set(vol_info.get("all_checkpoints", []))
    if volume_checkpoint_names:
        print(f"[UPLOAD] Volume already has {len(volume_checkpoint_names)} checkpoints: "
              f"{sorted(volume_checkpoint_names, key=lambda x: int(x.rsplit('-', 1)[-1]))}")

    # Checkpoints that exist locally but NOT on the volume
    new_checkpoints = local_checkpoint_names - volume_checkpoint_names
    skipped_checkpoints = local_checkpoint_names & volume_checkpoint_names

    if skipped_checkpoints:
        print(f"[UPLOAD] Skipping {len(skipped_checkpoints)} checkpoints already on volume: "
              f"{sorted(skipped_checkpoints, key=lambda x: int(x.rsplit('-', 1)[-1]))}")

    if not new_checkpoints:
        print("[UPLOAD] All local checkpoints already exist on volume. Nothing to upload.")
        return False

    print(f"[UPLOAD] Will upload {len(new_checkpoints)} NEW checkpoints: "
          f"{sorted(new_checkpoints, key=lambda x: int(x.rsplit('-', 1)[-1]))}")

    # ---- Collect files to upload (only from new checkpoint dirs + non-checkpoint content) ----
    volume_models_base = os.path.dirname(TRAINING_CONFIG["output_dir"])  # /data/models
    file_map = {}  # remote_path -> local_path
    total_size = 0

    for root, dirs, files in os.walk(local_ckpt_dir):
        # Determine which top-level subdir of local_ckpt_dir this path belongs to
        rel_to_ckpt_dir = os.path.relpath(root, local_ckpt_dir)
        top_level_dir = rel_to_ckpt_dir.split(os.sep)[0] if rel_to_ckpt_dir != "." else None

        # Skip checkpoint directories that already exist on the volume
        if top_level_dir and top_level_dir.startswith("checkpoint") and top_level_dir not in new_checkpoints:
            continue

        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_models_dir)
            remote_path = os.path.join(volume_models_base, rel_path)
            file_size = os.path.getsize(local_path)
            total_size += file_size
            file_map[remote_path] = local_path

    if not file_map:
        print("[UPLOAD] No new files to upload after deduplication.")
        return False

    print(f"[UPLOAD] Collected {len(file_map)} files ({total_size / 1024 / 1024:.1f} MB) to upload.")

    # Upload in batches to avoid memory issues (50 MB per batch)
    BATCH_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
    batch = {}
    batch_size = 0
    batch_num = 0

    for remote_path, local_path in file_map.items():
        with open(local_path, "rb") as f:
            data = f.read()
        batch[remote_path] = data
        batch_size += len(data)

        if batch_size >= BATCH_SIZE_BYTES:
            batch_num += 1
            print(f"[UPLOAD] Uploading batch {batch_num} ({len(batch)} files, {batch_size / 1024 / 1024:.1f} MB)...")
            upload_checkpoints_to_volume.remote(batch)
            batch = {}
            batch_size = 0

    # Upload remaining files
    if batch:
        batch_num += 1
        print(f"[UPLOAD] Uploading batch {batch_num} ({len(batch)} files, {batch_size / 1024 / 1024:.1f} MB)...")
        upload_checkpoints_to_volume.remote(batch)

    print(f"[UPLOAD] Done! Uploaded {len(file_map)} files for {len(new_checkpoints)} new checkpoints.")
    return True


# ============================================================================
# Helper: Download models from volume to local directory
# ============================================================================
# Tracks which checkpoint directories have already been downloaded locally.
# This set is populated across download calls within a single `modal run` session
# so that subsequent per-event downloads only fetch NEW checkpoints.
_downloaded_checkpoints: set = set()


def _download_models_to_local(script_dir: str):
    """
    Incrementally download the models directory from the Modal volume.

    - On the first call, downloads everything (full sync).
    - On subsequent calls, skips checkpoint dirs that were already downloaded,
      only fetching newly created checkpoints + non-checkpoint content
      (e.g. validation_visualization/).

    This prevents OOM and avoids re-downloading ~210 MB per existing checkpoint.
    """
    global _downloaded_checkpoints

    local_models_dir = os.path.join(script_dir, "models")

    # On first call, also seed the skip-set from what's already on disk
    if not _downloaded_checkpoints:
        output_subdir = os.path.basename(TRAINING_CONFIG["output_dir"])
        local_output_dir = os.path.join(local_models_dir, output_subdir)
        if os.path.isdir(local_output_dir):
            for d in os.listdir(local_output_dir):
                if os.path.isdir(os.path.join(local_output_dir, d)) and d.startswith("checkpoint"):
                    _downloaded_checkpoints.add(d)
            if _downloaded_checkpoints:
                print(f"[DOWNLOAD] Found {len(_downloaded_checkpoints)} checkpoints already on disk. "
                      f"Will skip re-downloading them.")

    print(f"[DOWNLOAD] Downloading models from volume to {local_models_dir}...")
    if _downloaded_checkpoints:
        print(f"[DOWNLOAD]   (skipping {len(_downloaded_checkpoints)} already-downloaded checkpoints)")

    result = download_models_from_volume.remote(
        already_downloaded_checkpoints=list(_downloaded_checkpoints)
    )

    if result["status"] == "empty":
        print("[DOWNLOAD] No models found on volume.")
        return False

    files = result["files"]
    all_vol_checkpoints = result.get("all_checkpoints", [])
    total_bytes = sum(len(b) for b in files.values())
    print(f"[DOWNLOAD] Received {len(files)} files ({total_bytes / 1024 / 1024:.1f} MB) from volume.")

    for rel_path, file_bytes in files.items():
        local_path = os.path.join(local_models_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(file_bytes)

    # Update our tracking set with ALL checkpoints now on the volume
    # (they're either already on disk or just downloaded)
    for ckpt_name in all_vol_checkpoints:
        _downloaded_checkpoints.add(ckpt_name)

    print(f"[DOWNLOAD] Saved {len(files)} files to {local_models_dir}")
    print(f"[DOWNLOAD] Total checkpoints tracked locally: {len(_downloaded_checkpoints)}")
    return True


# ============================================================================
# Helper: Poll-based signal monitoring with incremental downloads
# ============================================================================
POLL_INTERVAL_SECONDS = 90      # seconds between polls
POLL_MAX_CONSECUTIVE_FAILURES = 5  # give up after N consecutive poll failures
POLL_STALE_MINUTES = 180        # warn if no new signal for this long


def _poll_and_download(script_dir: str, poll_interval: int = POLL_INTERVAL_SECONDS):
    """
    Poll the volume for new training signals and download incrementally.

    Architecture:
    - Each poll is an independent short-lived RPC (~seconds).
    - If one poll fails, we retry on the next cycle.
    - No long-lived gRPC stream — fully resilient to network interruptions.
    - Exits when .training_complete signal appears.

    This replaces the old remote_gen() streaming pattern which was vulnerable
    to StreamTerminatedError on multi-hour training runs.
    """
    import time

    known_signals: set = set()
    event_count = 0
    consecutive_failures = 0
    last_new_signal_time = time.time()

    # First poll: take a baseline snapshot (anything here is from before we launched)
    # We still download for these (incremental download handles dedup), but we
    # do NOT treat a pre-existing .training_complete as meaning "done".
    try:
        initial = check_training_signals.remote()
        baseline_signals = set(initial["signals"])
        baseline_had_complete = initial["training_complete"]
        if baseline_signals:
            print(f"[POLL] Baseline: {len(baseline_signals)} pre-existing signals "
                  f"({initial['num_checkpoints']} checkpoints on volume)")
            if baseline_had_complete:
                print("[POLL] Note: Found stale .training_complete from a previous run — ignoring it.")
        known_signals = baseline_signals.copy()
    except Exception as e:
        print(f"[POLL] Warning: initial poll failed ({e}), starting with empty baseline.")
        baseline_had_complete = False

    print(f"\n[POLL] Starting signal polling (every {poll_interval}s)...")
    print("[POLL] Each poll is an independent RPC — resilient to network interruptions.")
    print("[POLL] Press Ctrl+C to stop polling (training continues on Modal regardless).\n")

    while True:
        time.sleep(poll_interval)

        # ---- Poll ----
        try:
            poll_result = check_training_signals.remote()
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            print(f"[POLL] Warning: poll failed ({consecutive_failures}/"
                  f"{POLL_MAX_CONSECUTIVE_FAILURES}): {e}")
            if consecutive_failures >= POLL_MAX_CONSECUTIVE_FAILURES:
                print(f"[POLL] ERROR: {POLL_MAX_CONSECUTIVE_FAILURES} consecutive failures. "
                      f"Doing a final download attempt...")
                try:
                    _download_models_to_local(script_dir)
                except Exception:
                    pass
                print("[POLL] Training is still running on Modal. Re-run with "
                      "'--action download_models' later to fetch results.")
                return None
            continue

        current_signals = set(poll_result["signals"])
        training_complete = poll_result["training_complete"]
        num_checkpoints = poll_result["num_checkpoints"]

        new_signals = current_signals - known_signals

        # ---- React to new signals ----
        if new_signals:
            last_new_signal_time = time.time()
            for sig in sorted(new_signals):
                event_count += 1
                print(f"\n[POLL #{event_count}] New signal detected: {sig}")
            known_signals = current_signals

            print(f"[POLL] Downloading new content ({num_checkpoints} checkpoints on volume)...")
            try:
                _download_models_to_local(script_dir)
                print(f"[POLL] Download complete.\n")
            except Exception as e:
                print(f"[POLL] Warning: download failed ({e}), will retry next cycle.\n")
        else:
            idle_minutes = (time.time() - last_new_signal_time) / 60
            status_parts = [f"{num_checkpoints} ckpts"]
            if idle_minutes > 5:
                status_parts.append(f"idle {idle_minutes:.0f}m")
            print(f"[POLL] No new signals. ({', '.join(status_parts)})")
            if idle_minutes > POLL_STALE_MINUTES:
                print(f"[POLL] WARNING: No new signal for {idle_minutes:.0f} minutes. "
                      f"Training may have stalled or crashed on Modal.")

        # ---- Check for training completion ----
        # Only trust .training_complete if it's NEW (not from a previous run)
        if training_complete and not baseline_had_complete:
            print(f"\n{'=' * 60}")
            print(f"[POLL] Training complete signal detected!")
            print(f"[POLL] Final download...")
            try:
                _download_models_to_local(script_dir)
            except Exception as e:
                print(f"[POLL] Warning: final download failed ({e})")
            print(f"[POLL] All done. Processed {event_count} new signal events.")
            print(f"{'=' * 60}")
            return {"status": "success", "signals_processed": event_count}
        elif training_complete and baseline_had_complete:
            # The .training_complete was already there at baseline.
            # It might have been re-created by the new run — check if
            # we've seen at least one new signal since launch, which
            # would confirm a new run happened.
            if event_count > 0:
                print(f"\n{'=' * 60}")
                print(f"[POLL] Training complete signal detected (re-confirmed after new signals)!")
                print(f"[POLL] Final download...")
                try:
                    _download_models_to_local(script_dir)
                except Exception as e:
                    print(f"[POLL] Warning: final download failed ({e})")
                print(f"[POLL] All done. Processed {event_count} new signal events.")
                print(f"{'=' * 60}")
                return {"status": "success", "signals_processed": event_count}


# ============================================================================
# Local Entrypoint
# ============================================================================
@app.local_entrypoint()
def main(
    action: str = "train",
    checkpoint: str = "latest",
    prompt: str = "Move the robotic arm to the right",
    image_url: str = None,
    single_gpu: bool = True,
):
    """
    Local entrypoint to run training, inference, or utilities.

    Architecture:
    - Training is launched via .spawn() (fire-and-forget) on Modal.
    - The local process polls the volume for new signal files every ~90 seconds.
    - Each poll is an independent short-lived RPC (~seconds).
    - This is fully resilient to network interruptions — no long-lived gRPC
      stream that can die on multi-hour runs (unlike the old remote_gen() pattern).
    - If the local process dies, training continues on Modal unaffected.
      Re-run with --action download_models to fetch results later.

    Features:
    - Automatically uploads local checkpoints to the volume before training
    - Automatically resumes from the latest checkpoint (preemption-safe)
    - Incrementally downloads only new checkpoints after each training event

    Args:
        action: One of "train", "resume", "inference", "download", "list_checkpoints", "download_models"
        checkpoint: Checkpoint path for resume (default: "latest")
        prompt: Edit prompt for inference
        image_url: Input image URL for inference
        single_gpu: Use single GPU instead of multi-GPU
    """
    # Determine the directory where THIS script lives (for local models path)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print(f"Flux Kontext LoRA (ERM) Finetuning - Action: {action}")
    print("=" * 60)

    result = None

    if action == "train":
        # ---- Step 1: Upload local checkpoints to volume if present ----
        print("\n--- Step 1: Check for local checkpoints to upload ---")
        _upload_local_checkpoints_if_present(script_dir)

        # ---- Step 2: Launch training (fire-and-forget via .spawn()) ----
        print("\n--- Step 2: Launch training on Modal ---")
        if single_gpu:
            print("Spawning single-GPU training...")
            fc = train_flux_kontext_single_gpu.spawn()
        else:
            print(f"Spawning multi-GPU training on {NUM_GPUS} {GPU_TYPE} GPUs...")
            fc = train_flux_kontext.spawn()

        print("Training launched on Modal (fire-and-forget).")
        print("Remote function runs independently — local disconnects won't affect it.")

        # ---- Step 3: Poll for signals and download incrementally ----
        print("\n--- Step 3: Poll-based monitoring & incremental downloads ---")
        result = _poll_and_download(script_dir)

        if result:
            print("\nTraining completed successfully!")
            print(f"  Signals processed: {result.get('signals_processed', '?')}")
        else:
            print("\nPolling ended (network issues or timeout).")
            print("Training is still running on Modal.")
            print("Run with '--action download_models' to fetch results later.")

    elif action == "resume":
        # Upload local checkpoints first (in case we have newer ones locally)
        print("\n--- Step 1: Check for local checkpoints to upload ---")
        _upload_local_checkpoints_if_present(script_dir)

        # ---- Step 2: Launch resume training (fire-and-forget via .spawn()) ----
        print(f"\n--- Step 2: Launch resume training (from checkpoint: {checkpoint}) ---")
        fc = resume_training.spawn(checkpoint_path=checkpoint)
        print("Resume training launched on Modal (fire-and-forget).")

        # ---- Step 3: Poll for signals and download incrementally ----
        print("\n--- Step 3: Poll-based monitoring & incremental downloads ---")
        result = _poll_and_download(script_dir)

        if result:
            print("\nResumed training completed successfully!")
            print(f"  Signals processed: {result.get('signals_processed', '?')}")
        else:
            print("\nPolling ended (network issues or timeout).")
            print("Training is still running on Modal.")
            print("Run with '--action download_models' to fetch results later.")

    elif action == "inference":
        print("Running inference...")
        result = run_inference.remote(prompt=prompt, input_image_url=image_url)
        print("\nInference Result:")
        print(f"  Status: {result['status']}")
        print(f"  Output: {result['output_path']}")
        print(f"  Task: {result.get('task', 'ERM')}")

    elif action == "download":
        print("Downloading and caching dataset...")
        result = download_dataset.remote()
        print("\nDownload Result:")
        print(f"  Status: {result['status']}")
        print(f"  Message: {result['message']}")

    elif action == "download_models":
        # Explicit action to download models from volume
        print("Downloading models from volume...")
        _download_models_to_local(script_dir)
        result = {"status": "success", "message": "Models downloaded from volume"}

    elif action == "list_checkpoints":
        print("Listing checkpoints...")
        result = list_checkpoints.remote()
        print("\nCheckpoints:")
        print(f"  Output dir: {result['output_dir']}")
        print(f"  Total: {result.get('total_checkpoints', 0)}")
        for cp in result.get("checkpoints", []):
            print(f"    - {cp}")
        print(f"  Final weights saved: {result.get('has_final_weights', False)}")

    else:
        raise ValueError(
            f"Invalid action: {action}. "
            "Use 'train', 'resume', 'inference', 'download', 'download_models', or 'list_checkpoints'"
        )

    print("=" * 60)
    return result
