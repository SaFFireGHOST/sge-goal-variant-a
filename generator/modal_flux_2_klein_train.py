"""
Modal script to launch Flux 2 Klein LoRA finetuning with multi-GPU support using accelerate.

This script uses Modal's serverless infrastructure to train the Flux 2 Klein model
across multiple GPUs using accelerate launch for distributed training.

Usage:
    # Run training
    modal run modal_flux_2_klein_train.py --action train

    # Run inference only (after training)
    modal run modal_flux_2_klein_train.py --action inference

    # Download dataset first (optional, for caching)
    modal run modal_flux_2_klein_train.py --action download
"""

import modal
import os

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
    "pretrained_model_name_or_path": "black-forest-labs/FLUX.2-klein-base-9B",
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
    "output_dir": "/data/models/flux-2-klein-lora-calvin",
    "logging_dir": "/data/logs/flux-2-klein-calvin",
    "cache_dir": "/data/datasets/.cache",
    # Training hyperparameters
    "resolution": 256,
    "train_batch_size": 160,
    "gradient_accumulation_steps": 1,
    "num_train_epochs": 75,
    # max_train_steps intentionally omitted; epochs take precedence
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
    # Checkpointing
    "checkpointing_epochs": 1,
    "checkpoints_total_limit": 75,
    "validation_epochs": 1,
    "resume_from_checkpoint": None,
    # Other settings
    "seed": 0,
    "max_sequence_length": 512,
    "guidance_scale": 1.0,  # Match Kontext
    "weighting_scheme": "none",
    "dataloader_num_workers": 4,
    "train_text_encoder": False,
    "report_to": "tensorboard",
    # Static loss hyperparameters
    "t_rec_max": 0.35,
    "p_rec_fraction": 1.0,
    "lambda_static": 7.0,
    # Validation/Testing configuration
    "max_val_samples": 15,
    "max_test_samples": None,
    "eval_viz_num": 10,
    "val_batch_size": 150,
    "test_batch_size": 150,
}

# ============================================================================
# Modal App Setup
# ============================================================================
app = modal.App("flux2-klein-calvin-image-editing")

# Create a volume for persistent storage
volume = modal.Volume.from_name("flux2-klein-calvin-image-editing-volume", create_if_missing=True)

# Volume-relative path for models (mirrors output_dir structure)
# TRAINING_CONFIG["output_dir"] = "/data/models/flux-2-klein-lora"
# So models dir on the volume is "models" (relative to /data mount)
VOLUME_MODELS_REMOTE_DIR = "models"  # Relative to volume root (/data/)
LOCAL_MODELS_DIR = "models"  # Relative to CWD (flux_2_klein_finetuning/)

# CUDA Configuration
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# ============================================================================
# Define the Modal image with all required dependencies
# ============================================================================
LOCAL_DIR = os.path.dirname(__file__)

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    # Install PyTorch first (stable wheels, like the Kontext runner)
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision",
        "torchaudio",
    )
    # Install main dependencies
    .uv_pip_install(
        "accelerate",
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
        "tensorboard",
    )
    # Install diffusers from git for Flux 2 Klein support
    .run_commands("apt-get update && apt-get install -y git")
    .run_commands("pip install --upgrade pip setuptools wheel")
    .run_commands("pip install git+https://github.com/huggingface/diffusers.git")
    # Install flash-attn (needs torch present)
    .run_commands("pip install flash-attn --no-build-isolation")
    # Set environment variables
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": "/data/datasets/.cache",
        }
    )
    # Add only the required files to minimize mount size
    .add_local_file(
        os.path.join(LOCAL_DIR, "flux_2_klein_lora_finetuning.py"),
        remote_path="/root/flux_2_klein_finetuning/flux_2_klein_lora_finetuning.py",
    )
    .add_local_file(
        os.path.join(LOCAL_DIR, "modal_flux_2_klein_train.py"),
        remote_path="/root/flux_2_klein_finetuning/modal_flux_2_klein_train.py",
    )
)

# HuggingFace and WandB secrets
huggingface_secret = modal.Secret.from_name(
    "huggingface-secret-2"
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
                            f"Temp={temp}C | Power={power}W"
                        )

            except Exception as e:
                print(f"Error monitoring GPU: {e}")

            stop_monitoring.wait(interval)

    monitor_thread = threading.Thread(target=monitor_gpu_utilization, daemon=True)
    monitor_thread.start()

    return stop_monitoring, monitor_thread


def verify_hf_access(model_id: str, token: str | None) -> None:
    """Fail fast with a clear message if the HF token cannot access the model."""
    if not token:
        print("HF token NOT found. Cannot verify gated model access.")
        return

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError
    except Exception:
        print("huggingface_hub not available to verify access.")
        return

    try:
        hf_hub_download(
            repo_id=model_id,
            filename="tokenizer/tokenizer_config.json",
            token=token,
            local_files_only=False,
        )
    except GatedRepoError as e:
        raise RuntimeError(
            "HF token does not have access to the gated repo. "
            "Request access and update the Modal secret 'huggingface-secret-2'."
        ) from e


def build_training_command(config: dict, num_gpus: int, script_name: str = "flux_2_klein_lora_finetuning.py") -> list:
    """
    Build the CLI command for running the training script. This attempts to robustly locate the training
    script on the container filesystem and raises a helpful error if it cannot be found.

    Args:
        config: Training configuration dictionary
        num_gpus: Number of GPUs to use
        script_name: The name of the training script file to look for.

    Returns:
        List of command arguments
    """
    import os

    # Candidate locations to check
    candidates = [
        f"/root/flux_2_klein_finetuning/{script_name}",
        f"/root/{script_name}",
        os.path.join(os.getcwd(), script_name),
    ]

    script_path = None
    for c in candidates:
        if os.path.exists(c):
            script_path = c
            break

    # If not found, do a quick search under /root to help identify where files landed
    if script_path is None:
        for root_dir, _, files in os.walk("/root"):
            if script_name in files:
                script_path = os.path.join(root_dir, script_name)
                break

    if script_path is None:
        # Provide diagnostic info to help debugging
        root_list = []
        try:
            root_list = os.listdir("/root")
        except Exception:
            pass
        raise FileNotFoundError(
            f"Training script '{script_name}' not found. Searched candidates: {candidates}. /root contents: {root_list}"
        )

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
            script_path,
        ]
    else:
        # Single GPU: run python directly
        cmd = [
            "python",
            "-u",
            script_path,
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
        "max_train_steps": "--max_train_steps",
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
        "resume_from_checkpoint": "--resume_from_checkpoint",
        "seed": "--seed",
        "max_sequence_length": "--max_sequence_length",
        "guidance_scale": "--guidance_scale",
        "weighting_scheme": "--weighting_scheme",
        "dataloader_num_workers": "--dataloader_num_workers",
        "max_val_samples": "--max_val_samples",
        "max_test_samples": "--max_test_samples",
        "eval_viz_num": "--eval_viz_num",
        "val_batch_size": "--val_batch_size",
        "test_batch_size": "--test_batch_size",
        "mixed_precision": "--mixed_precision",
        "report_to": "--report_to",
        "repeats": "--repeats",
        "t_rec_max": "--t_rec_max",
        "p_rec_fraction": "--p_rec_fraction",
        "lambda_static": "--lambda_static",
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
    # Always enable TF32
    cmd.append("--allow_tf32")

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
    dirs = [
        d
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not dirs:
        return None

    def _sort_key(name: str) -> int:
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
        for item in os.listdir(output_dir):
            if item.startswith("checkpoint"):
                all_checkpoints.append(item)

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
    try:
        if hasattr(dataset, "keys"):
            train_size = getattr(dataset["train"], "num_rows", None) if "train" in dataset else None
            val_size = getattr(dataset["validation"], "num_rows", None) if "validation" in dataset else None
            test_size = getattr(dataset["test"], "num_rows", None) if "test" in dataset else None
            print(f"  Train size: {train_size if train_size is not None else 'unknown'}")
            print(f"  Validation size: {val_size if val_size is not None else 'unknown'}")
            print(f"  Test size: {test_size if test_size is not None else 'unknown'}")
        else:
            size = getattr(dataset, "num_rows", None)
            print(f"  Train size: {size if size is not None else 'unknown'}")
    except Exception:
        print("  Dataset size info unavailable")

    # Commit the volume to persist the downloaded dataset
    volume.commit()

    return {
        "status": "success",
        "dataset_name": TRAINING_CONFIG["dataset_name"],
        "message": "Dataset downloaded and cached successfully",
    }


# ============================================================================
# Multi-GPU Training Function (generator - yields signals to local entrypoint)
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=86400,  # 24 hour timeout for long training runs
    memory=65536,  # 64 GB RAM
    is_generator=True,
)
def train_flux_2_klein():
    """Train Flux 2 Klein LoRA with multi-GPU support using accelerate launch.
    Yields signal dicts back to the local entrypoint so it can trigger real-time
    local downloads after each checkpoint, validation, and test stage.
    """
    import collections
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

    print(f"Starting Flux 2 Klein LoRA training with {NUM_GPUS} {GPU_TYPE} GPUs...")
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

    # Diagnostic: check for HF token availability in environment
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        try:
            print(f"HF token present (len={len(hf_token)}): {hf_token[:4]}... (masked)")
        except Exception:
            print("HF token present (masked)")
    else:
        print("HF token NOT found in environment. Ensure Modal secret 'huggingface-secret-2' defines HF_TOKEN or HUGGING_FACE_HUB_TOKEN.")

    verify_hf_access(TRAINING_CONFIG["pretrained_model_name_or_path"], hf_token)

    # Build the training command (with diagnostics if script cannot be found)
    try:
        cmd = build_training_command(config, NUM_GPUS)
    except FileNotFoundError as e:
        import os
        print(f"Error locating training script: {e}")
        try:
            print("/root contents:", os.listdir('/root'))
        except Exception:
            print("Could not list /root contents")
        if os.path.exists('/root/flux_2_klein_finetuning'):
            try:
                print("/root/flux_2_klein_finetuning contents:", os.listdir('/root/flux_2_klein_finetuning'))
            except Exception:
                print("Could not list /root/flux_2_klein_finetuning contents")
        raise

    print(f"Running command:\n{' '.join(cmd)}\n")
    print("=" * 80)

    try:
        # Run training with output streaming to detect checkpoint/validation signals
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_LAUNCH_BLOCKING"] = "1"
        process = subprocess.Popen(
            cmd,
            cwd="/root/flux_2_klein_finetuning",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        if process.stdout is None:
            raise RuntimeError("Training process did not provide stdout stream.")
        last_lines = collections.deque(maxlen=200)
        for line in process.stdout:
            # Print all output in real-time
            sys.stdout.write(line)
            sys.stdout.flush()
            last_lines.append(line.rstrip())

            # Detect checkpoint save signal - commit volume and yield to local
            if "###CHECKPOINT_SAVED###" in line:
                print("\n>>> [MODAL] Checkpoint saved detected. Committing volume...")
                try:
                    volume.commit()
                    print(">>> [MODAL] Volume committed successfully after checkpoint save.")
                except Exception as e:
                    print(f">>> [MODAL] Warning: Volume commit failed: {e}")
                yield {"signal": "checkpoint_saved", "detail": line.strip()}

            # Detect validation complete signal - commit volume and yield to local
            elif "###VALIDATION_COMPLETE###" in line:
                print("\n>>> [MODAL] Validation complete detected. Committing volume...")
                try:
                    volume.commit()
                    print(">>> [MODAL] Volume committed successfully after validation.")
                except Exception as e:
                    print(f">>> [MODAL] Warning: Volume commit failed: {e}")
                yield {"signal": "validation_complete", "detail": line.strip()}

            # Detect final test complete signal - commit volume and yield to local
            elif "###TEST_COMPLETE###" in line:
                print("\n>>> [MODAL] Final test complete detected. Committing volume...")
                try:
                    volume.commit()
                    print(">>> [MODAL] Volume committed successfully after final test.")
                except Exception as e:
                    print(f">>> [MODAL] Warning: Volume commit failed: {e}")
                yield {"signal": "test_complete", "detail": line.strip()}

        remaining = process.stdout.read()
        if remaining:
            sys.stdout.write(remaining)
            sys.stdout.flush()
            for tail_line in remaining.splitlines():
                last_lines.append(tail_line.rstrip())

        process.wait()
        if process.returncode != 0:
            if last_lines:
                print("\n----- TRAIN LOGS (tail) -----")
                print("\n".join(last_lines))
            else:
                diag = subprocess.run(cmd, text=True, capture_output=True)
                if diag.stdout:
                    print("\n----- TRAIN STDOUT (diag) -----")
                    print(diag.stdout[-4000:])
                if diag.stderr:
                    print("\n----- TRAIN STDERR (diag) -----")
                    print(diag.stderr[-4000:])
            raise subprocess.CalledProcessError(process.returncode, cmd)

    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        # Still commit volume to preserve any partial progress
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

    # Commit the volume to persist the trained model
    print("\nCommitting volume to persist trained model...")
    volume.commit()

    yield {
        "signal": "training_complete",
        "status": "success",
        "output_dir": TRAINING_CONFIG["output_dir"],
        "num_gpus": NUM_GPUS,
        "gpu_type": GPU_TYPE,
        "resumed_from": latest_ckpt,
        "message": f"Flux 2 Klein LoRA training completed on {NUM_GPUS} {GPU_TYPE} GPUs",
    }


# ============================================================================
# Single-GPU Training Function (generator - yields signals to local entrypoint)
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=86400,  # 24 hour timeout
    memory=65536,  # 64 GB RAM
    is_generator=True,
)
def train_flux_2_klein_single_gpu():
    """Train Flux 2 Klein LoRA on a single GPU with auto-resume.
    Yields signal dicts back to the local entrypoint so it can trigger real-time
    local downloads after each checkpoint, validation, and test stage.
    """
    import collections
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

    print("Starting Flux 2 Klein LoRA training on single GPU...")

    # Diagnostic: check for HF token availability in environment
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        try:
            print(f"HF token present (len={len(hf_token)}): {hf_token[:4]}... (masked)")
        except Exception:
            print("HF token present (masked)")
    else:
        print("HF token NOT found in environment. Ensure Modal secret 'huggingface-secret-2' defines HF_TOKEN or HUGGING_FACE_HUB_TOKEN.")

    verify_hf_access(TRAINING_CONFIG["pretrained_model_name_or_path"], hf_token)

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
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_LAUNCH_BLOCKING"] = "1"
        process = subprocess.Popen(
            cmd,
            cwd="/root/flux_2_klein_finetuning",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        if process.stdout is None:
            raise RuntimeError("Training process did not provide stdout stream.")
        last_lines = collections.deque(maxlen=200)
        for line in process.stdout:
            # Print all output in real-time
            sys.stdout.write(line)
            sys.stdout.flush()
            last_lines.append(line.rstrip())

            # Detect checkpoint save signal - commit volume and yield to local
            if "###CHECKPOINT_SAVED###" in line:
                print("\n>>> [MODAL] Checkpoint saved detected. Committing volume...")
                try:
                    volume.commit()
                    print(">>> [MODAL] Volume committed successfully after checkpoint save.")
                except Exception as e:
                    print(f">>> [MODAL] Warning: Volume commit failed: {e}")
                yield {"signal": "checkpoint_saved", "detail": line.strip()}

            # Detect validation complete signal - commit volume and yield to local
            elif "###VALIDATION_COMPLETE###" in line:
                print("\n>>> [MODAL] Validation complete detected. Committing volume...")
                try:
                    volume.commit()
                    print(">>> [MODAL] Volume committed successfully after validation.")
                except Exception as e:
                    print(f">>> [MODAL] Warning: Volume commit failed: {e}")
                yield {"signal": "validation_complete", "detail": line.strip()}

            # Detect final test complete signal - commit volume and yield to local
            elif "###TEST_COMPLETE###" in line:
                print("\n>>> [MODAL] Final test complete detected. Committing volume...")
                try:
                    volume.commit()
                    print(">>> [MODAL] Volume committed successfully after final test.")
                except Exception as e:
                    print(f">>> [MODAL] Warning: Volume commit failed: {e}")
                yield {"signal": "test_complete", "detail": line.strip()}

        remaining = process.stdout.read()
        if remaining:
            sys.stdout.write(remaining)
            sys.stdout.flush()
            for tail_line in remaining.splitlines():
                last_lines.append(tail_line.rstrip())

        process.wait()
        if process.returncode != 0:
            if last_lines:
                print("\n----- TRAIN LOGS (tail) -----")
                print("\n".join(last_lines))
            else:
                diag = subprocess.run(cmd, text=True, capture_output=True)
                if diag.stdout:
                    print("\n----- TRAIN STDOUT (diag) -----")
                    print(diag.stdout[-4000:])
                if diag.stderr:
                    print("\n----- TRAIN STDERR (diag) -----")
                    print(diag.stderr[-4000:])
            raise subprocess.CalledProcessError(process.returncode, cmd)

    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        # Still commit volume to preserve any partial progress
        print(">>> [MODAL] Committing volume to preserve partial progress...")
        try:
            volume.commit()
        except Exception:
            pass
        raise

    volume.commit()

    yield {
        "signal": "training_complete",
        "status": "success",
        "output_dir": TRAINING_CONFIG["output_dir"],
        "resumed_from": latest_ckpt,
        "message": "Flux 2 Klein LoRA training completed on single GPU",
    }


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=21600,
    memory=65536,
)
def run_flux_2_klein_validation_only(
    max_val_samples: int = 15,
    max_test_samples: int = 0,
    eval_viz_num: int = 10,
    original_only_no_masks: bool = False,
    original_only_split: str = "validation",
    original_only_max_samples: int = 15,
):
    """Run pre-training validation/test generation only (no optimizer steps)."""
    import os
    import subprocess

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    os.makedirs(TRAINING_CONFIG["output_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["logging_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["cache_dir"], exist_ok=True)

    script_path = "/root/flux_2_klein_finetuning/flux_2_klein_lora_finetuning.py"

    cmd = [
        "python",
        "-u",
        script_path,
        "--pretrained_model_name_or_path",
        str(TRAINING_CONFIG["pretrained_model_name_or_path"]),
        "--dataset_name",
        str(TRAINING_CONFIG["dataset_name"]),
        "--edited_image_column",
        str(TRAINING_CONFIG["edited_image_column"]),
        "--original_image_column",
        str(TRAINING_CONFIG["original_image_column"]),
        "--edit_instruction_column",
        str(TRAINING_CONFIG["edit_instruction_column"]),
        "--initial_edit_region_mask_column",
        str(TRAINING_CONFIG["initial_edit_region_mask_column"]),
        "--final_edit_region_mask_column",
        str(TRAINING_CONFIG["final_edit_region_mask_column"]),
        "--output_dir",
        str(TRAINING_CONFIG["output_dir"]),
        "--logging_dir",
        str(TRAINING_CONFIG["logging_dir"]),
        "--cache_dir",
        str(TRAINING_CONFIG["cache_dir"]),
        "--resolution",
        str(TRAINING_CONFIG["resolution"]),
        "--train_batch_size",
        str(TRAINING_CONFIG["train_batch_size"]),
        "--gradient_accumulation_steps",
        str(TRAINING_CONFIG["gradient_accumulation_steps"]),
        "--num_train_epochs",
        "0",
        "--learning_rate",
        str(TRAINING_CONFIG["learning_rate"]),
        "--lr_scheduler",
        str(TRAINING_CONFIG["lr_scheduler"]),
        "--lr_warmup_steps",
        str(TRAINING_CONFIG["lr_warmup_steps"]),
        "--optimizer",
        str(TRAINING_CONFIG["optimizer"]),
        "--rank",
        str(TRAINING_CONFIG["rank"]),
        "--lora_alpha",
        str(TRAINING_CONFIG["lora_alpha"]),
        "--lora_dropout",
        str(TRAINING_CONFIG["lora_dropout"]),
        "--checkpointing_epochs",
        str(TRAINING_CONFIG["checkpointing_epochs"]),
        "--checkpoints_total_limit",
        str(TRAINING_CONFIG["checkpoints_total_limit"]),
        "--validation_epochs",
        str(TRAINING_CONFIG["validation_epochs"]),
        "--seed",
        str(TRAINING_CONFIG["seed"]),
        "--max_sequence_length",
        str(TRAINING_CONFIG["max_sequence_length"]),
        "--guidance_scale",
        str(TRAINING_CONFIG["guidance_scale"]),
        "--weighting_scheme",
        str(TRAINING_CONFIG["weighting_scheme"]),
        "--dataloader_num_workers",
        str(TRAINING_CONFIG["dataloader_num_workers"]),
        "--mixed_precision",
        str(TRAINING_CONFIG["mixed_precision"]),
        "--report_to",
        str(TRAINING_CONFIG["report_to"]),
        "--repeats",
        str(TRAINING_CONFIG.get("repeats", 1)),
        "--t_rec_max",
        str(TRAINING_CONFIG["t_rec_max"]),
        "--p_rec_fraction",
        str(TRAINING_CONFIG["p_rec_fraction"]),
        "--lambda_static",
        str(TRAINING_CONFIG["lambda_static"]),
        "--max_val_samples",
        str(max_val_samples),
        "--max_test_samples",
        str(max_test_samples),
        "--eval_viz_num",
        str(eval_viz_num),
        "--val_batch_size",
        str(TRAINING_CONFIG["val_batch_size"]),
        "--test_batch_size",
        str(TRAINING_CONFIG["test_batch_size"]),
        "--allow_tf32",
    ]

    if TRAINING_CONFIG.get("dataset_config_name"):
        cmd.extend(["--dataset_config_name", str(TRAINING_CONFIG["dataset_config_name"])])
    if TRAINING_CONFIG.get("revision"):
        cmd.extend(["--revision", str(TRAINING_CONFIG["revision"])])
    if TRAINING_CONFIG.get("variant"):
        cmd.extend(["--variant", str(TRAINING_CONFIG["variant"])])
    if TRAINING_CONFIG.get("gradient_checkpointing"):
        cmd.append("--gradient_checkpointing")
    if TRAINING_CONFIG.get("use_8bit_adam"):
        cmd.append("--use_8bit_adam")
    if original_only_no_masks:
        cmd.append("--original_only_test_outputs")
        cmd.extend(["--original_only_split", str(original_only_split)])
        cmd.extend(["--original_only_max_samples", str(original_only_max_samples)])

    print("Running validation-only command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    volume.commit()

    val_dir = f"{TRAINING_CONFIG['output_dir']}/validation_visualization"
    test_dir = f"{TRAINING_CONFIG['output_dir']}/test_visualization"
    original_only_dir = f"{TRAINING_CONFIG['output_dir']}/original_only_visualization/{original_only_split}"
    return {
        "status": "success",
        "output_dir": TRAINING_CONFIG["output_dir"],
        "validation_visualization_dir": val_dir,
        "test_visualization_dir": test_dir,
        "original_only_visualization_dir": original_only_dir if original_only_no_masks else None,
        "original_only_no_masks": original_only_no_masks,
        "message": "Validation-only run complete",
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
    lora_path: str | None = None,
    prompt: str = "Move the cat to the right side of the image",
    input_image_url: str | None = None,
):
    """
    Run inference with the trained Flux 2 Klein LoRA model.
    """
    import os
    import sys

    # Add local directory to path
    sys.path.insert(0, "/root/flux_2_klein_finetuning")

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

    import importlib
    import torch
    from diffusers.utils.loading_utils import load_image

    lora_path = lora_path or TRAINING_CONFIG["output_dir"]

    print("Loading Flux 2 Klein pipeline...")
    diffusers = importlib.import_module("diffusers")
    Flux2KleinPipeline = getattr(diffusers, "Flux2KleinPipeline")
    pipeline = Flux2KleinPipeline.from_pretrained(
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
    If so, upload NEW checkpoints (not already on the volume) using
    'modal volume put' CLI — much more efficient than RPC for large files.
    The local models/ dir maps to /data/models/ on the volume.
    """
    import os
    import subprocess
    import sys
    import time

    local_dir = os.path.dirname(__file__)
    local_models_path = os.path.join(local_dir, LOCAL_MODELS_DIR)
    if not os.path.exists(local_models_path):
        print(f"  No local '{LOCAL_MODELS_DIR}/' directory found. Skipping upload.")
        return False

    # Determine the upload root — prefer models/<output_basename>/ if it exists
    # e.g. output_basename = "flux-2-klein-lora"
    output_basename = os.path.basename(TRAINING_CONFIG["output_dir"].rstrip("/"))
    preferred_root = os.path.join(local_models_path, output_basename)

    if os.path.isdir(preferred_root):
        upload_root = preferred_root
        # Remote path is relative to volume root (volume mounts at /data)
        remote_base = f"{VOLUME_MODELS_REMOTE_DIR}/{output_basename}"
    else:
        upload_root = local_models_path
        remote_base = VOLUME_MODELS_REMOTE_DIR

    # Discover local checkpoint directories
    local_checkpoint_dirs = sorted([
        d for d in os.listdir(upload_root)
        if d.startswith("checkpoint") and os.path.isdir(os.path.join(upload_root, d))
    ])

    if not local_checkpoint_dirs:
        print(f"  No checkpoint directories found in {upload_root}. Skipping upload.")
        return False

    print(f"  Found {len(local_checkpoint_dirs)} local checkpoint(s): {local_checkpoint_dirs}")

    # Query the volume for existing checkpoints to avoid redundant uploads
    print("  Querying Modal volume for existing checkpoints...")
    try:
        vol_info = get_volume_checkpoint_info.remote()
        existing_on_volume = set(vol_info.get("all_checkpoints", []))
        print(f"  Volume has {len(existing_on_volume)} checkpoint(s): "
              f"{sorted(existing_on_volume) if existing_on_volume else 'none'}")
    except Exception as e:
        print(f"  Warning: Could not query volume ({e}). Will upload all local checkpoints.")
        existing_on_volume = set()

    new_checkpoints = [c for c in local_checkpoint_dirs if c not in existing_on_volume]
    if not new_checkpoints:
        print("  All local checkpoints already exist on volume. Skipping upload.")
        return False

    print(f"  Uploading {len(new_checkpoints)} new checkpoint(s): {new_checkpoints}")

    uploaded = 0
    for ckpt_name in new_checkpoints:
        local_ckpt_path = os.path.join(upload_root, ckpt_name)
        remote_ckpt_path = f"{remote_base}/{ckpt_name}"

        print(f"    Uploading {ckpt_name} -> volume:{remote_ckpt_path} ...")
        t0 = time.time()
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m",
                    "modal", "volume", "put",
                    "flux2-klein-calvin-image-editing-volume",
                    local_ckpt_path,
                    remote_ckpt_path,
                ],
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min per checkpoint (they can be large)
            )
            elapsed = time.time() - t0
            if result.returncode == 0:
                print(f"    Uploaded {ckpt_name} OK ({elapsed:.1f}s)")
                uploaded += 1
            else:
                print(f"    Warning: Upload of {ckpt_name} failed (exit {result.returncode}, {elapsed:.1f}s)")
                if result.stderr:
                    print(f"    stderr: {result.stderr.strip()[:500]}")
        except subprocess.TimeoutExpired:
            print(f"    Warning: Upload of {ckpt_name} timed out (30 min)")
        except FileNotFoundError:
            print("    Error: 'modal' CLI not found. Install with: pip install modal")
            return False
        except Exception as e:
            print(f"    Warning: Upload of {ckpt_name} failed: {e}")

    print(f"  Upload complete. {uploaded}/{len(new_checkpoints)} checkpoint(s) uploaded.")
    return uploaded > 0


# ============================================================================
# Local Helper: Download models from Modal volume
# ============================================================================
def _download_models_from_volume(reason: str = ""):
    """
    Download the 'models' directory from Modal volume to the local machine.
    Uses 'modal volume get' CLI command for efficient transfer.

    Args:
        reason: Short tag for log messages (e.g. "After checkpoint").
    """
    import os
    import subprocess
    import sys
    import time

    local_dir = os.path.dirname(__file__)
    local_models_path = os.path.join(local_dir, LOCAL_MODELS_DIR)

    tag = f"[{reason}] " if reason else ""
    print(f"\n>>> {tag}Downloading models from Modal volume to {local_models_path}...")

    t0 = time.time()
    try:
        result = subprocess.run(
            [
                sys.executable, "-m",
                "modal",
                "volume",
                "get",
                "--force",
                "flux2-klein-calvin-image-editing-volume",
                VOLUME_MODELS_REMOTE_DIR,
                local_dir,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            print(f">>> {tag}Download complete ({elapsed:.1f}s).")
            if result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                if len(lines) > 5:
                    print(f"    ({len(lines)} files downloaded)")
                else:
                    print(result.stdout.strip())
        else:
            print(f">>> Warning: Download failed (exit code {result.returncode}, {elapsed:.1f}s)")
            if result.stderr:
                print(f"    stderr: {result.stderr.strip()[:500]}")

    except subprocess.TimeoutExpired:
        print(">>> Warning: Download timed out after 10 minutes")
    except FileNotFoundError:
        print(">>> Warning: 'modal' CLI not found. Install it with: pip install modal")
    except Exception as e:
        print(f">>> Warning: Download failed: {e}")


# ============================================================================
# Local Entrypoint
# ============================================================================
@app.local_entrypoint()
def main(
    action: str = "train",
    checkpoint: str = "latest",
    prompt: str = "Move the object to the right",
    image_url: str | None = None,
    single_gpu: bool = False,
    original_only_no_masks: bool = False,
    original_only_split: str = "validation",
    original_only_max_samples: int = 15,
):
    """
    Local entrypoint to run training, inference, or utilities.

    Args:
        action: One of "train", "inference", "download"
        checkpoint: Checkpoint path for resume (default: "latest")
        prompt: Edit prompt for inference
        image_url: Input image URL for inference
        single_gpu: Use single GPU instead of multi-GPU
    """
    print("=" * 60)
    print(f"Flux 2 Klein LoRA Finetuning - Action: {action}")
    print("=" * 60)

    if action == "train":
        # =============================================================
        # Step 1: Upload local checkpoints to volume (if any exist)
        # =============================================================
        print("\n[Step 1/2] Checking for local checkpoints to upload to Modal volume...")
        uploaded = _upload_local_checkpoints_to_volume()
        if uploaded:
            print("  Local checkpoints uploaded successfully.")
        else:
            print("  No new local checkpoints to upload.")

        # =============================================================
        # Step 2: Start training (streams signals for real-time local downloads)
        # =============================================================
        print("\n[Step 2/2] Starting training (with real-time local checkpointing)...")
        if single_gpu:
            print("  Mode: single-GPU training")
            signal_stream = train_flux_2_klein_single_gpu.remote_gen()
        else:
            print(f"  Mode: multi-GPU training on {NUM_GPUS} {GPU_TYPE} GPUs")
            signal_stream = train_flux_2_klein.remote_gen()

        download_count = 0
        for event in signal_stream:
            signal = event.get("signal", "unknown")

            if signal == "checkpoint_saved":
                download_count += 1
                print(f"\n>>> [LOCAL] Checkpoint saved on remote. Downloading models (#{download_count})...")
                _download_models_from_volume(reason="After checkpoint")
                print(f">>> [LOCAL] Download #{download_count} complete (after checkpoint).")

            elif signal == "validation_complete":
                download_count += 1
                print(f"\n>>> [LOCAL] Validation complete on remote. Downloading models (#{download_count})...")
                _download_models_from_volume(reason="After validation")
                print(f">>> [LOCAL] Download #{download_count} complete (after validation).")

            elif signal == "test_complete":
                download_count += 1
                print(f"\n>>> [LOCAL] Final test complete on remote. Downloading models (#{download_count})...")
                _download_models_from_volume(reason="After test")
                print(f">>> [LOCAL] Download #{download_count} complete (after final test).")

            elif signal == "training_complete":
                print("\n>>> Training Result:")
                print(f"  Status: {event.get('status')}")
                print(f"  Output: {event.get('output_dir')}")
                if event.get("resumed_from"):
                    print(f"  Resumed from: {event['resumed_from']}")
                print(f"  Message: {event.get('message')}")

                download_count += 1
                print(f"\n>>> [LOCAL] Final download of models (#{download_count})...")
                _download_models_from_volume(reason="Final")
                print(f">>> [LOCAL] Final download complete.")

            else:
                print(f"\n>>> [LOCAL] Received unknown signal: {signal}")

        print(f"\n>>> [LOCAL] Total downloads performed: {download_count}")
        result = {"status": "success", "message": "Training stream completed"}

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
        _download_models_from_volume(reason="Manual download")
        result = {"status": "success", "message": "Models downloaded"}

    elif action == "validate":
        if original_only_no_masks:
            print("Running original-only output test (prompt + original image, no masks, no metrics)...")
        else:
            print("Running validation-only pass (no training updates)...")
        result = run_flux_2_klein_validation_only.remote(
            original_only_no_masks=original_only_no_masks,
            original_only_split=original_only_split,
            original_only_max_samples=original_only_max_samples,
        )
        print("\nValidation Result:")
        print(f"  Status: {result.get('status')}")
        print(f"  Output: {result.get('output_dir')}")
        print(f"  Validation Viz: {result.get('validation_visualization_dir')}")
        print(f"  Test Viz: {result.get('test_visualization_dir')}")
        if result.get("original_only_visualization_dir"):
            print(f"  Original-only Viz: {result.get('original_only_visualization_dir')}")

        print("\nDownloading models (including validation/test visualization artifacts)...")
        _download_models_from_volume(reason="After validation-only run")
        result["downloaded_local_dir"] = os.path.join(os.path.dirname(__file__), LOCAL_MODELS_DIR)

    else:
        raise ValueError(
            f"Invalid action: {action}. "
            "Use 'train', 'validate', 'inference', 'download', 'download_models', or 'list_checkpoints'"
        )

    print("=" * 60)
    return result
