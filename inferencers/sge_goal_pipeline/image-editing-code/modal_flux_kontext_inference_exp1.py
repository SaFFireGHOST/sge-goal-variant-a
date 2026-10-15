"""
Modal script to run image editing inference on a local test data folder.

Workflow:
  1. Uploads the local test data folder and checkpoint to a temporary Modal volume.
  2. Runs inference on the Modal GPU: loads the base model + LoRA weights, iterates
     over every test-case sub-directory, and saves "edited_image.jpg" in each.
  3. Downloads the processed test data folder back to local, replacing the original.
  4. Deletes the temporary Modal volume so future runs start fresh.

Usage:
    conda activate lgie
    cd image-editing-code
    modal run modal_flux_kontext_inference.py \
        --data-dir ../dummy_test_data \
        --checkpoint-dir ../image-editing-checkpoint/checkpoint-epoch-18
"""

import modal
import os
import io
import argparse

# ============================================================================
# Modal App Setup
# ============================================================================
app = modal.App("flux-kontext-exp1-bridge-tel-imgedit")

VOLUME_NAME = "flux-kontext-exp1-bridge-tel-imgedit-vol"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# GPU configuration
GPU_TYPE = "L40S"
GPU_CONFIG = f"{GPU_TYPE}:1"

# CUDA Configuration
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# Paths on the Modal volume (mounted at /data)
REMOTE_DATA_DIR = "/data/test_data"
REMOTE_CHECKPOINT_DIR = "/data/checkpoint"

# ============================================================================
# Modal Image with all required dependencies
# ============================================================================
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision",
        "torchaudio",
    )
    .uv_pip_install(
        "accelerate",
        "diffusers",
        "transformers",
        "peft",
        "pillow",
        "numpy",
        "scipy",
        "huggingface-hub",
        "hf_transfer",
        "scikit-image",
        "tqdm",
        "packaging",
        "wheel",
        "sentencepiece",
    )
    .run_commands("pip install flash-attn --no-build-isolation")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": "/data/.cache",
        }
    )
    # Add the codebase (pipeline + eval utils)
    .add_local_dir(
        ".",
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

huggingface_secret = modal.Secret.from_name("huggingface-secret")


# ============================================================================
# Remote: Upload files to volume
# ============================================================================
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=86400,
)
def upload_files_to_volume(file_data: list[dict]):
    """
    Upload a batch of files to the Modal volume.
    file_data: list of {"remote_path": str, "content": bytes}
    """
    for item in file_data:
        remote_path = item["remote_path"]
        content = item["content"]
        os.makedirs(os.path.dirname(remote_path), exist_ok=True)
        with open(remote_path, "wb") as f:
            f.write(content)
    volume.commit()
    print(f"Uploaded {len(file_data)} files to volume.")
    return {"status": "success", "files": len(file_data)}


# ============================================================================
# Remote: Run inference on all test cases
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=86400,  # 4 hours
    memory=65536,   # 64 GB RAM
)
def run_inference():
    """
    Load the pipeline + LoRA weights, iterate over every test-case sub-directory
    inside REMOTE_DATA_DIR, and save 'edited_image.jpg' in each.
    """
    import sys
    sys.path.insert(0, "/root/flux_finetuning")

    import torch
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    # Import our custom pipeline and prompt builder
    from pipeline_flux_kontext_multi_imgs import FluxKontextPipeline
    from flux_kontext_eval_utils import build_enhanced_prompt

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    device = "cuda"
    dtype = torch.bfloat16

    # ------------------------------------------------------------------
    # 1. Load base pipeline
    # ------------------------------------------------------------------
    print("Loading base Flux Kontext pipeline...")
    pipeline = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=dtype,
    )
    pipeline = pipeline.to(device)

    # ------------------------------------------------------------------
    # 2. Load LoRA weights from checkpoint
    # ------------------------------------------------------------------
    lora_path = os.path.join(REMOTE_CHECKPOINT_DIR, "pytorch_lora_weights.safetensors")
    if os.path.exists(lora_path):
        print(f"Loading LoRA weights from {lora_path} ...")
        pipeline.load_lora_weights(REMOTE_CHECKPOINT_DIR)
        print("LoRA weights loaded successfully.")
    else:
        print(f"WARNING: No LoRA weights found at {lora_path}. Running with base model only.")

    pipeline.set_progress_bar_config(disable=False)

    # Reproducible generation
    seed = 42
    generator = torch.Generator(device=device).manual_seed(seed)

    # ------------------------------------------------------------------
    # 3. Discover test cases
    # ------------------------------------------------------------------
    test_case_dirs = sorted(
        [
            d
            for d in os.listdir(REMOTE_DATA_DIR)
            if os.path.isdir(os.path.join(REMOTE_DATA_DIR, d))
        ],
        key=lambda x: int(x) if x.isdigit() else x,
    )
    print(f"Found {len(test_case_dirs)} test case directories.")

    # ------------------------------------------------------------------
    # 4. Run inference for each test case
    # ------------------------------------------------------------------
    num_inference_steps = 20  # matches validation defaults

    for tc_name in tqdm(test_case_dirs, desc="Inference"):
        tc_dir = os.path.join(REMOTE_DATA_DIR, tc_name)

        # ----------------------------------------------------------
        # RESUME: Skip if edited_image.jpg already exists
        # ----------------------------------------------------------
        output_check = os.path.join(tc_dir, "edited_image.jpg")
        if os.path.isfile(output_check):
            print(f"  SKIP {tc_name}: edited_image.jpg already exists")
            continue

        # Helper: find a file by base name, trying common image extensions
        def find_image_file(directory, base_name):
            """Find base_name.{jpg,jpeg,png} in directory. Returns path or None."""
            for ext in (".jpg", ".jpeg", ".png"):
                p = os.path.join(directory, base_name + ext)
                if os.path.exists(p):
                    return p
            return None

        # Required files (extension-agnostic for images)
        original_path = find_image_file(tc_dir, "original_image")
        init_mask_path = find_image_file(tc_dir, "initial_edit_region_mask")
        final_mask_path = find_image_file(tc_dir, "final_edit_region_mask")
        instruction_path = os.path.join(tc_dir, "edit_instruction.txt")

        # Validate that all required inputs exist
        missing = []
        if original_path is None:
            missing.append("original_image.*")
        if init_mask_path is None:
            missing.append("initial_edit_region_mask.*")
        if final_mask_path is None:
            missing.append("final_edit_region_mask.*")
        if not os.path.exists(instruction_path):
            missing.append("edit_instruction.txt")
        if missing:
            print(f"  SKIPPING {tc_name}: missing {missing}")
            continue

        # Load inputs
        orig_image = Image.open(original_path).convert("RGB")
        init_mask = Image.open(init_mask_path).convert("RGB")
        final_mask = Image.open(final_mask_path).convert("RGB")

        with open(instruction_path, "r", encoding="utf-8") as f:
            # First line is the edit instruction (other lines may be metadata)
            edit_instruction = f.readline().strip()

        # Build enhanced prompt (same as validation code)
        prompt = build_enhanced_prompt(edit_instruction)

        # Determine output dimensions from original image
        orig_w, orig_h = orig_image.size

        # Run pipeline (mirrors the validation code in flux_kontext_eval_utils.py)
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt, prompt_2=None
            )
            prompt_embeds = prompt_embeds.to(device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)

        with torch.autocast("cuda"):
            out = pipeline(
                original_image=orig_image,
                initial_edit_region_mask=init_mask,
                final_edit_region_mask=final_mask,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=num_inference_steps,
                height=orig_h,
                width=orig_w,
                _auto_resize=False,
                max_area=orig_h * orig_w,
                generator=generator,
            )

        # Extract predicted image
        if hasattr(out, "images"):
            pred_img = out.images[0]
        elif isinstance(out, (list, tuple)):
            candidate = out[0]
            pred_img = candidate[0] if isinstance(candidate, (list, tuple)) else candidate
        else:
            pred_img = out

        # Resize to original dimensions if pipeline output differs
        if pred_img is not None and isinstance(pred_img, Image.Image):
            pred_w, pred_h = pred_img.size
            if (pred_w, pred_h) != (orig_w, orig_h):
                pred_img = pred_img.resize((orig_w, orig_h), resample=Image.BILINEAR)

        # Save
        output_path = os.path.join(tc_dir, "edited_image.jpg")
        if pred_img is not None:
            pred_img.save(output_path, "JPEG", quality=95)
            print(f"  Saved: {output_path}")
        else:
            print(f"  WARNING: Pipeline returned None for {tc_name}")

    # Commit volume so the caller can download the results
    volume.commit()
    print("All inference complete. Volume committed.")
    return {"status": "success", "test_cases": len(test_case_dirs)}


# ============================================================================
# Remote: List all files under the test data directory on the volume
# ============================================================================
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=86400,
)
def list_volume_files():
    """Return a list of all file paths (relative to REMOTE_DATA_DIR) on the volume."""
    results = []
    for root, dirs, files in os.walk(REMOTE_DATA_DIR):
        for fname in files:
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, REMOTE_DATA_DIR)
            results.append(rel)
    return results


# ============================================================================
# Remote: Download a single file from the volume
# ============================================================================
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=86400,
)
def download_file_from_volume(relative_path: str):
    """Read a single file from the volume and return its bytes."""
    full = os.path.join(REMOTE_DATA_DIR, relative_path)
    with open(full, "rb") as f:
        return f.read()


# ============================================================================
# Local entrypoint
# ============================================================================
@app.local_entrypoint()
def main(
    data_dir: str = "../dummy_test_data",
    checkpoint_dir: str = "../image-editing-checkpoint/checkpoint-epoch-18",
    resume: bool = False,
):
    """
    Local entry point.

    Parameters
    ----------
    data_dir : str
        Path to the local test data folder (each sub-folder is a test case).
    checkpoint_dir : str
        Path to the local checkpoint directory containing pytorch_lora_weights.safetensors.
    resume : bool
        If True, skip uploading data/checkpoint (already on volume) and resume inference.
    """
    import shutil
    import time

    data_dir = os.path.abspath(data_dir)
    checkpoint_dir = os.path.abspath(checkpoint_dir)

    print(f"Data directory   : {data_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Resume mode      : {resume}")

    # ------------------------------------------------------------------
    # 1. Collect all files to upload (test data + checkpoint)
    # ------------------------------------------------------------------
    if resume:
        print("\n[1/5] SKIPPED (resume mode — data already on volume)")
        print("\n[2/5] SKIPPED (resume mode — checkpoint already on volume)")
    else:
        print("\n[1/5] Collecting files for upload ...")
        upload_batch: list[dict] = []

        # Test data
        for root, dirs, files in os.walk(data_dir):
            for fname in files:
                local_path = os.path.join(root, fname)
                rel = os.path.relpath(local_path, data_dir)
                remote_path = os.path.join(REMOTE_DATA_DIR, rel)
                with open(local_path, "rb") as f:
                    content = f.read()
                upload_batch.append({"remote_path": remote_path, "content": content})

        # Checkpoint
        for root, dirs, files in os.walk(checkpoint_dir):
            for fname in files:
                local_path = os.path.join(root, fname)
                rel = os.path.relpath(local_path, checkpoint_dir)
                remote_path = os.path.join(REMOTE_CHECKPOINT_DIR, rel)
                with open(local_path, "rb") as f:
                    content = f.read()
                upload_batch.append({"remote_path": remote_path, "content": content})

        total_bytes = sum(len(item["content"]) for item in upload_batch)
        print(f"  {len(upload_batch)} files, {total_bytes / 1e6:.1f} MB total")

        # ------------------------------------------------------------------
        # 2. Upload in batches (Modal has message-size limits ~50 MB)
        # ------------------------------------------------------------------
        print("\n[2/5] Uploading files to Modal volume ...")
        BATCH_SIZE_BYTES = 40 * 1024 * 1024  # 40 MB per batch
        batch: list[dict] = []
        batch_bytes = 0

        for item in upload_batch:
            item_size = len(item["content"])
            if batch and (batch_bytes + item_size > BATCH_SIZE_BYTES):
                upload_files_to_volume.remote(batch)
                batch = []
                batch_bytes = 0
            batch.append(item)
            batch_bytes += item_size

        if batch:
            upload_files_to_volume.remote(batch)

        print("  Upload complete.")

    # ------------------------------------------------------------------
    # 3. Run inference remotely
    # ------------------------------------------------------------------
    print("\n[3/5] Running inference on Modal GPU ...")
    t0 = time.time()
    result = run_inference.remote()
    elapsed = time.time() - t0
    print(f"  Inference finished in {elapsed:.1f}s — {result}")

    # ------------------------------------------------------------------
    # 4. Download processed test data and replace local copy
    # ------------------------------------------------------------------
    print("\n[4/5] Downloading processed test data from Modal volume ...")

    # Get list of all files on the volume
    remote_files = list_volume_files.remote()
    print(f"  {len(remote_files)} files on volume")

    # Download each file
    # We only replace the data_dir; checkpoint files are not modified.
    # Remove existing local data dir and recreate
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    for rel_path in remote_files:
        content = download_file_from_volume.remote(rel_path)
        local_path = os.path.join(data_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(content)

    print(f"  Downloaded and replaced {data_dir}")

    # ------------------------------------------------------------------
    # 5. Delete the temporary Modal volume
    # ------------------------------------------------------------------
    print("\n[5/5] Deleting temporary Modal volume ...")
    try:
        modal.Volume.delete(VOLUME_NAME)
        print(f"  Volume '{VOLUME_NAME}' deleted.")
    except Exception as e:
        print(f"  Warning: Could not delete volume: {e}")
        print(f"  You can delete it manually: modal volume rm {VOLUME_NAME}")

    print("\nDone! Edited images saved to each test-case sub-directory as 'edited_image.jpg'.")
