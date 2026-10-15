"""
Modal script to run Final Edit Region (FER) Mask Prediction inference on cloud GPUs.

Uses Modal's serverless infrastructure to run the Flux Kontext pipeline
with trained LoRA weights for predicting final edit region masks.

The pipeline:
1. Uploads test data (original images + initial edit region masks + edit instructions)
   and LoRA checkpoint from a local directory to a Modal volume.
2. Loads the base Flux Kontext model + LoRA adapter on a cloud GPU.
3. Runs inference on each test case, saving final_edit_region_mask.jpg.
4. Downloads the predicted masks back to the local directory.

Usage:
    # Run prediction on the default dummy_test_data directory
    modal run modal_fer_mask_inference.py

    # Run with a custom data directory
    modal run modal_fer_mask_inference.py --data-dir /path/to/test_data

    # Run with a custom checkpoint directory
    modal run modal_fer_mask_inference.py --checkpoint-dir /path/to/checkpoint
"""

import modal
import os

# ============================================================================
# GPU Configuration
# ============================================================================
GPU_TYPE = "L40S"  # A100 (80 GB) for Flux Kontext + LoRA inference
GPU_CONFIG = GPU_TYPE

# ============================================================================
# Inference Configuration — matches training hyperparameters
# ============================================================================
INFERENCE_CONFIG = {
    "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-Kontext-dev",
    "guidance_scale": 1.0,          # Must match training config
    "num_inference_steps": 28,      # Default denoising steps
    "resolution": 256,              # Must match training resolution
    "seed": 0,
    "max_sequence_length": 512,
}

# The enhanced prompt template appended to every edit instruction during
# training.  We must replicate it at inference time for consistency.
DEFAULT_INSTRUCTION = """TASK: Edit Region Mask Prediction for Robotic Manipulation

You are provided with the following inputs:
1. ORIGINAL IMAGE: A scene containing objects, including a robotic arm and dynamic objects that can be manipulated.
2. INITIAL EDIT REGION MASK: A binary mask (white=1, black=0) showing the current locations of dynamic objects in the original image. White pixels indicate regions occupied by objects that will be manipulated/moved.
3. EDIT INSTRUCTION (above): A text description of the manipulation/edit to perform on the dynamic objects.

YOUR TASK: Generate a FINAL EDIT REGION MASK - a binary mask (white=1, black=0) predicting where the dynamic objects should appear AFTER the edit instruction is executed.
- White pixels (1): Regions where dynamic objects will be located after manipulation
- Black pixels (0): Background regions that remain unchanged

IMPORTANT:
- Focus ONLY on predicting object locations, not colors or textures
- The mask should be consistent across all RGB channels
- Use the initial mask to understand which objects are dynamic
- Use the edit instruction to determine the target locations
- Output should be a clean binary mask showing final object positions"""

# ============================================================================
# Modal App Setup
# ============================================================================
app = modal.App("flux-kontext-exp1-bridge-tel-fer")

volume = modal.Volume.from_name(
    "flux-kontext-exp1-bridge-tel-fer-vol", create_if_missing=True
)

# CUDA Configuration
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# ============================================================================
# Modal Image — all dependencies for Flux Kontext inference
# ============================================================================
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    # PyTorch
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision",
        "torchaudio",
    )
    # Inference dependencies
    .uv_pip_install(
        "accelerate",
        "diffusers",
        "transformers",
        "peft",
        "pillow",
        "numpy",
        "huggingface-hub",
        "hf_transfer",
        "sentencepiece",
        "tqdm",
        "packaging",
        "scikit-image",
        "scipy",
    )
    # Install flash-attn for faster inference
    .run_commands("pip install wheel && pip install flash-attn --no-build-isolation")
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_HOME": "/data/.cache",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    # Add the fer-mask-prediction-code directory
    .add_local_dir(
        ".",
        remote_path="/root/fer_mask_prediction",
        ignore=[
            "*.git*", "__pycache__", "*.pyc", ".DS_Store",
            "*.egg-info", "dummy_test_data", "models",
        ],
    )
)

# HuggingFace secret for model download
huggingface_secret = modal.Secret.from_name("huggingface-secret")


# ============================================================================
# Upload Test Data to Volume
# ============================================================================
@app.function(image=image, volumes={"/data": volume}, timeout=86400)
def upload_test_data(file_map: dict):
    """Upload test data files to the Modal volume."""
    total = len(file_map)
    print(f"Uploading {total} test data files to volume...")
    for remote_path, file_bytes in file_map.items():
        os.makedirs(os.path.dirname(remote_path), exist_ok=True)
        with open(remote_path, "wb") as f:
            f.write(file_bytes)
    volume.commit()
    print(f"Uploaded {total} files successfully.")
    return {"status": "success", "files_uploaded": total}


# ============================================================================
# Upload Checkpoint to Volume
# ============================================================================
@app.function(image=image, volumes={"/data": volume}, timeout=86400)
def upload_checkpoint(file_map: dict):
    """Upload LoRA checkpoint files to the Modal volume."""
    total = len(file_map)
    print(f"Uploading {total} checkpoint files to volume...")
    for remote_path, file_bytes in file_map.items():
        os.makedirs(os.path.dirname(remote_path), exist_ok=True)
        with open(remote_path, "wb") as f:
            f.write(file_bytes)
    volume.commit()
    print(f"Uploaded {total} checkpoint files successfully.")
    return {"status": "success", "files_uploaded": total}


# ============================================================================
# Run FER Mask Prediction on GPU
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=86400,
    memory=32768,  # 32 GB system RAM
)
def predict_final_masks(
    test_data_dir: str = "/data/test_data",
    checkpoint_dir: str = "/data/checkpoint",
):
    """
    Run the Flux Kontext FER mask prediction pipeline on a cloud GPU.

    For each test case directory containing:
      - original_image.{jpg,png}
      - initial_edit_region_mask.jpg
      - edit_instruction.txt

    Produces:
      - final_edit_region_mask.jpg
    """
    import sys
    sys.path.insert(0, "/root/fer_mask_prediction")

    import torch
    import numpy as np
    from PIL import Image
    from pipeline_flux_kontext_multi_imgs import FluxKontextPipeline

    volume.reload()

    # ----------------------------------------------------------------
    # Validate test data and checkpoint paths
    # ----------------------------------------------------------------
    if not os.path.exists(test_data_dir):
        raise FileNotFoundError(f"Test data directory not found: {test_data_dir}")

    lora_weights_path = os.path.join(checkpoint_dir, "pytorch_lora_weights.safetensors")
    if not os.path.isfile(lora_weights_path):
        raise FileNotFoundError(
            f"LoRA weights file not found: {lora_weights_path}\n"
            f"Contents of checkpoint_dir: {os.listdir(checkpoint_dir) if os.path.isdir(checkpoint_dir) else 'DIR NOT FOUND'}"
        )

    # ----------------------------------------------------------------
    # Load base Flux Kontext pipeline
    # ----------------------------------------------------------------
    cfg = INFERENCE_CONFIG
    print(f"Loading base model: {cfg['pretrained_model_name_or_path']}")
    weight_dtype = torch.bfloat16

    pipeline = FluxKontextPipeline.from_pretrained(
        cfg["pretrained_model_name_or_path"],
        torch_dtype=weight_dtype,
    )

    # ----------------------------------------------------------------
    # Load LoRA adapter weights
    # ----------------------------------------------------------------
    print(f"Loading LoRA weights from: {lora_weights_path}")
    pipeline.load_lora_weights(
        checkpoint_dir,
        weight_name="pytorch_lora_weights.safetensors",
    )
    print("LoRA weights loaded successfully.")

    # Move pipeline to GPU
    pipeline = pipeline.to("cuda", dtype=weight_dtype)
    try:
        pipeline.set_progress_bar_config(disable=True)
    except Exception:
        pass

    # Generator for reproducibility
    generator = torch.Generator(device="cuda").manual_seed(cfg["seed"])

    # ----------------------------------------------------------------
    # Discover test cases
    # ----------------------------------------------------------------
    test_cases = sorted([
        d for d in os.listdir(test_data_dir)
        if os.path.isdir(os.path.join(test_data_dir, d)) and not d.startswith(".")
    ])

    print(f"\nFound {len(test_cases)} test cases: {test_cases}")
    results = {}

    for case_id in test_cases:
        case_dir = os.path.join(test_data_dir, case_id)

        # ----------------------------------------------------------
        # RESUME: Skip if final_edit_region_mask already exists
        # ----------------------------------------------------------
        output_path_check = os.path.join(case_dir, "final_edit_region_mask.jpg")
        if os.path.isfile(output_path_check):
            print(f"  SKIP {case_id}: final_edit_region_mask.jpg already exists")
            results[case_id] = {"status": "skipped_resume", "reason": "already processed"}
            continue

        print(f"\n{'='*50}")
        print(f"Processing test case: {case_id}")
        print(f"{'='*50}")

        # ----------------------------------------------------------
        # Load inputs
        # ----------------------------------------------------------
        # 1. Original image
        orig_path = None
        for ext in ["jpg", "jpeg", "png"]:
            candidate = os.path.join(case_dir, f"original_image.{ext}")
            if os.path.isfile(candidate):
                orig_path = candidate
                break
        if orig_path is None:
            print(f"  WARNING: No original_image found in {case_dir}, skipping.")
            results[case_id] = {"status": "skipped", "reason": "no original_image"}
            continue

        orig_image = Image.open(orig_path).convert("RGB")
        print(f"  Original image: {orig_path} ({orig_image.size})")

        # 2. Initial edit region mask
        mask_path = os.path.join(case_dir, "initial_edit_region_mask.jpg")
        if not os.path.isfile(mask_path):
            # Try other extensions
            for ext in ["png", "jpeg"]:
                alt = os.path.join(case_dir, f"initial_edit_region_mask.{ext}")
                if os.path.isfile(alt):
                    mask_path = alt
                    break
        if not os.path.isfile(mask_path):
            print(f"  WARNING: No initial_edit_region_mask found in {case_dir}, skipping.")
            results[case_id] = {"status": "skipped", "reason": "no initial_edit_region_mask"}
            continue

        init_mask = Image.open(mask_path).convert("RGB")
        print(f"  Initial mask: {mask_path} ({init_mask.size})")

        # 3. Edit instruction
        instr_path = os.path.join(case_dir, "edit_instruction.txt")
        if not os.path.isfile(instr_path):
            print(f"  WARNING: No edit_instruction.txt found in {case_dir}, skipping.")
            results[case_id] = {"status": "skipped", "reason": "no edit_instruction"}
            continue

        with open(instr_path, "r", encoding="utf-8") as f:
            edit_instruction = f.read().strip()
        print(f"  Edit instruction: {edit_instruction}")

        # ----------------------------------------------------------
        # Build prompt (same as training: edit_instruction + DEFAULT_INSTRUCTION)
        # ----------------------------------------------------------
        prompt = edit_instruction + "\n\n" + DEFAULT_INSTRUCTION.strip()

        # ----------------------------------------------------------
        # Determine target dimensions
        # Use original image dimensions but ensure they're compatible
        # with the pipeline (divisible by vae_scale_factor * 2 = 32)
        # ----------------------------------------------------------
        target_width, target_height = orig_image.size
        multiple_of = 32  # vae_scale_factor(8) * 2(packing)
        target_width = (target_width // multiple_of) * multiple_of
        target_height = (target_height // multiple_of) * multiple_of

        # Ensure we don't exceed available GPU memory — cap area
        max_area = target_width * target_height
        # For very large images, cap at 512x512 area to be safe
        cap_area = cfg["resolution"] ** 2
        if max_area > cap_area:
            # Scale down to fit within cap_area while preserving aspect ratio
            aspect = target_width / target_height
            target_width = int((cap_area * aspect) ** 0.5)
            target_height = int((cap_area / aspect) ** 0.5)
            target_width = (target_width // multiple_of) * multiple_of
            target_height = (target_height // multiple_of) * multiple_of
            max_area = target_width * target_height

        print(f"  Target dimensions: {target_width}x{target_height}")

        # ----------------------------------------------------------
        # Encode prompt
        # ----------------------------------------------------------
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt, prompt_2=None
            )
            prompt_embeds = prompt_embeds.to("cuda")
            pooled_prompt_embeds = pooled_prompt_embeds.to("cuda")

        # ----------------------------------------------------------
        # Run pipeline inference
        # ----------------------------------------------------------
        print("  Running pipeline inference...")
        with torch.autocast("cuda"):
            out = pipeline(
                original_image=orig_image,
                initial_edit_region_mask=init_mask,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                generator=generator,
                height=target_height,
                width=target_width,
                guidance_scale=cfg["guidance_scale"],
                num_inference_steps=cfg["num_inference_steps"],
                max_sequence_length=cfg["max_sequence_length"],
                _auto_resize=False,
                max_area=max_area,
            )

        # Extract predicted image
        if hasattr(out, "images"):
            pred_img = out.images[0]
        elif isinstance(out, (list, tuple)):
            candidate = out[0]
            pred_img = candidate[0] if isinstance(candidate, (list, tuple)) else candidate
        else:
            pred_img = out

        if pred_img is None:
            print(f"  ERROR: Pipeline returned None for case {case_id}")
            results[case_id] = {"status": "error", "reason": "pipeline returned None"}
            continue

        # Convert to PIL if needed
        if not isinstance(pred_img, Image.Image):
            if isinstance(pred_img, np.ndarray):
                pred_img = Image.fromarray(pred_img.astype(np.uint8)).convert("RGB")
            else:
                print(f"  ERROR: Unexpected output type: {type(pred_img)}")
                results[case_id] = {"status": "error", "reason": f"unexpected type {type(pred_img)}"}
                continue

        # Resize prediction back to original image dimensions if needed
        orig_w, orig_h = Image.open(orig_path).size
        if pred_img.size != (orig_w, orig_h):
            pred_img = pred_img.resize((orig_w, orig_h), resample=Image.BILINEAR)
            print(f"  Resized prediction to original dimensions: {orig_w}x{orig_h}")

        # ----------------------------------------------------------
        # Save final edit region mask
        # ----------------------------------------------------------
        output_path = os.path.join(case_dir, "final_edit_region_mask.jpg")
        pred_img.save(output_path, quality=95)
        print(f"  Saved: {output_path}")

        results[case_id] = {
            "status": "success",
            "output_path": output_path,
            "output_size": pred_img.size,
        }

    # Commit volume with results
    volume.commit()
    print(f"\n{'='*50}")
    print(f"Inference complete. Processed {len(results)} test cases.")
    print(f"{'='*50}")

    return {"status": "success", "results": results}


# ============================================================================
# Download Entire Test Data Folder from Volume
# ============================================================================
@app.function(image=image, volumes={"/data": volume}, timeout=86400)
def download_test_data_folder(test_data_dir: str = "/data/test_data"):
    """Download the entire test data folder from the volume (including predictions)."""
    volume.reload()

    file_map = {}
    if not os.path.exists(test_data_dir):
        return {"status": "empty", "files": {}}

    for root, dirs, files in os.walk(test_data_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            if fname.startswith("."):
                continue
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, test_data_dir)
            with open(full_path, "rb") as f:
                file_map[rel_path] = f.read()

    print(f"Found {len(file_map)} total files in test data folder to download.")
    return {"status": "success", "files": file_map}


# ============================================================================
# Local Entrypoint
# ============================================================================
@app.local_entrypoint()
def main(
    data_dir: str = None,
    checkpoint_dir: str = None,
    resume: bool = False,
):
    """
    Run the full FER mask prediction inference pipeline:
      1. Upload entire test data folder to Modal volume.
      2. Upload LoRA checkpoint to Modal volume.
      3. Run Flux Kontext + LoRA inference on a cloud GPU (all test cases).
      4. Download the entire modified test data folder and replace local copy.

    Args:
        data_dir:       Path to the local test-data directory.
                        Defaults to ``<workspace>/dummy_test_data``.
        checkpoint_dir: Path to the local LoRA checkpoint directory.
                        Defaults to ``<workspace>/fer-mask-prediction-checkpoint/checkpoint-epoch-27``.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(script_dir)

    if data_dir is None:
        data_dir = os.path.join(workspace_root, "dummy_test_data")
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(
            workspace_root, "fer-mask-prediction-checkpoint", "checkpoint-epoch-27"
        )

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Test data directory not found: {data_dir}")
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    remote_data_dir = "/data/test_data"
    remote_checkpoint_dir = "/data/checkpoint"

    print("=" * 70)
    print("  FER Mask Prediction Inference Pipeline")
    print("  (Flux Kontext + LoRA)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Upload test data (skip if resuming)
    # ------------------------------------------------------------------
    if resume:
        print(f"\n--- Step 1: SKIPPED (resume mode — data already on volume) ---")
    else:
        print(f"\n--- Step 1: Upload entire test data folder from {data_dir} ---")
        file_map = {}
        for root, dirs, files in os.walk(data_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in files:
                if fname.startswith("."):
                    continue
                local_path = os.path.join(root, fname)
                rel_path = os.path.relpath(local_path, data_dir)
                remote_path = os.path.join(remote_data_dir, rel_path)
                with open(local_path, "rb") as f:
                    file_map[remote_path] = f.read()

        print(f"  Collected {len(file_map)} test data files to upload.")
        upload_test_data.remote(file_map)
        print("  Test data uploaded.")

    # ------------------------------------------------------------------
    # Step 2: Upload LoRA checkpoint (skip if resuming)
    # ------------------------------------------------------------------
    if resume:
        print(f"\n--- Step 2: SKIPPED (resume mode — checkpoint already on volume) ---")
    else:
        print(f"\n--- Step 2: Upload LoRA checkpoint from {checkpoint_dir} ---")
        ckpt_map = {}
        for fname in os.listdir(checkpoint_dir):
            fpath = os.path.join(checkpoint_dir, fname)
            if os.path.isfile(fpath) and not fname.startswith("."):
                # Only upload the LoRA weights file (skip optimizer, scheduler, etc.)
                if fname in ("pytorch_lora_weights.safetensors", "training_metadata.json"):
                    remote_path = os.path.join(remote_checkpoint_dir, fname)
                    with open(fpath, "rb") as f:
                        ckpt_map[remote_path] = f.read()

        print(f"  Collected {len(ckpt_map)} checkpoint files to upload.")
        upload_checkpoint.remote(ckpt_map)
        print("  Checkpoint uploaded.")

    # ------------------------------------------------------------------
    # Step 3: Run inference
    # ------------------------------------------------------------------
    print(f"\n--- Step 3: Run FER mask inference on Modal ({GPU_TYPE}) ---")
    result = predict_final_masks.remote(remote_data_dir, remote_checkpoint_dir)
    print(f"  Prediction status: {result['status']}")
    if "results" in result:
        for case_id, info in result["results"].items():
            status = info.get("status", "unknown")
            if status == "success":
                print(f"    Case {case_id}: SUCCESS — output size {info.get('output_size')}")
            else:
                print(f"    Case {case_id}: {status} — {info.get('reason', '')}")

    # ------------------------------------------------------------------
    # Step 4: Download entire test data folder and replace local copy
    # ------------------------------------------------------------------
    print(f"\n--- Step 4: Download entire test data folder from volume ---")
    dl_result = download_test_data_folder.remote(remote_data_dir)

    saved_count = 0
    if dl_result["status"] == "success" and dl_result["files"]:
        # Remove existing local test data folder and recreate it
        import shutil
        if os.path.isdir(data_dir):
            shutil.rmtree(data_dir)
            print(f"  Removed existing local folder: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)

        for rel_path, file_bytes in dl_result["files"].items():
            local_path = os.path.join(data_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(file_bytes)
            saved_count += 1
        print(f"  Wrote {saved_count} files to: {data_dir}")
    else:
        print("  No files found in remote test data folder.")

    # ------------------------------------------------------------------
    # Step 5: Delete Modal volume to ensure future runs start fresh
    # ------------------------------------------------------------------
    print(f"\n--- Step 5: Delete Modal volume ---")
    try:
        modal.Volume.delete("flux-kontext-exp1-bridge-tel-fer-vol")
        print("  Volume 'fer-mask-prediction-inference-volume' deleted successfully.")
    except Exception as e:
        print(f"  Warning: Could not delete volume: {e}")

    print(f"\n{'='*70}")
    print(f"  Pipeline complete!")
    print(f"  Local test data folder replaced with {saved_count} files at: {data_dir}")
    print(f"{'='*70}")
