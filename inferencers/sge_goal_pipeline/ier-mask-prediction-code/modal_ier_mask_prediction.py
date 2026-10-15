"""
Modal script to run Initial Edit Region (IER) Mask Prediction on cloud GPUs.

Uses Modal's serverless infrastructure to run the V2 multi-stage pipeline
(Florence-2 + Grounding DINO + BLIP-2 + SAM) for predicting initial edit
region masks in robotic manipulation images.

The pipeline (single Modal app run):
1. Uploads the entire test data folder to a Modal volume.
2. Runs multi-stage inference on ALL test cases in one GPU container.
3. Masks are written alongside the original data on the volume.
4. Downloads the complete test data folder (with masks) and replaces the
   local copy.

Usage:
    # Run prediction on the default dummy_test_data directory
    modal run modal_ier_mask_prediction.py

    # Run with a custom data directory
    modal run modal_ier_mask_prediction.py --data-dir /path/to/test_data
"""

import json
import modal
import os
import shutil

# ============================================================================
# GPU Configuration
# ============================================================================
GPU_TYPE = "A10G"       # A10G (24 GB) is sufficient for Grounding DINO + SAM inference
GPU_CONFIG = GPU_TYPE

# ============================================================================
# Modal App Setup
# ============================================================================
app = modal.App("ier-mask-prediction")

volume = modal.Volume.from_name(
    "ier-mask-prediction-volume", create_if_missing=True
)

# Volume path where test data lives
REMOTE_DATA_DIR = "/data/test_data"

# CUDA Configuration
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# ============================================================================
# Modal Image — all dependencies for V2 multi-stage pipeline
# ============================================================================
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    # PyTorch
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision",
    )
    # V2 pipeline dependencies:
    #   transformers  — Florence-2, Grounding DINO, BLIP-2, SAM
    #   einops, timm  — required by Florence-2 custom code
    #   accelerate    — required by BLIP-2 for float16 loading
    .uv_pip_install(
        "transformers==4.46.3",
        "pillow",
        "numpy",
        "scipy",
        "huggingface-hub",
        "hf_transfer",
        "tqdm",
        "einops",
        "timm",
        "accelerate",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_HOME": "/data/.cache",
    })
    # Add prediction code to image
    .add_local_dir(
        ".",
        remote_path="/root/ier_mask_prediction",
        ignore=[
            "*.git*", "__pycache__", "*.pyc", ".DS_Store",
            "*.egg-info", "dummy_test_data", "models",
        ],
    )
)


# ============================================================================
# Upload entire test data folder to the volume
# ============================================================================
@app.function(image=image, volumes={"/data": volume}, timeout=1800)
def upload_test_data(file_map: dict):
    """
    Upload test data files to the Modal volume.

    Clears existing test data on the volume first so we start fresh,
    then writes every file in ``file_map`` (remote_path → bytes).
    """
    import shutil as _shutil

    # Wipe previous test data so stale masks don't persist
    if os.path.exists(REMOTE_DATA_DIR):
        _shutil.rmtree(REMOTE_DATA_DIR)

    total = len(file_map)
    print(f"Uploading {total} files to volume ({REMOTE_DATA_DIR}) ...")
    for remote_path, file_bytes in file_map.items():
        os.makedirs(os.path.dirname(remote_path), exist_ok=True)
        with open(remote_path, "wb") as f:
            f.write(file_bytes)
    volume.commit()
    print(f"Uploaded {total} files successfully.")
    return {"status": "success", "files_uploaded": total}


# ============================================================================
# Run Mask Prediction on GPU  (processes ALL test cases in one call)
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/data": volume},
    timeout=7200,     # 2 hours — generous for large datasets
    memory=8192,      # 8 GB system RAM
)
def predict_masks():
    """
    Run the V2 multi-stage IER mask prediction pipeline on a cloud GPU.

    Pipeline:
        Florence-2 + Grounding DINO -> NMS -> BLIP-2 VQA verification -> SAM

    Reads every test-case subdirectory under ``REMOTE_DATA_DIR`` on the
    volume, predicts masks, writes them back alongside the originals,
    and commits the volume.
    """
    import sys
    import time
    sys.path.insert(0, "/root/ier_mask_prediction")

    volume.reload()

    from predict_initial_edit_region_mask import InitialEditRegionMaskPredictor

    print(f"Starting IER mask prediction (V2) on {GPU_CONFIG} GPU ...")
    print(f"Test data directory: {REMOTE_DATA_DIR}\n")

    predictor = InitialEditRegionMaskPredictor(
        device="cuda",
    )

    inference_start = time.perf_counter()
    results = predictor.process_test_cases(REMOTE_DATA_DIR)
    inference_s = time.perf_counter() - inference_start

    volume.commit()
    print("\nVolume committed with predicted masks.")

    return {"status": "success", "results": results, "inference_s": inference_s}


# ============================================================================
# Download the complete test data folder (originals + masks) from the volume
# ============================================================================
@app.function(image=image, volumes={"/data": volume}, timeout=1800)
def download_test_data():
    """
    Walk the entire ``REMOTE_DATA_DIR`` on the volume and return every file
    as a dict mapping relative paths to bytes.
    """
    volume.reload()

    file_map: dict[str, bytes] = {}
    if not os.path.exists(REMOTE_DATA_DIR):
        print("WARNING: Remote test data directory does not exist.")
        return {"status": "empty", "files": {}}

    for root, _dirs, files in os.walk(REMOTE_DATA_DIR):
        for fname in files:
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, REMOTE_DATA_DIR)
            with open(full_path, "rb") as f:
                file_map[rel_path] = f.read()

    print(f"Collected {len(file_map)} files for download.")
    return {"status": "success", "files": file_map}


# ============================================================================
# Local Entrypoint — single `modal run` invocation does everything
# ============================================================================
@app.local_entrypoint()
def main(
    data_dir: str = None,
):
    """
    Run the full IER mask prediction pipeline in one shot:

      1. Upload the entire local test-data folder to a Modal volume.
      2. Run Grounded SAM on ALL test cases in a single GPU container.
      3. Download the complete test-data folder (now including masks)
         from the volume and replace the local copy with it.

    Args:
        data_dir: Path to the local test-data directory.  Defaults to
                  ``<workspace>/dummy_test_data``.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(script_dir)

    if data_dir is None:
        data_dir = os.path.join(workspace_root, "dummy_test_data")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Test data directory not found: {data_dir}")

    print("=" * 60)
    print("IER Mask Prediction Pipeline  (V2: Florence-2 + GDINO + BLIP-2 + SAM)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Upload entire test data folder to the volume
    # ------------------------------------------------------------------
    print(f"\n--- Step 1: Upload test data from {data_dir} ---")
    file_map: dict[str, bytes] = {}
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for fname in files:
            if fname.startswith('.'):
                continue
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, data_dir)
            remote_path = os.path.join(REMOTE_DATA_DIR, rel_path)
            with open(local_path, "rb") as f:
                file_map[remote_path] = f.read()

    print(f"Collected {len(file_map)} files to upload.")
    upload_test_data.remote(file_map)

    # ------------------------------------------------------------------
    # Step 2: Run prediction (single GPU call for ALL test cases)
    # ------------------------------------------------------------------
    print(f"\n--- Step 2: Run mask prediction on Modal ({GPU_TYPE}) ---")
    result = predict_masks.remote()
    print(f"Prediction status: {result['status']}")
    if "inference_s" in result:
        print(f"IER inference-only latency: {result['inference_s']:.3f}s")
    if "results" in result:
        for case_id, info in sorted(result["results"].items()):
            print(f"  Case {case_id}: coverage={info['coverage_pct']:.1f}%")
    print(
        "LATENCY_METRICS_JSON: "
        + json.dumps(
            {
                "component": "IER",
                "inference_s": result.get("inference_s"),
                "test_cases": len(result.get("results", {})),
            },
            sort_keys=True,
        )
    )

    # ------------------------------------------------------------------
    # Step 3: Download complete test data folder and replace local copy
    # ------------------------------------------------------------------
    print(f"\n--- Step 3: Download complete test data (with masks) ---")
    dl_result = download_test_data.remote()

    if dl_result["status"] == "success" and dl_result["files"]:
        # Remove the existing local test data folder
        print(f"  Replacing local directory: {data_dir}")
        shutil.rmtree(data_dir)
        os.makedirs(data_dir, exist_ok=True)

        # Write every file from the volume to the local directory
        for rel_path, file_bytes in sorted(dl_result["files"].items()):
            local_path = os.path.join(data_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(file_bytes)

        total = len(dl_result["files"])
        mask_count = sum(
            1 for p in dl_result["files"] if "mask" in p.lower()
        )
        print(f"  Downloaded {total} files ({mask_count} masks).")
    else:
        print("  WARNING: No files found on volume to download.")

    print(f"\n{'=' * 60}")
    print("Pipeline complete!")
    print(f"Results saved to: {data_dir}")
    print(f"{'=' * 60}")
