"""
Modal script: 10-worker parallel image editing inference with real-time sync.

- 1 Modal app, 1 shared volume, 10 parallel B200 GPU workers.
- Each worker processes its assigned partition of test cases.
- A local sync loop downloads newly produced edited_image.jpg files in real time.

Usage:
    conda activate lgie
    cd image-editing-code
    modal run modal_flux_kontext_inference_parallel.py \
        --data-dir ../bridge_data_v2_tel_flux2_klein_fer_masks_test_data \
        --checkpoint-dir ../bridge-image-editing-checkpoint/kontext-bridge-image-editing-checkpoint-epoch-13
"""

import modal
import os

# ============================================================================
# Modal App Setup
# ============================================================================
APP_NAME = "flux2-klein-flux-kontext-bridge-e2e-inference"
VOLUME_NAME = "flux2-klein-flux-kontext-bridge-e2e-inference-volume"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

NUM_WORKERS = 50

# GPU configuration
GPU_TYPE = "B200"
GPU_CONFIG = f"{GPU_TYPE}:1"

# CUDA Configuration
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# Paths on the Modal volume
REMOTE_DATA_DIR = "/data/test_data"
REMOTE_CHECKPOINT_DIR = "/data/checkpoint"

# ============================================================================
# Modal Image
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
    .add_local_dir(
        ".",
        remote_path="/root/flux_finetuning",
        ignore=[
            "*.git*", "__pycache__", "*.pyc", "*.pyo", ".env", ".venv",
            "*.log", ".DS_Store", "*.egg-info", ".pytest_cache",
            "models", "models/**", "*.safetensors", "*.bin", "*.pkl",
        ],
    )
)

huggingface_secret = modal.Secret.from_name("huggingface-secret")


# ============================================================================
# Remote: Upload files to volume
# ============================================================================
@app.function(image=image, volumes={"/data": volume}, timeout=86400)
def upload_files_to_volume(file_data: list[dict]):
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
# Remote: Worker inference function — processes a partition of test cases
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=86400,
    memory=65536,
)
def run_worker(worker_id: int, total_workers: int):
    """
    Each worker loads the model, then processes only test cases where
    (sorted_index % total_workers) == worker_id.
    Commits volume after every sample so sync can pick up results.
    """
    import sys
    sys.path.insert(0, "/root/flux_finetuning")

    import torch
    import numpy as np
    from PIL import Image

    from pipeline_flux_kontext_multi_imgs import FluxKontextPipeline
    from flux_kontext_eval_utils import build_enhanced_prompt

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    device = "cuda"
    dtype = torch.bfloat16

    volume.reload()

    print(f"[Worker {worker_id}/{total_workers}] Loading pipeline...")
    pipeline = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=dtype,
    )
    pipeline = pipeline.to(device)

    lora_path = os.path.join(REMOTE_CHECKPOINT_DIR, "pytorch_lora_weights.safetensors")
    if os.path.exists(lora_path):
        print(f"[Worker {worker_id}] Loading LoRA weights...")
        pipeline.load_lora_weights(REMOTE_CHECKPOINT_DIR)
        print(f"[Worker {worker_id}] LoRA weights loaded.")
    else:
        print(f"[Worker {worker_id}] WARNING: No LoRA weights found.")

    pipeline.set_progress_bar_config(disable=True)

    seed = 42
    generator = torch.Generator(device=device).manual_seed(seed)

    # Discover all test cases, sorted numerically
    all_dirs = sorted(
        [d for d in os.listdir(REMOTE_DATA_DIR)
         if os.path.isdir(os.path.join(REMOTE_DATA_DIR, d))],
        key=lambda x: int(x) if x.isdigit() else x,
    )

    # Partition: this worker handles indices where idx % total_workers == worker_id
    my_dirs = [d for i, d in enumerate(all_dirs) if i % total_workers == worker_id]
    print(f"[Worker {worker_id}] Assigned {len(my_dirs)} / {len(all_dirs)} test cases")

    def find_image_file(directory, base_name):
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(directory, base_name + ext)
            if os.path.exists(p):
                return p
        return None

    num_inference_steps = 20
    processed = 0
    skipped = 0

    for tc_name in my_dirs:
        tc_dir = os.path.join(REMOTE_DATA_DIR, tc_name)

        # Skip if already done
        output_path = os.path.join(tc_dir, "edited_image.jpg")
        if os.path.isfile(output_path):
            skipped += 1
            continue

        original_path = find_image_file(tc_dir, "original_image")
        init_mask_path = find_image_file(tc_dir, "initial_edit_region_mask")
        final_mask_path = find_image_file(tc_dir, "final_edit_region_mask")
        instruction_path = os.path.join(tc_dir, "edit_instruction.txt")

        missing = []
        if original_path is None: missing.append("original_image")
        if init_mask_path is None: missing.append("initial_edit_region_mask")
        if final_mask_path is None: missing.append("final_edit_region_mask")
        if not os.path.exists(instruction_path): missing.append("edit_instruction.txt")
        if missing:
            print(f"[Worker {worker_id}] SKIP {tc_name}: missing {missing}")
            continue

        orig_image = Image.open(original_path).convert("RGB")
        init_mask = Image.open(init_mask_path).convert("RGB")
        final_mask = Image.open(final_mask_path).convert("RGB")
        with open(instruction_path, "r", encoding="utf-8") as f:
            edit_instruction = f.readline().strip()

        prompt = build_enhanced_prompt(edit_instruction)
        orig_w, orig_h = orig_image.size

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

        if hasattr(out, "images"):
            pred_img = out.images[0]
        elif isinstance(out, (list, tuple)):
            candidate = out[0]
            pred_img = candidate[0] if isinstance(candidate, (list, tuple)) else candidate
        else:
            pred_img = out

        if pred_img is not None and isinstance(pred_img, Image.Image):
            pred_w, pred_h = pred_img.size
            if (pred_w, pred_h) != (orig_w, orig_h):
                pred_img = pred_img.resize((orig_w, orig_h), resample=Image.BILINEAR)

        if pred_img is not None:
            pred_img.save(output_path, "JPEG", quality=95)
            processed += 1
            # Commit after every sample so sync can pick it up
            volume.commit()
            print(f"[Worker {worker_id}] Saved {tc_name} ({processed} done, {skipped} skipped)")
        else:
            print(f"[Worker {worker_id}] WARNING: Pipeline returned None for {tc_name}")

    volume.commit()
    print(f"[Worker {worker_id}] DONE — processed {processed}, skipped {skipped}")
    return {"worker_id": worker_id, "processed": processed, "skipped": skipped}


# ============================================================================
# Remote: List edited_image.jpg files on volume (for sync)
# ============================================================================
@app.function(image=image, volumes={"/data": volume}, timeout=86400)
def list_edited_images():
    """Return list of test case dirs that have edited_image.jpg."""
    volume.reload()
    results = []
    if not os.path.exists(REMOTE_DATA_DIR):
        return results
    for d in os.listdir(REMOTE_DATA_DIR):
        p = os.path.join(REMOTE_DATA_DIR, d, "edited_image.jpg")
        if os.path.isfile(p):
            results.append(d)
    return results


# ============================================================================
# Remote: Download a single edited_image.jpg
# ============================================================================
@app.function(image=image, volumes={"/data": volume}, timeout=86400)
def download_edited_image(tc_name: str):
    """Read edited_image.jpg for a test case and return bytes."""
    volume.reload()
    p = os.path.join(REMOTE_DATA_DIR, tc_name, "edited_image.jpg")
    if not os.path.isfile(p):
        return None
    with open(p, "rb") as f:
        return f.read()


# ============================================================================
# Remote: Download all files for the full test data folder
# ============================================================================
@app.function(image=image, volumes={"/data": volume}, timeout=86400)
def download_all_files():
    """Walk the entire test data dir and return {rel_path: bytes}."""
    volume.reload()
    file_map = {}
    if not os.path.exists(REMOTE_DATA_DIR):
        return file_map
    for root, dirs, files in os.walk(REMOTE_DATA_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            if fname.startswith("."):
                continue
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, REMOTE_DATA_DIR)
            with open(full, "rb") as f:
                file_map[rel] = f.read()
    return file_map


# ============================================================================
# Local entrypoint
# ============================================================================
@app.local_entrypoint()
def main(
    data_dir: str = "../bridge_data_v2_tel_flux2_klein_fer_masks_test_data",
    checkpoint_dir: str = "../bridge-image-editing-checkpoint/kontext-bridge-image-editing-checkpoint-epoch-13",
    resume: bool = False,
):
    import time
    import shutil
    import threading

    data_dir = os.path.abspath(data_dir)
    checkpoint_dir = os.path.abspath(checkpoint_dir)

    print(f"Data directory      : {data_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Workers             : {NUM_WORKERS}")
    print(f"GPU                 : {GPU_TYPE}")
    print(f"Resume mode         : {resume}")

    # ------------------------------------------------------------------
    # 1-2. Upload data + checkpoint (skip if resume)
    # ------------------------------------------------------------------
    if resume:
        print("\n[1/4] SKIPPED (resume mode)")
        print("[2/4] SKIPPED (resume mode)")
    else:
        print("\n[1/4] Collecting files for upload ...")
        upload_batch = []

        for root, dirs, files in os.walk(data_dir):
            for fname in files:
                local_path = os.path.join(root, fname)
                rel = os.path.relpath(local_path, data_dir)
                remote_path = os.path.join(REMOTE_DATA_DIR, rel)
                with open(local_path, "rb") as f:
                    content = f.read()
                upload_batch.append({"remote_path": remote_path, "content": content})

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

        print("\n[2/4] Uploading to Modal volume ...")
        BATCH_SIZE_BYTES = 40 * 1024 * 1024
        batch = []
        batch_bytes = 0
        for item in upload_batch:
            sz = len(item["content"])
            if batch and (batch_bytes + sz > BATCH_SIZE_BYTES):
                upload_files_to_volume.remote(batch)
                batch = []
                batch_bytes = 0
            batch.append(item)
            batch_bytes += sz
        if batch:
            upload_files_to_volume.remote(batch)
        print("  Upload complete.")

    # ------------------------------------------------------------------
    # 3. Launch real-time sync thread + 10 workers
    # ------------------------------------------------------------------
    print(f"\n[3/4] Launching {NUM_WORKERS} parallel workers + real-time sync ...")

    # Track which edited images we've already downloaded locally
    already_synced = set()
    # Pre-populate with any edited_image.jpg already on local disk
    for d in os.listdir(data_dir):
        local_ei = os.path.join(data_dir, d, "edited_image.jpg")
        if os.path.isfile(local_ei):
            already_synced.add(d)
    print(f"  {len(already_synced)} test cases already have edited_image.jpg locally")

    sync_stop = threading.Event()

    def sync_loop():
        """Periodically poll the volume for new edited_image.jpg and download them."""
        while not sync_stop.is_set():
            try:
                remote_done = list_edited_images.remote()
                new_ones = [tc for tc in remote_done if tc not in already_synced]
                if new_ones:
                    print(f"  [SYNC] Found {len(new_ones)} new edited images to download")
                    for tc_name in new_ones:
                        content = download_edited_image.remote(tc_name)
                        if content is not None:
                            local_path = os.path.join(data_dir, tc_name, "edited_image.jpg")
                            os.makedirs(os.path.dirname(local_path), exist_ok=True)
                            with open(local_path, "wb") as f:
                                f.write(content)
                            already_synced.add(tc_name)
                    print(f"  [SYNC] Total synced: {len(already_synced)}")
            except Exception as e:
                print(f"  [SYNC] Error: {e}")
            sync_stop.wait(30)  # poll every 30 seconds

    sync_thread = threading.Thread(target=sync_loop, daemon=True)
    sync_thread.start()

    # Launch all workers in parallel using starmap
    t0 = time.time()
    worker_args = [(i, NUM_WORKERS) for i in range(NUM_WORKERS)]
    results = list(run_worker.starmap(worker_args))
    elapsed = time.time() - t0

    print(f"\n  All workers finished in {elapsed:.1f}s")
    for r in results:
        print(f"    Worker {r['worker_id']}: processed={r['processed']}, skipped={r['skipped']}")

    # ------------------------------------------------------------------
    # 4. Final sync — download any remaining edited images
    # ------------------------------------------------------------------
    sync_stop.set()
    sync_thread.join(timeout=5)

    print(f"\n[4/4] Final sync — downloading any remaining edited images ...")
    try:
        remote_done = list_edited_images.remote()
        new_ones = [tc for tc in remote_done if tc not in already_synced]
        if new_ones:
            print(f"  Downloading {len(new_ones)} remaining edited images ...")
            for tc_name in new_ones:
                content = download_edited_image.remote(tc_name)
                if content is not None:
                    local_path = os.path.join(data_dir, tc_name, "edited_image.jpg")
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    with open(local_path, "wb") as f:
                        f.write(content)
                    already_synced.add(tc_name)
    except Exception as e:
        print(f"  Final sync error: {e}")

    print(f"\n  Total edited images synced locally: {len(already_synced)}")
    print("\nDone! Edited images saved to each test-case sub-directory as 'edited_image.jpg'.")
