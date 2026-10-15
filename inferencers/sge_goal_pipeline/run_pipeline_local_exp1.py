#!/usr/bin/env python3
"""Run the full pipeline locally and save inference-time benchmarks."""

import argparse
import contextlib
import gc
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
import importlib.util


DEFAULT_DATA_DIR = "/home/riddhi/SGE-Goal-Inferencers/flux-kontext-dev-ft/bridge_and_calvin/dummy_test_data"
DEFAULT_FER_CHECKPOINT_DIR = (
    "/home/riddhi/SGE-Goal-Inferencers/flux-kontext-dev-ft/bridge_and_calvin/"
    "bridge-fer-mask-prediction-checkpoint/kontext-bridge-fer-checkpoint-epoch-38"
)
DEFAULT_IMAGE_EDITING_CHECKPOINT_DIR = (
    "/home/riddhi/SGE-Goal-Inferencers/flux-kontext-dev-ft/bridge_and_calvin/"
    "bridge-image-editing-checkpoint/kontext-ie-checkpoint-epoch-18"
)

FER_INFERENCE_CONFIG = {
    "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-Kontext-dev",
    "guidance_scale": 1.0,
    "num_inference_steps": 28,
    "resolution": 256,
    "seed": 0,
    "max_sequence_length": 512,
}

FER_DEFAULT_INSTRUCTION = """TASK: Edit Region Mask Prediction for Robotic Manipulation

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

DEFAULT_BENCHMARK_FILENAME = "local_inference_benchmarks.json"


def load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_image_file(directory: str, base_name: str) -> Optional[str]:
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"):
        path = os.path.join(directory, base_name + ext)
        if os.path.exists(path):
            return path
    return None


def list_test_cases(data_dir: str) -> List[str]:
    test_cases = [
        d
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
    ]
    return sorted(test_cases, key=lambda x: int(x) if x.isdigit() else x)


def torch_sync_if_needed(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize_times(times: List[float]) -> Dict[str, Optional[float]]:
    if not times:
        return {"mean": None, "std": None, "count": 0}
    arr = np.array(times, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "count": int(arr.size),
    }


def cleanup_torch(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full pipeline locally and save inference benchmarks."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--fer-checkpoint-dir", default=DEFAULT_FER_CHECKPOINT_DIR)
    parser.add_argument(
        "--image-editing-checkpoint-dir", default=DEFAULT_IMAGE_EDITING_CHECKPOINT_DIR
    )
    parser.add_argument(
        "--benchmark-out",
        default=None,
        help="Optional path for benchmark JSON (defaults to data dir).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (cuda or cpu).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip steps that already have outputs.",
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Load one stage at a time to reduce peak VRAM.",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    fer_checkpoint_dir = os.path.abspath(args.fer_checkpoint_dir)
    image_editing_checkpoint_dir = os.path.abspath(args.image_editing_checkpoint_dir)
    benchmark_out = (
        os.path.abspath(args.benchmark_out)
        if args.benchmark_out
        else os.path.join(data_dir, DEFAULT_BENCHMARK_FILENAME)
    )

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not os.path.isdir(fer_checkpoint_dir):
        raise FileNotFoundError(f"FER checkpoint directory not found: {fer_checkpoint_dir}")
    if not os.path.isdir(image_editing_checkpoint_dir):
        raise FileNotFoundError(
            f"Image editing checkpoint directory not found: {image_editing_checkpoint_dir}"
        )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ier_dir = os.path.join(base_dir, "ier-mask-prediction-code")
    fer_dir = os.path.join(base_dir, "fer-mask-prediction-code")
    img_dir = os.path.join(base_dir, "image-editing-code")

    # Make package-local imports resolvable for modules loaded from file paths.
    sys.path.insert(0, img_dir)
    sys.path.insert(0, fer_dir)
    sys.path.insert(0, ier_dir)

    ier_path = os.path.join(ier_dir, "predict_initial_edit_region_mask.py")
    fer_pipeline_path = os.path.join(fer_dir, "pipeline_flux_kontext_multi_imgs.py")
    img_pipeline_path = os.path.join(img_dir, "pipeline_flux_kontext_multi_imgs.py")
    img_utils_path = os.path.join(img_dir, "flux_kontext_eval_utils.py")

    ier_module = load_module("ier_predict", ier_path)
    fer_pipeline_module = load_module("fer_pipeline", fer_pipeline_path)
    img_pipeline_module = load_module("img_pipeline", img_pipeline_path)
    img_utils_module = load_module("img_utils", img_utils_path)

    InitialEditRegionMaskPredictor = ier_module.InitialEditRegionMaskPredictor
    FluxKontextPipelineFER = fer_pipeline_module.FluxKontextPipeline
    FluxKontextPipelineIMG = img_pipeline_module.FluxKontextPipeline
    build_enhanced_prompt = img_utils_module.build_enhanced_prompt
    fer_default_instruction = FER_DEFAULT_INSTRUCTION
    fer_config = FER_INFERENCE_CONFIG

    device = args.device
    autocast_device = "cuda" if device.startswith("cuda") else "cpu"

    print("=" * 70)
    print("Local Pipeline: IER -> FER -> Image Editing")
    print("=" * 70)
    print(f"Data dir: {data_dir}")
    print(f"FER checkpoint: {fer_checkpoint_dir}")
    print(f"Image editing checkpoint: {image_editing_checkpoint_dir}")
    print(f"Device: {device}")
    print("=" * 70)

    test_cases = list_test_cases(data_dir)
    print(f"\nFound {len(test_cases)} test cases.")

    per_case: Dict[str, Dict[str, Any]] = {}
    ier_times: List[float] = []
    fer_times: List[float] = []
    img_times: List[float] = []
    total_times: List[float] = []
    for case_id in test_cases:
        per_case[case_id] = {"status": "pending"}

    if args.low_vram:
        # Stage 1: IER mask prediction (load once, then free GPU memory)
        print("\n[Stage 1] Loading IER predictor...")
        ier_predictor = InitialEditRegionMaskPredictor(device=device)
        for case_id in test_cases:
            case_dir = os.path.join(data_dir, case_id)
            ier_output_path = os.path.join(case_dir, "initial_edit_region_mask.jpg")
            if args.resume and os.path.isfile(ier_output_path):
                per_case[case_id]["ier_sec"] = None
                per_case[case_id]["ier_status"] = "skipped"
                continue

            instruction_path = os.path.join(case_dir, "edit_instruction.txt")
            if not os.path.isfile(instruction_path):
                per_case[case_id]["ier_status"] = "missing_instruction"
                continue

            with open(instruction_path, "r", encoding="utf-8") as f:
                instruction = f.read().strip()

            image_path = find_image_file(case_dir, "original_image")
            if image_path is None:
                per_case[case_id]["ier_status"] = "missing_original_image"
                continue

            image = Image.open(image_path).convert("RGB")
            torch_sync_if_needed(device)
            start = time.perf_counter()
            mask = ier_predictor.predict_mask(image, instruction)
            torch_sync_if_needed(device)
            elapsed = time.perf_counter() - start

            mask_uint8 = (mask * 255).astype(np.uint8)
            mask_image = Image.fromarray(mask_uint8, mode="L")
            mask_image.save(ier_output_path, quality=95)

            per_case[case_id]["ier_sec"] = elapsed
            per_case[case_id]["ier_status"] = "success"
            ier_times.append(elapsed)

        del ier_predictor
        cleanup_torch(device)

        # Stage 2: FER mask prediction pipeline (load once, then free GPU memory)
        print("\n[Stage 2] Loading FER pipeline...")
        fer_weight_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        fer_pipeline = FluxKontextPipelineFER.from_pretrained(
            fer_config["pretrained_model_name_or_path"],
            torch_dtype=fer_weight_dtype,
        )
        fer_pipeline = fer_pipeline.to(device, dtype=fer_weight_dtype)
        fer_lora_path = os.path.join(fer_checkpoint_dir, "pytorch_lora_weights.safetensors")
        if not os.path.isfile(fer_lora_path):
            raise FileNotFoundError(f"FER LoRA weights not found: {fer_lora_path}")
        fer_pipeline.load_lora_weights(
            fer_checkpoint_dir, weight_name="pytorch_lora_weights.safetensors"
        )
        try:
            fer_pipeline.set_progress_bar_config(disable=True)
        except Exception:
            pass

        fer_generator = (
            torch.Generator(device=device).manual_seed(fer_config.get("seed", 0))
            if device.startswith("cuda")
            else None
        )

        for case_id in test_cases:
            case_dir = os.path.join(data_dir, case_id)
            fer_output_path = os.path.join(case_dir, "final_edit_region_mask.jpg")
            if args.resume and os.path.isfile(fer_output_path):
                per_case[case_id]["fer_sec"] = None
                per_case[case_id]["fer_status"] = "skipped"
                continue

            orig_path = find_image_file(case_dir, "original_image")
            init_mask_path = find_image_file(case_dir, "initial_edit_region_mask")
            instruction_path = os.path.join(case_dir, "edit_instruction.txt")

            missing = []
            if orig_path is None:
                missing.append("original_image")
            if init_mask_path is None:
                missing.append("initial_edit_region_mask")
            if not os.path.isfile(instruction_path):
                missing.append("edit_instruction")

            if missing:
                per_case[case_id]["fer_status"] = f"missing_{','.join(missing)}"
                continue

            orig_image = Image.open(orig_path).convert("RGB")
            init_mask = Image.open(init_mask_path).convert("RGB")
            with open(instruction_path, "r", encoding="utf-8") as f:
                edit_instruction = f.read().strip()

            prompt = edit_instruction + "\n\n" + fer_default_instruction.strip()
            target_width, target_height = orig_image.size
            multiple_of = 32
            target_width = (target_width // multiple_of) * multiple_of
            target_height = (target_height // multiple_of) * multiple_of

            max_area = target_width * target_height
            cap_area = fer_config["resolution"] ** 2
            if max_area > cap_area:
                aspect = target_width / target_height
                target_width = int((cap_area * aspect) ** 0.5)
                target_height = int((cap_area / aspect) ** 0.5)
                target_width = (target_width // multiple_of) * multiple_of
                target_height = (target_height // multiple_of) * multiple_of
                max_area = target_width * target_height

            torch_sync_if_needed(device)
            start = time.perf_counter()
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, _ = fer_pipeline.encode_prompt(
                    prompt, prompt_2=None
                )
                prompt_embeds = prompt_embeds.to(device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(device)

            with (
                torch.autocast(autocast_device)
                if device.startswith("cuda")
                else contextlib.nullcontext()
            ):
                out = fer_pipeline(
                    original_image=orig_image,
                    initial_edit_region_mask=init_mask,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    generator=fer_generator,
                    height=target_height,
                    width=target_width,
                    guidance_scale=fer_config["guidance_scale"],
                    num_inference_steps=fer_config["num_inference_steps"],
                    max_sequence_length=fer_config["max_sequence_length"],
                    _auto_resize=False,
                    max_area=max_area,
                )
            torch_sync_if_needed(device)
            elapsed = time.perf_counter() - start

            if hasattr(out, "images"):
                pred_img = out.images[0]
            elif isinstance(out, (list, tuple)):
                candidate = out[0]
                pred_img = candidate[0] if isinstance(candidate, (list, tuple)) else candidate
            else:
                pred_img = out

            if pred_img is None:
                per_case[case_id]["fer_status"] = "pipeline_returned_none"
                continue

            if not isinstance(pred_img, Image.Image):
                if isinstance(pred_img, np.ndarray):
                    pred_img = Image.fromarray(pred_img.astype(np.uint8)).convert("RGB")
                else:
                    per_case[case_id]["fer_status"] = f"unexpected_output_{type(pred_img)}"
                    continue

            orig_w, orig_h = Image.open(orig_path).size
            if pred_img.size != (orig_w, orig_h):
                pred_img = pred_img.resize((orig_w, orig_h), resample=Image.BILINEAR)
            pred_img.save(fer_output_path, quality=95)
            per_case[case_id]["fer_sec"] = elapsed
            per_case[case_id]["fer_status"] = "success"
            fer_times.append(elapsed)

        del fer_pipeline
        cleanup_torch(device)

        # Stage 3: Image editing pipeline (load once)
        print("\n[Stage 3] Loading image editing pipeline...")
        img_weight_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        img_pipeline = FluxKontextPipelineIMG.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype=img_weight_dtype,
        )
        img_pipeline = img_pipeline.to(device, dtype=img_weight_dtype)
        img_lora_path = os.path.join(
            image_editing_checkpoint_dir, "pytorch_lora_weights.safetensors"
        )
        if not os.path.isfile(img_lora_path):
            raise FileNotFoundError(f"Image editing LoRA weights not found: {img_lora_path}")
        img_pipeline.load_lora_weights(image_editing_checkpoint_dir)
        try:
            img_pipeline.set_progress_bar_config(disable=True)
        except Exception:
            pass

        img_generator = (
            torch.Generator(device=device).manual_seed(42)
            if device.startswith("cuda")
            else None
        )

        for case_id in test_cases:
            case_dir = os.path.join(data_dir, case_id)
            img_output_path = os.path.join(case_dir, "edited_image.jpg")
            if args.resume and os.path.isfile(img_output_path):
                per_case[case_id]["image_editing_sec"] = None
                per_case[case_id]["image_editing_status"] = "skipped"
                continue

            orig_path = find_image_file(case_dir, "original_image")
            init_mask_path = find_image_file(case_dir, "initial_edit_region_mask")
            final_mask_path = find_image_file(case_dir, "final_edit_region_mask")
            instruction_path = os.path.join(case_dir, "edit_instruction.txt")

            missing = []
            if orig_path is None:
                missing.append("original_image")
            if init_mask_path is None:
                missing.append("initial_edit_region_mask")
            if final_mask_path is None:
                missing.append("final_edit_region_mask")
            if not os.path.isfile(instruction_path):
                missing.append("edit_instruction")

            if missing:
                per_case[case_id]["image_editing_status"] = f"missing_{','.join(missing)}"
                continue

            orig_image = Image.open(orig_path).convert("RGB")
            init_mask = Image.open(init_mask_path).convert("RGB")
            final_mask = Image.open(final_mask_path).convert("RGB")
            with open(instruction_path, "r", encoding="utf-8") as f:
                edit_instruction = f.readline().strip()

            prompt = build_enhanced_prompt(edit_instruction)
            orig_w, orig_h = orig_image.size

            torch_sync_if_needed(device)
            start = time.perf_counter()
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, _ = img_pipeline.encode_prompt(
                    prompt, prompt_2=None
                )
                prompt_embeds = prompt_embeds.to(device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(device)

            with (
                torch.autocast(autocast_device)
                if device.startswith("cuda")
                else contextlib.nullcontext()
            ):
                out = img_pipeline(
                    original_image=orig_image,
                    initial_edit_region_mask=init_mask,
                    final_edit_region_mask=final_mask,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    num_inference_steps=20,
                    height=orig_h,
                    width=orig_w,
                    _auto_resize=False,
                    max_area=orig_h * orig_w,
                    generator=img_generator,
                )
            torch_sync_if_needed(device)
            elapsed = time.perf_counter() - start

            if hasattr(out, "images"):
                pred_img = out.images[0]
            elif isinstance(out, (list, tuple)):
                candidate = out[0]
                pred_img = candidate[0] if isinstance(candidate, (list, tuple)) else candidate
            else:
                pred_img = out

            if pred_img is None:
                per_case[case_id]["image_editing_status"] = "pipeline_returned_none"
                continue

            if not isinstance(pred_img, Image.Image):
                if isinstance(pred_img, np.ndarray):
                    pred_img = Image.fromarray(pred_img.astype(np.uint8)).convert("RGB")
                else:
                    per_case[case_id]["image_editing_status"] = (
                        f"unexpected_output_{type(pred_img)}"
                    )
                    continue

            pred_w, pred_h = pred_img.size
            if (pred_w, pred_h) != (orig_w, orig_h):
                pred_img = pred_img.resize((orig_w, orig_h), resample=Image.BILINEAR)
            pred_img.save(img_output_path, "JPEG", quality=95)
            per_case[case_id]["image_editing_sec"] = elapsed
            per_case[case_id]["image_editing_status"] = "success"
            img_times.append(elapsed)

        del img_pipeline
        cleanup_torch(device)
    else:
        # Stage 1: IER mask prediction (load once)
        print("\n[Stage 1] Loading IER predictor...")
        ier_predictor = InitialEditRegionMaskPredictor(device=device)

        # Stage 2: FER mask prediction pipeline (load once)
        print("\n[Stage 2] Loading FER pipeline...")
        fer_weight_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        fer_pipeline = FluxKontextPipelineFER.from_pretrained(
            fer_config["pretrained_model_name_or_path"],
            torch_dtype=fer_weight_dtype,
        )
        fer_pipeline = fer_pipeline.to(device, dtype=fer_weight_dtype)
        fer_lora_path = os.path.join(fer_checkpoint_dir, "pytorch_lora_weights.safetensors")
        if not os.path.isfile(fer_lora_path):
            raise FileNotFoundError(f"FER LoRA weights not found: {fer_lora_path}")
        fer_pipeline.load_lora_weights(
            fer_checkpoint_dir, weight_name="pytorch_lora_weights.safetensors"
        )
        try:
            fer_pipeline.set_progress_bar_config(disable=True)
        except Exception:
            pass

        fer_generator = (
            torch.Generator(device=device).manual_seed(fer_config.get("seed", 0))
            if device.startswith("cuda")
            else None
        )

        # Stage 3: Image editing pipeline (load once)
        print("\n[Stage 3] Loading image editing pipeline...")
        img_weight_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        img_pipeline = FluxKontextPipelineIMG.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype=img_weight_dtype,
        )
        img_pipeline = img_pipeline.to(device, dtype=img_weight_dtype)
        img_lora_path = os.path.join(
            image_editing_checkpoint_dir, "pytorch_lora_weights.safetensors"
        )
        if not os.path.isfile(img_lora_path):
            raise FileNotFoundError(f"Image editing LoRA weights not found: {img_lora_path}")
        img_pipeline.load_lora_weights(image_editing_checkpoint_dir)
        try:
            img_pipeline.set_progress_bar_config(disable=True)
        except Exception:
            pass

        img_generator = (
            torch.Generator(device=device).manual_seed(42)
            if device.startswith("cuda")
            else None
        )

        for case_id in test_cases:
            case_dir = os.path.join(data_dir, case_id)

            # IER prediction
            ier_output_path = os.path.join(case_dir, "initial_edit_region_mask.jpg")
            if args.resume and os.path.isfile(ier_output_path):
                per_case[case_id]["ier_sec"] = None
                per_case[case_id]["ier_status"] = "skipped"
            else:
                instruction_path = os.path.join(case_dir, "edit_instruction.txt")
                if not os.path.isfile(instruction_path):
                    per_case[case_id]["ier_status"] = "missing_instruction"
                else:
                    with open(instruction_path, "r", encoding="utf-8") as f:
                        instruction = f.read().strip()

                    image_path = find_image_file(case_dir, "original_image")
                    if image_path is None:
                        per_case[case_id]["ier_status"] = "missing_original_image"
                    else:
                        image = Image.open(image_path).convert("RGB")
                        torch_sync_if_needed(device)
                        start = time.perf_counter()
                        mask = ier_predictor.predict_mask(image, instruction)
                        torch_sync_if_needed(device)
                        elapsed = time.perf_counter() - start

                        mask_uint8 = (mask * 255).astype(np.uint8)
                        mask_image = Image.fromarray(mask_uint8, mode="L")
                        mask_image.save(ier_output_path, quality=95)

                        per_case[case_id]["ier_sec"] = elapsed
                        per_case[case_id]["ier_status"] = "success"
                        ier_times.append(elapsed)

            # FER prediction
            fer_output_path = os.path.join(case_dir, "final_edit_region_mask.jpg")
            if args.resume and os.path.isfile(fer_output_path):
                per_case[case_id]["fer_sec"] = None
                per_case[case_id]["fer_status"] = "skipped"
            else:
                orig_path = find_image_file(case_dir, "original_image")
                init_mask_path = find_image_file(case_dir, "initial_edit_region_mask")
                instruction_path = os.path.join(case_dir, "edit_instruction.txt")

                missing = []
                if orig_path is None:
                    missing.append("original_image")
                if init_mask_path is None:
                    missing.append("initial_edit_region_mask")
                if not os.path.isfile(instruction_path):
                    missing.append("edit_instruction")

                if missing:
                    per_case[case_id]["fer_status"] = f"missing_{','.join(missing)}"
                else:
                    orig_image = Image.open(orig_path).convert("RGB")
                    init_mask = Image.open(init_mask_path).convert("RGB")
                    with open(instruction_path, "r", encoding="utf-8") as f:
                        edit_instruction = f.read().strip()

                    prompt = edit_instruction + "\n\n" + fer_default_instruction.strip()
                    target_width, target_height = orig_image.size
                    multiple_of = 32
                    target_width = (target_width // multiple_of) * multiple_of
                    target_height = (target_height // multiple_of) * multiple_of

                    max_area = target_width * target_height
                    cap_area = fer_config["resolution"] ** 2
                    if max_area > cap_area:
                        aspect = target_width / target_height
                        target_width = int((cap_area * aspect) ** 0.5)
                        target_height = int((cap_area / aspect) ** 0.5)
                        target_width = (target_width // multiple_of) * multiple_of
                        target_height = (target_height // multiple_of) * multiple_of
                        max_area = target_width * target_height

                    torch_sync_if_needed(device)
                    start = time.perf_counter()
                    with torch.no_grad():
                        prompt_embeds, pooled_prompt_embeds, _ = fer_pipeline.encode_prompt(
                            prompt, prompt_2=None
                        )
                        prompt_embeds = prompt_embeds.to(device)
                        pooled_prompt_embeds = pooled_prompt_embeds.to(device)

                    with (
                        torch.autocast(autocast_device)
                        if device.startswith("cuda")
                        else contextlib.nullcontext()
                    ):
                        out = fer_pipeline(
                            original_image=orig_image,
                            initial_edit_region_mask=init_mask,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            generator=fer_generator,
                            height=target_height,
                            width=target_width,
                            guidance_scale=fer_config["guidance_scale"],
                            num_inference_steps=fer_config["num_inference_steps"],
                            max_sequence_length=fer_config["max_sequence_length"],
                            _auto_resize=False,
                            max_area=max_area,
                        )
                    torch_sync_if_needed(device)
                    elapsed = time.perf_counter() - start

                    if hasattr(out, "images"):
                        pred_img = out.images[0]
                    elif isinstance(out, (list, tuple)):
                        candidate = out[0]
                        pred_img = (
                            candidate[0] if isinstance(candidate, (list, tuple)) else candidate
                        )
                    else:
                        pred_img = out

                    if pred_img is None:
                        per_case[case_id]["fer_status"] = "pipeline_returned_none"
                    else:
                        if not isinstance(pred_img, Image.Image):
                            if isinstance(pred_img, np.ndarray):
                                pred_img = Image.fromarray(pred_img.astype(np.uint8)).convert(
                                    "RGB"
                                )
                            else:
                                per_case[case_id]["fer_status"] = (
                                    f"unexpected_output_{type(pred_img)}"
                                )
                                pred_img = None

                    if pred_img is not None:
                        orig_w, orig_h = Image.open(orig_path).size
                        if pred_img.size != (orig_w, orig_h):
                            pred_img = pred_img.resize(
                                (orig_w, orig_h), resample=Image.BILINEAR
                            )
                        pred_img.save(fer_output_path, quality=95)
                        per_case[case_id]["fer_sec"] = elapsed
                        per_case[case_id]["fer_status"] = "success"
                        fer_times.append(elapsed)

            # Image editing
            img_output_path = os.path.join(case_dir, "edited_image.jpg")
            if args.resume and os.path.isfile(img_output_path):
                per_case[case_id]["image_editing_sec"] = None
                per_case[case_id]["image_editing_status"] = "skipped"
            else:
                orig_path = find_image_file(case_dir, "original_image")
                init_mask_path = find_image_file(case_dir, "initial_edit_region_mask")
                final_mask_path = find_image_file(case_dir, "final_edit_region_mask")
                instruction_path = os.path.join(case_dir, "edit_instruction.txt")

                missing = []
                if orig_path is None:
                    missing.append("original_image")
                if init_mask_path is None:
                    missing.append("initial_edit_region_mask")
                if final_mask_path is None:
                    missing.append("final_edit_region_mask")
                if not os.path.isfile(instruction_path):
                    missing.append("edit_instruction")

                if missing:
                    per_case[case_id]["image_editing_status"] = (
                        f"missing_{','.join(missing)}"
                    )
                else:
                    orig_image = Image.open(orig_path).convert("RGB")
                    init_mask = Image.open(init_mask_path).convert("RGB")
                    final_mask = Image.open(final_mask_path).convert("RGB")
                    with open(instruction_path, "r", encoding="utf-8") as f:
                        edit_instruction = f.readline().strip()

                    prompt = build_enhanced_prompt(edit_instruction)
                    orig_w, orig_h = orig_image.size

                    torch_sync_if_needed(device)
                    start = time.perf_counter()
                    with torch.no_grad():
                        prompt_embeds, pooled_prompt_embeds, _ = img_pipeline.encode_prompt(
                            prompt, prompt_2=None
                        )
                        prompt_embeds = prompt_embeds.to(device)
                        pooled_prompt_embeds = pooled_prompt_embeds.to(device)

                    with (
                        torch.autocast(autocast_device)
                        if device.startswith("cuda")
                        else contextlib.nullcontext()
                    ):
                        out = img_pipeline(
                            original_image=orig_image,
                            initial_edit_region_mask=init_mask,
                            final_edit_region_mask=final_mask,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            num_inference_steps=20,
                            height=orig_h,
                            width=orig_w,
                            _auto_resize=False,
                            max_area=orig_h * orig_w,
                            generator=img_generator,
                        )
                    torch_sync_if_needed(device)
                    elapsed = time.perf_counter() - start

                    if hasattr(out, "images"):
                        pred_img = out.images[0]
                    elif isinstance(out, (list, tuple)):
                        candidate = out[0]
                        pred_img = (
                            candidate[0] if isinstance(candidate, (list, tuple)) else candidate
                        )
                    else:
                        pred_img = out

                    if pred_img is None:
                        per_case[case_id]["image_editing_status"] = "pipeline_returned_none"
                    else:
                        if not isinstance(pred_img, Image.Image):
                            if isinstance(pred_img, np.ndarray):
                                pred_img = Image.fromarray(pred_img.astype(np.uint8)).convert(
                                    "RGB"
                                )
                            else:
                                per_case[case_id]["image_editing_status"] = (
                                    f"unexpected_output_{type(pred_img)}"
                                )
                                pred_img = None

                    if pred_img is not None:
                        pred_w, pred_h = pred_img.size
                        if (pred_w, pred_h) != (orig_w, orig_h):
                            pred_img = pred_img.resize(
                                (orig_w, orig_h), resample=Image.BILINEAR
                            )
                        pred_img.save(img_output_path, "JPEG", quality=95)
                        per_case[case_id]["image_editing_sec"] = elapsed
                        per_case[case_id]["image_editing_status"] = "success"
                        img_times.append(elapsed)

    for case_id in test_cases:
        if (
            per_case[case_id].get("ier_status") == "success"
            and per_case[case_id].get("fer_status") == "success"
            and per_case[case_id].get("image_editing_status") == "success"
        ):
            total_time = (
                per_case[case_id]["ier_sec"]
                + per_case[case_id]["fer_sec"]
                + per_case[case_id]["image_editing_sec"]
            )
            per_case[case_id]["total_sec"] = total_time
            total_times.append(total_time)

        per_case[case_id]["status"] = "done"

    benchmarks = {
        "data_dir": data_dir,
        "fer_checkpoint_dir": fer_checkpoint_dir,
        "image_editing_checkpoint_dir": image_editing_checkpoint_dir,
        "device": device,
        "summary": {
            "ier_sec": summarize_times(ier_times),
            "fer_sec": summarize_times(fer_times),
            "image_editing_sec": summarize_times(img_times),
            "total_sec": summarize_times(total_times),
        },
        "per_case": per_case,
    }

    os.makedirs(os.path.dirname(benchmark_out), exist_ok=True)
    with open(benchmark_out, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)

    print("\nBenchmark results saved to:")
    print(benchmark_out)


if __name__ == "__main__":
    main()
