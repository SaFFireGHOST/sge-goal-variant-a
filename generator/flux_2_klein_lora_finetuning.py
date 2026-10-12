#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import gc
import itertools
import logging
import math
import os
import random
import shutil
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, set_peft_model_state_dict, get_peft_model_state_dict, prepare_model_for_kbit_training
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

import diffusers
import importlib
from typing import Any

def _import_or_get(module: str, name: str) -> Any:
    """Import a symbol with fallbacks without hard import failures."""
    try:
        return getattr(importlib.import_module(module), name)
    except Exception:
        return None

# Resolve Diffusers classes with fallbacks for different package layouts
AutoencoderKLFlux2 = getattr(diffusers, "AutoencoderKLFlux2", None) or _import_or_get(
    "diffusers.models", "AutoencoderKLFlux2"
)
FlowMatchEulerDiscreteScheduler = getattr(
    diffusers, "FlowMatchEulerDiscreteScheduler", None
) or _import_or_get("diffusers.schedulers", "FlowMatchEulerDiscreteScheduler") or _import_or_get(
    "diffusers.schedulers.scheduling_flow_match_euler_discrete", "FlowMatchEulerDiscreteScheduler"
)
Flux2KleinPipeline = getattr(diffusers, "Flux2KleinPipeline", None) or _import_or_get(
    "diffusers.pipelines", "Flux2KleinPipeline"
) or _import_or_get("diffusers.pipelines.flux_2_klein", "Flux2KleinPipeline")
Flux2Transformer2DModel = getattr(diffusers, "Flux2Transformer2DModel", None) or _import_or_get(
    "diffusers.models", "Flux2Transformer2DModel"
) or _import_or_get("diffusers.models.transformers", "Flux2Transformer2DModel")

_missing = [
    name
    for name, obj in {
        "AutoencoderKLFlux2": AutoencoderKLFlux2,
        "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
        "Flux2KleinPipeline": Flux2KleinPipeline,
        "Flux2Transformer2DModel": Flux2Transformer2DModel,
    }.items()
    if obj is None
]
if _missing:
    raise ImportError(
        "Missing Diffusers symbols: "
        + ", ".join(_missing)
        + ". Ensure you installed a Diffusers build that contains Flux 2 Klein." 
        + " (Using git+https://github.com/huggingface/diffusers.git is recommended.)"
    )
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils.state_dict_utils import convert_unet_state_dict_to_peft
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_torch_npu_available, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError(
        "You are trying to load your data using the datasets library. If you wish to train using custom "
        "captions please install the datasets library: `pip install datasets`."
    )

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.34.0.dev0")

logger = get_logger(__name__)

if is_torch_npu_available():
    npu = getattr(torch, "npu", None)
    if npu is not None:
        try:
            npu.config.allow_internal_format = False
        except Exception:
            pass


def build_enhanced_prompt(edit_instruction: str) -> str:
    """
    Build an enhanced prompt for the Flux Kontext model.
    """
    enhanced_prompt = f"""{edit_instruction}

Task: Robotic Manipulation Image Editing

You are performing an image editing task in the domain of robotic manipulation. Your goal is to generate the final edited scene after a robotic arm performs the manipulation action described above.

Inputs Provided:
1. Original Image: The initial scene before any manipulation occurs. This is your reference for the scene layout, lighting, and static elements.
2. Initial Edit Region Mask: A binary mask highlighting the dynamic object(s) in the original image that will be manipulated by the robotic arm. White pixels (value 1) indicate dynamic objects; black pixels (value 0) indicate static background.
3. Final Edit Region Mask: A binary mask showing where the dynamic object(s) should appear in the final edited image after manipulation. This serves as spatial guidance for object placement.

Expected Output:
Generate the final edited scene image where:
- The dynamic object(s) identified in the Initial Edit Region Mask have been moved/manipulated according to the edit instruction
- The object(s) must appear precisely in the regions indicated by the Final Edit Region Mask
- All static background elements from the original image should be preserved exactly
- The edited image should look photorealistic and physically plausible
- Maintain consistent lighting, shadows, and visual style with the original image"""
    
    
    return enhanced_prompt

# Helper functions for packing latents and IDs (Adapting Flux 2 Klein Mechanics)

def _patchify_latents(latents):
    """
    Patchifies latents (B, C, H, W) -> (B, C*4, H/2, W/2)
    """
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
    return latents

def _unpatchify_latents(latents):
    """
    Unpatchifies latents (B, C*4, H/2, W/2) -> (B, C, H, W)
    """
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
    return latents

def _pack_latents(latents):
    """
    pack latents: (batch_size, num_channels, height, width) -> (batch_size, height * width, num_channels)
    """
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
    return latents

def _unpack_latents(latents, height, width, vae_scale_factor):
    """
    Inverse of pack + patchify.
    Input: (B, H*W, C_packed)
    """
    # 1. Unpack to (B, C_packed, H_packed, W_packed)
    # The packed height/width are H/2, W/2 relative to input of pack
    # But wait, pack takes H, W of the input tensor.
    # The input tensor to pack was patchified (H/2, W/2).
    # Reconstruct H_pat, W_pat
    batch_size, seq_len, channels = latents.shape
    
    # We assume square or known aspect ratio, but here we can derive if we know original H, W
    # original H, W passed in are the *original image dims*? No, let's pass patchified dims.
    h_patch = height // 2
    w_patch = width // 2
    
    latents = latents.permute(0, 2, 1).reshape(batch_size, channels, h_patch, w_patch)
    latents = _unpatchify_latents(latents)
    return latents

def _prepare_latent_ids(latents):
    batch_size, _, height, width = latents.shape
    t = torch.arange(1)
    h = torch.arange(height)
    w = torch.arange(width)
    l = torch.arange(1)
    latent_ids = torch.cartesian_prod(t, h, w, l)
    latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
    return latent_ids

def _prepare_image_ids(image_latents, scale=10):
    if not isinstance(image_latents, list):
        raise ValueError(f"Expected `image_latents` to be a list, got {type(image_latents)}.")
    
    # create time offset for each reference image
    # Note: Diffusers implementation starts t at scale (10), then 20, etc.
    t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
    t_coords = [t.view(-1) for t in t_coords]

    image_latent_ids = []
    for x, t in zip(image_latents, t_coords):
        # x is (B, C, H, W) presumably, but reference says x.squeeze(0) -> (C, H, W).
        # We will handle batch properly.
        # Reference logic assumes list of (1, C, H, W).
        # We might have (B, C, H, W) for each.
        
        # Taking the first Item context for IDs structure
        _, height, width = x.shape[1], x.shape[2], x.shape[3] # B, C, H, W
        
        # IDs are shared across batch usually?
        # Using pure torch logic:
        # t is scalar.
        x_ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
        image_latent_ids.append(x_ids)

    image_latent_ids = torch.cat(image_latent_ids, dim=0)
    # Expand to batch
    batch_size = image_latents[0].shape[0]
    image_latent_ids = image_latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

    return image_latent_ids


def log_validation(
    transformer,
    scheduler,
    vae,
    text_encoder,
    tokenizer,
    args,
    accelerator,
    epoch,
):
    logger.info("Running validation...")

    # Deep-copy scheduler so pipeline inference does NOT corrupt training sigmas/timesteps
    inference_scheduler = copy.deepcopy(scheduler)

    pipeline = Flux2KleinPipeline(
        scheduler=inference_scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=accelerator.unwrap_model(transformer),
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # Load validation images
    dataset: Any = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        split="train",
        cache_dir=args.cache_dir,
    )
    # Pick random samples
    indices = range(0, min(args.num_validation_images, len(dataset)))
    
    images = []
    prompts = []
    
    for idx in indices:
        example = dataset[idx]
        # Load necessary inputs
        original = example[args.original_image_column].convert("RGB")
        init_mask = example[args.initial_edit_region_mask_column].convert("RGB")
        final_mask = example[args.final_edit_region_mask_column].convert("RGB")
        prompt = build_enhanced_prompt(example[args.edit_instruction_column])
        
        # Prepare Multimodal Input List
        # Note: Flux2KleinPipeline takes `image` list.
        # We pass [original, init_mask, final_mask]
        cond_images = [original, init_mask, final_mask]
        
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            image = pipeline(
                prompt=prompt,
                image=cond_images, # Multimodal conditioning
                height=original.height,
                width=original.width,
                num_inference_steps=20, # Fast validation
                generator=generator,
                guidance_scale=args.guidance_scale,
            ).images[0]
            
        images.append(image)
        prompts.append(prompt[:100]) # truncated

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            import wandb

            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {prompt}")
                        for i, (image, prompt) in enumerate(zip(images, prompts))
                    ]
                }
            )
            
    del pipeline
    torch.cuda.empty_cache()


def check_cuda_compatibility(device: torch.device):
    """Quick test to detect CUDA binary / kernel incompatibilities early.

    If the host environment has an incompatible PyTorch+CUDA binary for the GPU architecture
    you'll typically see "CUDA error: no kernel image is available for execution on the device".
    We run a small conv op on the specified device and exit with a helpful message if it fails.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Training will run on CPU if no GPUs are present.")
        return

    try:
        x = torch.randn((1, 3, 8, 8), device=device)
        conv = torch.nn.Conv2d(3, 3, 3, padding=1).to(device)
        _ = conv(x)  # If this raises a CUDA kernel error we'll catch it below
    except RuntimeError as e:
        msg = str(e)
        if "no kernel image is available for execution on the device" in msg:
            kernel = os.uname().release if hasattr(os, "uname") else "unknown"
            logger.error("CUDA kernel image error: %s", msg)
            logger.error("Detected Linux kernel: %s. Recommended kernel >= 5.5.0.", kernel)
            logger.error(
                "Likely causes: incompatible PyTorch/CUDA binaries for the GPU compute capability, or mismatched NVIDIA driver.\n"
                "Suggested checks:\n  - Run `nvidia-smi` to see the driver & GPU model.\n  - Run `python -c \"import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))\"`\n"
                "  - Check your container / host CUDA driver matches the PyTorch CUDA build.\n"
                "  - If using custom hardware, install a PyTorch wheel compiled for your GPU arch or set up a container with matching drivers.\n"
                "For debugging, you can set `CUDA_LAUNCH_BLOCKING=1` to get synchronous traces, but this won't fix an incompatible build.\n"
            )
            raise SystemExit(1) from e
        else:
            # Re-raise for other runtime errors
            raise


def run_runtime_diagnostics(device: Optional[torch.device] = None):
    """Runs basic host and torch diagnostics and a small matmul smoke test.

    Returns a dict of diagnostic outputs for logging and debugging.
    """
    info = {}
    try:
        info['uname'] = os.uname().release if hasattr(os, 'uname') else 'unknown'
    except Exception:
        info['uname'] = 'unknown'

    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        info['nvidia_smi'] = result.stdout.strip()
    except Exception as e:
        info['nvidia_smi'] = f'error: {e}'

    try:
        import torch as _torch
        info['torch_version'] = _torch.__version__
        version_info = getattr(_torch, "version", None)
        info['torch_cuda'] = getattr(version_info, "cuda", None)
        if _torch.cuda.is_available():
            info['device_name'] = _torch.cuda.get_device_name(0)
            try:
                info['device_cc'] = torch.cuda.get_device_properties(0).major
            except Exception:
                info['device_cc'] = 'unknown'
    except Exception as e:
        info['torch_info_error'] = str(e)

    # Simple batched matmul smoke test for different dtypes
    smoke = {}
    if torch.cuda.is_available() and device is not None:
        for dt in [torch.float32, getattr(torch, 'bfloat16', None), getattr(torch, 'float16', None)]:
            if dt is None:
                continue
            try:
                a = torch.randn(8, 64, 128, device=device, dtype=dt)
                b = torch.randn(8, 128, 64, device=device, dtype=dt)
                c = torch.bmm(a, b)
                smoke[str(dt)] = 'ok'
            except Exception as e:
                smoke[str(dt)] = f'err: {e}'
    else:
        smoke['cuda'] = 'not_available'

    info['smoke'] = smoke
    logger.info("Runtime diagnostics: %s", json.dumps(info, indent=2))
    return info


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest checkpoint directory in output_dir."""
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


def prune_checkpoints(output_dir: str, limit: Optional[int]) -> None:
    if limit is None or limit <= 0:
        return
    checkpoints = [
        d
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint") and os.path.isdir(os.path.join(output_dir, d))
    ]

    def _sort_key(name: str) -> int:
        parts = name.split("-")
        for p in reversed(parts):
            if p.isdigit():
                return int(p)
        return 0

    checkpoints = sorted(checkpoints, key=_sort_key)
    excess = len(checkpoints) - limit
    if excess <= 0:
        return

    for ckpt in checkpoints[:excess]:
        path = os.path.join(output_dir, ckpt)
        shutil.rmtree(path, ignore_errors=True)


def save_checkpoint(
    accelerator: Accelerator,
    output_dir: str,
    epoch: int,
    global_step: int,
    checkpoints_total_limit: Optional[int],
    num_update_steps_per_epoch: Optional[int] = None,
) -> str:
    """Save an epoch-based checkpoint and training state metadata.
    
    The checkpoint includes:
    - Full accelerator state (model, optimizer, LR scheduler, RNG states)
    - training_state.json with epoch, global_step, and steps_per_epoch
      so that training can resume precisely from where it stopped.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
    accelerator.save_state(save_path)
    state_path = os.path.join(save_path, "training_state.json")
    state_dict = {
        "epoch": epoch,
        "global_step": global_step,
    }
    if num_update_steps_per_epoch is not None:
        state_dict["num_update_steps_per_epoch"] = num_update_steps_per_epoch
    with open(state_path, "w") as f:
        json.dump(state_dict, f)
    prune_checkpoints(output_dir, checkpoints_total_limit)
    print(f"###CHECKPOINT_SAVED### epoch={epoch} path={save_path}")
    return save_path


def _mask_to_numpy(mask):
    if mask is None:
        return None
    if isinstance(mask, np.ndarray):
        arr = mask
    else:
        arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def _pil_to_numpy(img):
    if img is None:
        return None
    if isinstance(img, np.ndarray):
        arr = img
    else:
        img = img.convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    return arr


def _numpy_to_torch(img_np):
    t = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0)
    t = t * 2.0 - 1.0
    return t


def _safe_resize_pair(a_np, b_np):
    if a_np is None or b_np is None:
        return a_np, b_np
    if a_np.shape[:2] == b_np.shape[:2]:
        return a_np, b_np
    a_h, a_w = a_np.shape[:2]
    b_img = Image.fromarray((b_np * 255).astype(np.uint8)).convert("RGB")
    resample_bilinear = getattr(getattr(Image, "Resampling", Image), "BILINEAR", 2)
    b_img = b_img.resize((a_w, a_h), resample=resample_bilinear)
    b_np_resized = np.asarray(b_img).astype(np.float32) / 255.0
    return a_np, b_np_resized


def compute_eval_metrics(
    predicted_images,
    ground_truth_images,
    original_images,
    initial_edit_region_masks,
    final_edit_region_masks,
    prompts_list,
    is_final_validation,
):
    """
    Compute evaluation metrics over supplied lists of PIL images / masks / prompts.
    Returns dict of aggregated metrics.
    """
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    lpips_module = None
    try:
        import lpips as _lpips
        lpips_module = _lpips
    except Exception:
        lpips_module = None

    clip_model_cls = None
    clip_processor_cls = None
    try:
        from transformers import CLIPProcessor as _CLIPProcessor, CLIPModel as _CLIPModel
        clip_model_cls = _CLIPModel
        clip_processor_cls = _CLIPProcessor
    except Exception:
        clip_model_cls = None
        clip_processor_cls = None

    n = len(predicted_images)
    assert n == len(ground_truth_images) == len(original_images) == len(initial_edit_region_masks) == len(final_edit_region_masks) == len(prompts_list)

    per_sample = {
        "psnr": [],
        "ssim": [],
        "lpips": [],
        "masked_psnr": [],
        "masked_ssim": [],
        "masked_lpips": [],
        "unmasked_psnr": [],
        "unmasked_ssim": [],
        "preservation_changed_fraction": [],
        "clip_score_pred": [],
        "clip_score_gt": [],
        "clip_delta_pred_minus_gt": [],
    }

    metric_device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_model = None
    if lpips_module is not None:
        try:
            lpips_model = lpips_module.LPIPS(net="alex").to(metric_device)
            lpips_model.eval()
        except Exception:
            lpips_model = None

    clip_model = None
    clip_processor = None
    if clip_model_cls is not None and clip_processor_cls is not None:
        try:
            clip_model = clip_model_cls.from_pretrained("openai/clip-vit-base-patch32")
            clip_model = cast(Any, clip_model).to(metric_device)
            clip_model.eval()
            clip_processor = clip_processor_cls.from_pretrained("openai/clip-vit-base-patch32")
        except Exception:
            clip_model = None
            clip_processor = None

    def _compute_clip_scores(images_pil, texts):
        if clip_model is None or clip_processor is None:
            return None
        outputs = []
        batch_size = 16
        for i in range(0, len(images_pil), batch_size):
            batch_imgs = images_pil[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            imgs = [img if (img is not None) else Image.new("RGB", (224, 224), (0, 0, 0)) for img in batch_imgs]
            inputs = clip_processor(text=batch_texts, images=imgs, return_tensors="pt", padding=True, truncation=True).to(metric_device)
            with torch.no_grad():
                logits = clip_model(**inputs).logits_per_image
            outputs.extend(logits.diag().detach().cpu().tolist())
        return outputs

    clip_pred_scores = _compute_clip_scores(predicted_images, prompts_list)
    clip_gt_scores = _compute_clip_scores(ground_truth_images, prompts_list)

    for i in range(n):
        pred = predicted_images[i]
        gt = ground_truth_images[i]
        orig = original_images[i]
        init_mask = initial_edit_region_masks[i]
        final_mask = final_edit_region_masks[i]

        def _preserve_format_union(m1, m2):
            if m1 is None and m2 is None:
                return None
            if m1 is None:
                return m2
            if m2 is None:
                return m1
            m1_np = _mask_to_numpy(m1)
            m2_np = _mask_to_numpy(m2)
            if m1_np is None or m2_np is None:
                return None
            return np.maximum(m1_np, m2_np)

        mask_np = _preserve_format_union(init_mask, final_mask)

        pred_np = _pil_to_numpy(pred) if pred is not None else None
        gt_np = _pil_to_numpy(gt) if gt is not None else None
        orig_np = _pil_to_numpy(orig) if orig is not None else None

        if pred_np is None or gt_np is None:
            per_sample["psnr"].append(float("nan"))
            per_sample["ssim"].append(float("nan"))
            per_sample["lpips"].append(float("nan"))
            per_sample["masked_psnr"].append(float("nan"))
            per_sample["masked_ssim"].append(float("nan"))
            per_sample["masked_lpips"].append(float("nan"))
            per_sample["unmasked_psnr"].append(float("nan"))
            per_sample["unmasked_ssim"].append(float("nan"))
            per_sample["preservation_changed_fraction"].append(float("nan"))
            continue

        gt_np, pred_np = _safe_resize_pair(gt_np, pred_np)
        gt_np, orig_np = _safe_resize_pair(gt_np, orig_np)
        h, w = gt_np.shape[:2]

        if mask_np is not None and mask_np.shape[:2] != (h, w):
            mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
            resample_nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST", 0)
            mask_img = mask_img.resize((w, h), resample=resample_nearest)
            mask_np = np.asarray(mask_img).astype(np.float32) / 255.0

        try:
            psnr_val = peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0)
        except Exception:
            psnr_val = float("nan")
        per_sample["psnr"].append(psnr_val)

        try:
            _ssim_ws = min(7, gt_np.shape[0], gt_np.shape[1])
            if _ssim_ws % 2 == 0:
                _ssim_ws -= 1
            ssim_val = structural_similarity(gt_np, pred_np, channel_axis=-1, data_range=1.0, win_size=_ssim_ws) if _ssim_ws >= 3 else float("nan")
        except Exception:
            ssim_val = float("nan")
        per_sample["ssim"].append(ssim_val)

        if lpips_model is not None:
            try:
                gt_t = _numpy_to_torch(gt_np).to(metric_device)
                pred_t = _numpy_to_torch(pred_np).to(metric_device)
                with torch.no_grad():
                    lp_val = lpips_model(gt_t, pred_t).item()
            except Exception:
                lp_val = float("nan")
        else:
            lp_val = float("nan")
        per_sample["lpips"].append(lp_val)

        if mask_np is None:
            per_sample["masked_psnr"].append(float("nan"))
            per_sample["masked_ssim"].append(float("nan"))
            per_sample["masked_lpips"].append(float("nan"))
            per_sample["unmasked_psnr"].append(float("nan"))
            per_sample["unmasked_ssim"].append(float("nan"))
            per_sample["preservation_changed_fraction"].append(float("nan"))
        else:
            mask_bin = (mask_np > 0.5).astype(np.float32)
            inv_mask = 1.0 - mask_bin
            masked_gt = gt_np * mask_bin[..., None]
            masked_pred = pred_np * mask_bin[..., None]
            unmasked_gt = gt_np * inv_mask[..., None]
            unmasked_pred = pred_np * inv_mask[..., None]

            try:
                masked_psnr = peak_signal_noise_ratio(masked_gt, masked_pred, data_range=1.0)
            except Exception:
                masked_psnr = float("nan")
            per_sample["masked_psnr"].append(masked_psnr)

            try:
                _ssim_ws = min(7, masked_gt.shape[0], masked_gt.shape[1])
                if _ssim_ws % 2 == 0:
                    _ssim_ws -= 1
                masked_ssim = structural_similarity(masked_gt, masked_pred, channel_axis=-1, data_range=1.0, win_size=_ssim_ws) if _ssim_ws >= 3 else float("nan")
            except Exception:
                masked_ssim = float("nan")
            per_sample["masked_ssim"].append(masked_ssim)

            if lpips_model is not None:
                try:
                    gt_t = _numpy_to_torch(masked_gt).to(metric_device)
                    pred_t = _numpy_to_torch(masked_pred).to(metric_device)
                    with torch.no_grad():
                        masked_lpips = lpips_model(gt_t, pred_t).item()
                except Exception:
                    masked_lpips = float("nan")
            else:
                masked_lpips = float("nan")
            per_sample["masked_lpips"].append(masked_lpips)

            try:
                unmasked_psnr = peak_signal_noise_ratio(unmasked_gt, unmasked_pred, data_range=1.0)
            except Exception:
                unmasked_psnr = float("nan")
            per_sample["unmasked_psnr"].append(unmasked_psnr)

            try:
                _ssim_ws = min(7, unmasked_gt.shape[0], unmasked_gt.shape[1])
                if _ssim_ws % 2 == 0:
                    _ssim_ws -= 1
                unmasked_ssim = structural_similarity(unmasked_gt, unmasked_pred, channel_axis=-1, data_range=1.0, win_size=_ssim_ws) if _ssim_ws >= 3 else float("nan")
            except Exception:
                unmasked_ssim = float("nan")
            per_sample["unmasked_ssim"].append(unmasked_ssim)

            if orig_np is not None:
                diff = np.abs(pred_np - orig_np).mean(axis=-1)
                changed_fraction = float((diff * inv_mask > 0.05).mean())
            else:
                changed_fraction = float("nan")
            per_sample["preservation_changed_fraction"].append(changed_fraction)

        if clip_pred_scores is not None:
            per_sample["clip_score_pred"].append(clip_pred_scores[i])
        else:
            per_sample["clip_score_pred"].append(float("nan"))

        if clip_gt_scores is not None:
            per_sample["clip_score_gt"].append(clip_gt_scores[i])
        else:
            per_sample["clip_score_gt"].append(float("nan"))

        if clip_pred_scores is not None and clip_gt_scores is not None:
            per_sample["clip_delta_pred_minus_gt"].append(clip_pred_scores[i] - clip_gt_scores[i])
        else:
            per_sample["clip_delta_pred_minus_gt"].append(float("nan"))

    def _safe_stats(lst):
        vals = [x for x in lst if isinstance(x, (int, float)) and not math.isnan(x)]
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    aggregated = {}
    for k, v in per_sample.items():
        mean, std = _safe_stats(v)
        aggregated[f"{k}_mean"] = mean
        aggregated[f"{k}_std"] = std

    prefix = "test" if is_final_validation else "val"
    logger.info("[%s] Evaluation aggregated metrics:", prefix.upper())
    for k, v in aggregated.items():
        logger.info("  %s/%s: %s", prefix, k, v)

    return aggregated


def compute_batch_losses(
    batch,
    args,
    accelerator,
    vae,
    text_encoder,
    tokenizer,
    transformer,
    noise_scheduler,
    weight_dtype,
    vae_scale_factor,
    latents_bn_mean,
    latents_bn_std,
):
    """Compute flow and static losses for a batch.

    Uses the official Flux2 Klein latent processing pipeline:
    encode (mode) → patchify → BN normalize → noise → pack → transformer → unpack
    """

    def encode_img(pixel_values):
        """VAE encode using .mode() (no shift/scale normalization for Flux2)."""
        pixel_values = pixel_values.to(device=accelerator.device, dtype=weight_dtype)
        latents = vae.encode(pixel_values).latent_dist.mode()
        return latents

    # 1. Encode all images to raw latent space (no_grad: VAE is frozen)
    with torch.no_grad():
        latents_raw = encode_img(batch["pixel_values"])
        original_raw = encode_img(batch["original_image_pixel_values"])
        initial_mask_raw = encode_img(batch["initial_edit_region_mask_pixel_values"])
        final_mask_raw = encode_img(batch["final_edit_region_mask_pixel_values"])

    bsz, _, h, w = latents_raw.shape

    # 2. Patchify then batch-normalize (official Flux2 Klein pipeline)
    bn_mean = latents_bn_mean.to(device=latents_raw.device, dtype=latents_raw.dtype)
    bn_std = latents_bn_std.to(device=latents_raw.device, dtype=latents_raw.dtype)

    model_input = Flux2KleinPipeline._patchify_latents(latents_raw)
    model_input = (model_input - bn_mean) / bn_std

    original_patched = Flux2KleinPipeline._patchify_latents(original_raw)
    original_patched = (original_patched - bn_mean) / bn_std

    initial_patched = Flux2KleinPipeline._patchify_latents(initial_mask_raw)
    initial_patched = (initial_patched - bn_mean) / bn_std

    final_patched = Flux2KleinPipeline._patchify_latents(final_mask_raw)
    final_patched = (final_patched - bn_mean) / bn_std

    # 3. Prepare position IDs (use custom helpers that handle batched 4D tensors)
    model_input_ids = _prepare_latent_ids(model_input).to(
        device=accelerator.device, dtype=weight_dtype
    )
    cond_latents_list = [original_patched, initial_patched, final_patched]
    cond_input_ids = _prepare_image_ids(cond_latents_list).to(
        device=accelerator.device, dtype=weight_dtype
    )

    # 4. Add noise in patchified+BN space (before packing)
    noise = torch.randn_like(model_input)
    bsz_packed = bsz

    u = compute_density_for_timestep_sampling(
        weighting_scheme=args.weighting_scheme,
        batch_size=bsz_packed,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        mode_scale=args.mode_scale,
    )
    indices = (u * noise_scheduler.config.num_train_timesteps).long()
    indices = indices.clamp(0, min(noise_scheduler.config.num_train_timesteps - 1,
                                   len(noise_scheduler.sigmas) - 1,
                                   len(noise_scheduler.timesteps) - 1))

    timesteps = noise_scheduler.timesteps[indices].to(accelerator.device)

    sigmas = noise_scheduler.sigmas[indices].to(device=accelerator.device, dtype=model_input.dtype)
    # Expand sigmas to match model_input ndim (B, C, H, W) = 4
    while sigmas.ndim < model_input.ndim:
        sigmas = sigmas.unsqueeze(-1)

    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

    # 5. Pack latents and conditions, concatenate along sequence dim
    packed_noisy = Flux2KleinPipeline._pack_latents(noisy_model_input)
    packed_original = Flux2KleinPipeline._pack_latents(original_patched)
    packed_initial = Flux2KleinPipeline._pack_latents(initial_patched)
    packed_final = Flux2KleinPipeline._pack_latents(final_patched)

    orig_packed_shape = packed_noisy.shape
    hidden_states = torch.cat([packed_noisy, packed_original, packed_initial, packed_final], dim=1)
    img_ids = torch.cat([model_input_ids, cond_input_ids], dim=1)

    # 6. Encode prompt (no_grad: text encoder is frozen)
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt_qwen(
            tokenizer, text_encoder, batch["prompts"], args.max_sequence_length, accelerator.device
        )
    prompt_embeds = prompt_embeds.detach()
    text_ids = text_ids.detach()
    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
    # 7. Guidance
    guidance = None
    if getattr(transformer, 'config', None) and getattr(transformer.config, 'guidance_embeds', False):
        guidance = torch.tensor([args.guidance_scale], device=accelerator.device).expand(bsz)

    # 8. Transformer forward
    model_pred = transformer(
        hidden_states=hidden_states,
        timestep=timesteps / 1000,
        guidance=guidance,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=img_ids,
        return_dict=False,
    )[0]

    # 9. Trim to target region and unpack
    model_pred = model_pred[:, :orig_packed_shape[1], :]
    model_pred = Flux2KleinPipeline._unpack_latents_with_ids(
        model_pred, model_input_ids
    )

    # 10. Flow matching loss (in patchified+BN space)
    target = noise - model_input
    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
    flow_loss = torch.mean(
        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
        1,
    ).mean()

    # 11. Static loss (reconstruction in pixel space for low-noise samples)
    pred_clean = noisy_model_input - sigmas * model_pred  # clean prediction in patchified+BN space

    t_vals = sigmas.view(-1)
    rec_mask = t_vals < args.t_rec_max
    rec_idx = rec_mask.nonzero(as_tuple=False).view(-1)

    static_loss = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)

    if rec_idx.numel() > 0:
        keep_n = max(1, int(args.p_rec_fraction * rec_idx.numel()))
        if rec_idx.numel() > keep_n:
            rec_idx = rec_idx[:keep_n]

        pred_sel = pred_clean[rec_idx]
        orig_sel = original_raw[rec_idx]

        def decode_from_bn(lats_bn):
            """Reverse BN normalize, unpatchify, and VAE decode."""
            lats_patchified = lats_bn * bn_std + bn_mean
            lats_raw = _unpatchify_latents(lats_patchified)
            lats_raw = lats_raw.to(dtype=weight_dtype)
            return vae.decode(lats_raw, return_dict=False)[0]

        def decode_raw(lats):
            """Decode raw VAE latents directly."""
            lats = lats.to(dtype=weight_dtype)
            return vae.decode(lats, return_dict=False)[0]

        # Chunked gradient-checkpointed VAE decode: each chunk's decoder
        # activations are discarded after forward and recomputed one-at-a-time
        # during backward, preventing all 86+ GiB from coexisting in memory.
        _DECODE_CHUNK = 4
        _decode_chunks = []
        for _i in range(0, pred_sel.shape[0], _DECODE_CHUNK):
            _chunk = pred_sel[_i : _i + _DECODE_CHUNK]
            _decode_chunks.append(
                torch.utils.checkpoint.checkpoint(
                    decode_from_bn, _chunk, use_reentrant=False
                )
            )
        pred_img = torch.cat(_decode_chunks, dim=0)

        # orig_img and masks don't need gradients — detach to save massive activation memory
        with torch.no_grad():
            orig_img = decode_raw(orig_sel)

            init_mask_sel = initial_mask_raw[rec_idx]
            final_mask_sel = final_mask_raw[rec_idx]
            init_mask_img = decode_raw(init_mask_sel).mean(dim=1, keepdim=True)
            final_mask_img = decode_raw(final_mask_sel).mean(dim=1, keepdim=True)

            init_mask_img = (init_mask_img - init_mask_img.amin(dim=(2, 3), keepdim=True)) / (
                init_mask_img.amax(dim=(2, 3), keepdim=True) - init_mask_img.amin(dim=(2, 3), keepdim=True) + 1e-6
            )
            final_mask_img = (final_mask_img - final_mask_img.amin(dim=(2, 3), keepdim=True)) / (
                final_mask_img.amax(dim=(2, 3), keepdim=True) - final_mask_img.amin(dim=(2, 3), keepdim=True) + 1e-6
            )

            mask_combined = torch.maximum(init_mask_img, final_mask_img)
            unmasked_region = 1.0 - mask_combined

        sq_diff = (pred_img - orig_img) ** 2
        static_loss = (sq_diff * unmasked_region).mean()

    total_loss = flow_loss + args.lambda_static * static_loss
    return flow_loss, static_loss, total_loss


def run_validation_or_test(
    transformer,
    scheduler,
    vae,
    text_encoder,
    tokenizer,
    args,
    accelerator,
    epoch,
    is_final_validation: bool,
    weight_dtype,
    vae_scale_factor,
    latents_bn_mean,
    latents_bn_std,
):
    """Run validation or test: loss on all samples, metrics/visualization on subset."""
    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return None

    split = "test" if is_final_validation else "validation"
    logger.info("Running %s evaluation...", split)

    eval_args = copy.copy(args)
    eval_args.random_flip = False

    try:
        dataset = KontextDataset(args=eval_args, split=split, repeats=1, center_crop=True)
    except Exception as e:
        logger.warning("No %s split available: %s", split, e)
        return None

    batch_size = args.test_batch_size if is_final_validation else args.val_batch_size
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    flow_losses = []
    static_losses = []
    total_losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{split} loss", total=len(dataloader)):
            flow_loss, static_loss, total_loss = compute_batch_losses(
                batch,
                args,
                accelerator,
                vae,
                text_encoder,
                tokenizer,
                transformer,
                scheduler,
                weight_dtype,
                vae_scale_factor=vae_scale_factor,
                latents_bn_mean=latents_bn_mean,
                latents_bn_std=latents_bn_std,
            )
            flow_losses.append(flow_loss.item())
            static_losses.append(static_loss.item())
            total_losses.append(total_loss.item())

    mean_flow_loss = float(np.mean(flow_losses)) if flow_losses else float("nan")
    mean_static_loss = float(np.mean(static_losses)) if static_losses else float("nan")
    mean_total_loss = float(np.mean(total_losses)) if total_losses else float("nan")

    logger.info("[%s] Losses: flow=%s static=%s total=%s", split.upper(), mean_flow_loss, mean_static_loss, mean_total_loss)
    print(f"\n{'='*60}")
    print(f"[{split.upper()} - Epoch {epoch}] Loss Summary:")
    print(f"  Flow Loss:     {mean_flow_loss:.6f}")
    print(f"  SSC Loss:      {mean_static_loss:.6f}")
    print(f"  Total Loss:    {mean_total_loss:.6f}")
    print(f"{'='*60}")

    # Metrics + visualization stage
    try:
        hf_dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=split,
            cache_dir=args.cache_dir,
        )
    except Exception as e:
        logger.warning("Unable to load %s split for metrics: %s", split, e)
        return {
            "mean_flow_loss": mean_flow_loss,
            "mean_static_loss": mean_static_loss,
            "mean_total_loss": mean_total_loss,
            "metrics": None,
        }

    max_samples = args.max_test_samples if is_final_validation else args.max_val_samples
    total_samples = getattr(hf_dataset, "num_rows", None)
    if total_samples is None:
        logger.warning("Dataset does not expose num_rows; skipping metrics/visualization.")
        return {
            "mean_flow_loss": mean_flow_loss,
            "mean_static_loss": mean_static_loss,
            "mean_total_loss": mean_total_loss,
            "metrics": None,
        }
    if max_samples is not None:
        total_samples = min(total_samples, max_samples)

    viz_root = (
        os.path.join(args.output_dir, "test_visualization")
        if is_final_validation
        else os.path.join(args.output_dir, "validation_visualization")
    )
    if os.path.exists(viz_root):
        shutil.rmtree(viz_root, ignore_errors=True)
    os.makedirs(viz_root, exist_ok=True)

    # Deep-copy scheduler so pipeline inference does NOT corrupt training sigmas/timesteps
    inference_scheduler = copy.deepcopy(scheduler)

    pipeline = Flux2KleinPipeline(
        scheduler=inference_scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=accelerator.unwrap_model(transformer),
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    predicted_images = []
    ground_truth_images = []
    original_images = []
    initial_edit_region_masks = []
    final_edit_region_masks = []
    prompts_list = []

    select_fn = getattr(hf_dataset, "select", None)
    if callable(select_fn):
        subset = select_fn(range(total_samples))
        sample_iter = cast(Any, subset)
    else:
        sample_iter = itertools.islice(hf_dataset, total_samples)

    for i, example in enumerate(tqdm(sample_iter, desc=f"{split} metrics", total=total_samples)):
        original = example[args.original_image_column].convert("RGB")
        init_mask = example[args.initial_edit_region_mask_column].convert("RGB")
        final_mask = example[args.final_edit_region_mask_column].convert("RGB")
        edited = example[args.edited_image_column].convert("RGB")
        prompt = build_enhanced_prompt(example[args.edit_instruction_column])

        cond_images = [original, init_mask, final_mask]
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator = generator.manual_seed(args.seed + i)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = pipeline(
                prompt=prompt,
                image=cond_images,
                height=original.height,
                width=original.width,
                num_inference_steps=20,
                generator=generator,
                guidance_scale=args.guidance_scale,
            ).images[0]

        predicted_images.append(pred)
        ground_truth_images.append(edited)
        original_images.append(original)
        initial_edit_region_masks.append(init_mask)
        final_edit_region_masks.append(final_mask)
        prompts_list.append(prompt)

        if i < args.eval_viz_num:
            pred.save(os.path.join(viz_root, f"{i}_pred.png"))
            edited.save(os.path.join(viz_root, f"{i}_gt.png"))
            original.save(os.path.join(viz_root, f"{i}_orig.png"))
            init_mask.save(os.path.join(viz_root, f"{i}_init_mask.png"))
            final_mask.save(os.path.join(viz_root, f"{i}_final_mask.png"))
            with open(os.path.join(viz_root, f"{i}_prompt.txt"), "w") as pf:
                pf.write(prompt)

    metrics = compute_eval_metrics(
        predicted_images,
        ground_truth_images,
        original_images,
        initial_edit_region_masks,
        final_edit_region_masks,
        prompts_list,
        is_final_validation,
    )

    # Print metrics summary prominently
    if metrics:
        print(f"\n{'='*60}")
        print(f"[{split.upper()} - Epoch {epoch}] Metrics Summary:")
        for k, v in metrics.items():
            if isinstance(v, float) and not math.isnan(v):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")
        print(f"{'='*60}")

    # Cleanup GPU memory from pipeline, CLIP, LPIPS
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    accelerator.wait_for_everyone()

    return {
        "mean_flow_loss": mean_flow_loss,
        "mean_static_loss": mean_static_loss,
        "mean_total_loss": mean_total_loss,
        "metrics": metrics,
    }


def run_original_only_outputs(
    transformer,
    scheduler,
    vae,
    text_encoder,
    tokenizer,
    args,
    accelerator,
):
    """Generate outputs using only original image + prompt (no masks, no metrics/losses)."""
    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return None

    split = args.original_only_split
    logger.info("Running original-only output generation on split=%s", split)

    try:
        hf_dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=split,
            cache_dir=args.cache_dir,
        )
    except Exception as e:
        logger.warning("Unable to load %s split for original-only outputs: %s", split, e)
        return None

    total_samples = getattr(hf_dataset, "num_rows", None)
    max_samples = args.original_only_max_samples
    if total_samples is not None and max_samples is not None:
        total_samples = min(total_samples, max_samples)

    viz_root = os.path.join(args.output_dir, args.original_only_output_subdir, split)
    if os.path.exists(viz_root):
        shutil.rmtree(viz_root, ignore_errors=True)
    os.makedirs(viz_root, exist_ok=True)

    # Deep-copy scheduler so pipeline inference does NOT corrupt training sigmas/timesteps
    inference_scheduler = copy.deepcopy(scheduler)

    pipeline = Flux2KleinPipeline(
        scheduler=inference_scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=accelerator.unwrap_model(transformer),
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    select_fn = getattr(hf_dataset, "select", None)
    if total_samples is not None and callable(select_fn):
        subset = select_fn(range(total_samples))
        sample_iter = cast(Any, subset)
        iter_total = total_samples
    elif max_samples is not None:
        sample_iter = itertools.islice(hf_dataset, max_samples)
        iter_total = max_samples
    else:
        sample_iter = cast(Any, hf_dataset)
        iter_total = None

    generated = 0
    for i, example in enumerate(tqdm(sample_iter, desc=f"{split} original-only", total=iter_total)):
        original = example[args.original_image_column].convert("RGB")
        prompt = build_enhanced_prompt(example[args.edit_instruction_column])

        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator = generator.manual_seed(args.seed + i)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = pipeline(
                prompt=prompt,
                image=original,
                height=original.height,
                width=original.width,
                num_inference_steps=20,
                generator=generator,
                guidance_scale=args.guidance_scale,
            ).images[0]

        pred.save(os.path.join(viz_root, f"{i}_pred.png"))
        original.save(os.path.join(viz_root, f"{i}_orig.png"))
        with open(os.path.join(viz_root, f"{i}_prompt.txt"), "w") as pf:
            pf.write(prompt)
        generated += 1

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    accelerator.wait_for_everyone()

    print(f"###ORIGINAL_ONLY_COMPLETE### split={split} samples={generated} dir={viz_root}")
    return {
        "split": split,
        "num_samples": generated,
        "output_dir": viz_root,
    }


class KontextDataset(Dataset):
    def __init__(
        self,
        args,
        split="train",
        repeats=1,
        buckets=None,
        center_crop=False,
    ):
        self.args = args
        self.center_crop = center_crop
        self.buckets = buckets

        # Downloading and loading a dataset from the hub.
        dataset: Any = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=split,
            cache_dir=args.cache_dir,
        )
        self.dataset = dataset

        # Store column names
        self.edited_image_column = args.edited_image_column
        self.original_image_column = args.original_image_column
        self.initial_edit_region_mask_column = args.initial_edit_region_mask_column
        self.final_edit_region_mask_column = args.final_edit_region_mask_column
        self.edit_instruction_column = args.edit_instruction_column

        # Store repeats factor
        self.repeats = repeats

        # Calculate length considering repeats
        self.num_instance_images = len(self.dataset)
        self._length = self.num_instance_images * repeats
        
        # Pre-compute bucket assignments (simplified here for fixed res or single bucket)
        self.bucket_assignments = [0] * self.num_instance_images
        
        # Determine strict resolution if buckets are simple
        if not buckets:
            self.resolution = args.resolution
        else:
             self.resolution = buckets[0][0] # Assumption

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        # Map index to actual dataset index (handling repeats)
        actual_index = index % self.num_instance_images

        # Load images on-demand
        image = self.dataset[actual_index][self.edited_image_column]
        image = exif_transpose(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        original_image = self.dataset[actual_index][self.original_image_column]
        original_image = exif_transpose(original_image)
        if not original_image.mode == "RGB":
            original_image = original_image.convert("RGB")

        initial_edit_region_mask = self.dataset[actual_index][
            self.initial_edit_region_mask_column
        ]
        initial_edit_region_mask = exif_transpose(initial_edit_region_mask)
        if not initial_edit_region_mask.mode == "RGB":
            initial_edit_region_mask = initial_edit_region_mask.convert("RGB")

        final_edit_region_mask = self.dataset[actual_index][
            self.final_edit_region_mask_column
        ]
        final_edit_region_mask = exif_transpose(final_edit_region_mask)
        if not final_edit_region_mask.mode == "RGB":
            final_edit_region_mask = final_edit_region_mask.convert("RGB")

        # Fixed resolution for now
        size = (self.resolution, self.resolution)

        # Apply transformations
        image, original_image, initial_edit_region_mask, final_edit_region_mask = (
            self.tuple_transform(
                image,
                original_image=original_image,
                initial_edit_region_mask=initial_edit_region_mask,
                final_edit_region_mask=final_edit_region_mask,
                size=size,
                center_crop=self.center_crop,
                random_flip=self.args.random_flip,
            )
        )

        # Get caption and build enhanced prompt
        edit_instruction = self.dataset[actual_index][self.edit_instruction_column]
        enhanced_prompt = build_enhanced_prompt(edit_instruction)

        example = {
            "instance_image": image,
            "original_image": original_image,
            "initial_edit_region_mask": initial_edit_region_mask,
            "final_edit_region_mask": final_edit_region_mask,
            "instance_prompt": enhanced_prompt,
            "edit_instruction": edit_instruction, 
            "bucket_idx": 0,
        }

        return example

    def tuple_transform(
        self,
        image,
        original_image,
        initial_edit_region_mask,
        final_edit_region_mask,
        size=(224, 224),
        center_crop=False,
        random_flip=False,
    ):
        # 1. Resize (deterministic)
        resize = transforms.Resize(
            size, interpolation=transforms.InterpolationMode.BILINEAR
        )
        image = resize(image)
        original_image = resize(original_image)
        initial_edit_region_mask = resize(initial_edit_region_mask)
        final_edit_region_mask = resize(final_edit_region_mask)

        # 2. Crop
        if center_crop:
            crop = transforms.CenterCrop(size)
            image = crop(image)
            original_image = crop(original_image)
            initial_edit_region_mask = crop(initial_edit_region_mask)
            final_edit_region_mask = crop(final_edit_region_mask)
        else:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=size)
            image = TF.crop(image, i, j, h, w)
            original_image = TF.crop(original_image, i, j, h, w)
            initial_edit_region_mask = TF.crop(initial_edit_region_mask, i, j, h, w)
            final_edit_region_mask = TF.crop(final_edit_region_mask, i, j, h, w)

        # 3. Random horizontal flip
        if random_flip:
            do_flip = random.random() < 0.5
            if do_flip:
                image = TF.hflip(image)
                original_image = TF.hflip(original_image)
                initial_edit_region_mask = TF.hflip(initial_edit_region_mask)
                final_edit_region_mask = TF.hflip(final_edit_region_mask)

        # 4. ToTensor + Normalize (deterministic)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.5], [0.5])
        image = normalize(to_tensor(image))
        original_image = normalize(to_tensor(original_image))
        initial_edit_region_mask = normalize(to_tensor(initial_edit_region_mask))
        final_edit_region_mask = normalize(to_tensor(final_edit_region_mask))

        return (image, original_image, initial_edit_region_mask, final_edit_region_mask)


def collate_fn(examples):
    pixel_values = torch.stack([x["instance_image"] for x in examples]).to(memory_format=torch.contiguous_format).float()
    original_image_pixel_values = torch.stack([x["original_image"] for x in examples]).to(memory_format=torch.contiguous_format).float()
    initial_edit_region_mask_pixel_values = torch.stack([x["initial_edit_region_mask"] for x in examples]).to(memory_format=torch.contiguous_format).float()
    final_edit_region_mask_pixel_values = torch.stack([x["final_edit_region_mask"] for x in examples]).to(memory_format=torch.contiguous_format).float()
    prompts = [x["instance_prompt"] for x in examples]
    edit_instructions = [x["edit_instruction"] for x in examples]

    batch = {
        "pixel_values": pixel_values,
        "original_image_pixel_values": original_image_pixel_values,
        "initial_edit_region_mask_pixel_values": initial_edit_region_mask_pixel_values,
        "final_edit_region_mask_pixel_values": final_edit_region_mask_pixel_values,
        "prompts": prompts,
        "edit_instructions": edit_instructions,
    }
    return batch

# Qwen Encoding Logic for Flux 2 Klein
def _prepare_text_ids(prompt_embeds, device, dtype):
    """Prepare 4D position IDs for text tokens matching Flux2KleinPipeline._prepare_text_ids."""
    B, L, _ = prompt_embeds.shape
    out_ids = []
    for _i in range(B):
        t = torch.arange(1)
        h = torch.arange(1)
        w = torch.arange(1)
        l = torch.arange(L)
        coords = torch.cartesian_prod(t, h, w, l)
        out_ids.append(coords)
    return torch.stack(out_ids).to(device=device, dtype=dtype)


def encode_prompt_qwen(tokenizer, text_encoder, prompt, max_sequence_length, device,
                       text_encoder_out_layers=(9, 18, 27)):
    """Encode prompt matching Flux2KleinPipeline._get_qwen3_prompt_embeds exactly."""
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    # 1. Tokenize using chat template (matches pipeline)
    all_input_ids = []
    all_attention_masks = []
    for single_prompt in prompt:
        messages = [{"role": "user", "content": single_prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            # Fallback if enable_thinking not supported
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )
        all_input_ids.append(inputs["input_ids"])
        all_attention_masks.append(inputs["attention_mask"])

    input_ids = torch.cat(all_input_ids, dim=0).to(device)
    attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

    # 2. Encode with use_cache=False to avoid stale KV states
    output = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    # 3. Select specific intermediate hidden layers (matching pipeline: layers 9, 18, 27)
    hidden_states = output.hidden_states
    selected_layers = []
    for k in text_encoder_out_layers:
        if k < len(hidden_states):
            selected_layers.append(hidden_states[k])
        else:
            selected_layers.append(hidden_states[-1])

    dtype = text_encoder.dtype
    out = torch.stack(selected_layers, dim=1).to(dtype=dtype, device=device)  # [B, 3, L, D]
    B_out, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(B_out, seq_len, num_channels * hidden_dim)

    # 4. Create proper positional text_ids (each token gets a unique position)
    text_ids = _prepare_text_ids(prompt_embeds, device, dtype)

    return prompt_embeds, None, text_ids



def main(args):
    # Setup accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_dir))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Install wandb to use wandb.")

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
    logger.info(accelerator.state)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Apply debug and compatibility flags early
    if getattr(args, 'debug_cuda', False):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        logger.warning("CUDA_LAUNCH_BLOCKING set (debug_cuda=True)")

    if getattr(args, 'disable_8bit', False):
        args.use_8bit_adam = False
        logger.warning("8-bit Adam disabled via --disable_8bit")

    if getattr(args, 'disable_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        logger.warning("TF32 disabled via --disable_tf32")

    # Run runtime diagnostics to capture driver/kernel/torch info
    try:
        run_runtime_diagnostics(accelerator.device)
    except Exception as e:
        logger.warning("Runtime diagnostics failed: %s", e)

    # Weights
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load Tokenizer & Text Encoder (Qwen)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        token=hf_token,
    )
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
        token=hf_token,
    )
    
    # Load VAE & Scheduler
    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        token=hf_token,
    )

    # Compute batch normalization statistics from VAE for Flux2 Klein latent processing
    # (official pipeline uses vae.bn.running_mean / vae.bn.running_var for BN normalization)
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(accelerator.device)
    latents_bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(accelerator.device)
    logger.info("VAE BN stats: mean shape=%s, std shape=%s", latents_bn_mean.shape, latents_bn_std.shape)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        revision=args.revision,
        token=hf_token,
    )
    
    # Load Transformer
    transformer = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        token=hf_token,
    )
    
    # Initialize scheduler timesteps for training access
    # Should default to 1000 or config value
    num_train_timesteps = None
    if hasattr(noise_scheduler, "config"):
        cfg = noise_scheduler.config
        num_train_timesteps = getattr(cfg, "num_train_timesteps", None)
        if num_train_timesteps is None and isinstance(cfg, dict):
            num_train_timesteps = cfg.get("num_train_timesteps")
    if num_train_timesteps is None:
        num_train_timesteps = 1000

    # Only initialize scheduler timesteps if not already set (from_pretrained may pre-populate them).
    if noise_scheduler.sigmas is None or noise_scheduler.timesteps is None:
        noise_scheduler.set_timesteps(num_train_timesteps)
    logger.info("Scheduler: sigmas len=%d, timesteps len=%d, num_train_timesteps=%d",
                len(noise_scheduler.sigmas), len(noise_scheduler.timesteps), num_train_timesteps)
    
    # Freeze
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    
    # VAE scaling - Standard Flux VAE scale is 8
    vae_scale_factor = 8

    # Enable gradient checkpointing BEFORE adding LoRA adapter (critical for 9B param model)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled on transformer (before LoRA adapter).")

    # LoRA
    if args.lora_layers:
        target_modules = [x.strip() for x in args.lora_layers.split(",")]
    else:
        target_modules = ["attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0", "proj_mlp"]

    transformer_lora_config = LoraConfig(
        r=args.rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian", target_modules=target_modules
    )
    transformer.add_adapter(transformer_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # ── Custom save / load hooks so accelerator.save_state() only serialises LoRA weights ──
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            modules_to_save = {}

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    modules_to_save["transformer"] = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # Pop weight so that the full model is NOT saved again by accelerator
                if weights:
                    weights.pop()

            Flux2KleinPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                transformer_ = unwrap_model(model)
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = Flux2KleinPipeline.lora_state_dict(input_dir)
        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    "Loading adapter weights from state_dict led to unexpected keys: "
                    f"{', '.join(unexpected_keys)}"
                )

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Move to accel
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)

    if args.original_only_test_outputs:
        run_original_only_outputs(
            transformer,
            noise_scheduler,
            vae,
            text_encoder,
            tokenizer,
            args,
            accelerator,
        )
        accelerator.end_training()
        return

    # Opt
    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate)

    # Dataset
    train_dataset = KontextDataset(args=args, split="train", repeats=args.repeats)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.dataloader_num_workers
    )

    # Prepare
    transformer, optimizer, train_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader)

    # Training
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # Epochs take precedence over max_train_steps
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    logger.info(
        "Training config: %d epochs, %d steps/epoch, %d total steps, batch_size=%d, grad_accum=%d",
        args.num_train_epochs, num_update_steps_per_epoch, args.max_train_steps,
        args.train_batch_size, args.gradient_accumulation_steps,
    )

    global_step = 0
    starting_epoch = 0

    if args.resume_from_checkpoint:
        resume_path = args.resume_from_checkpoint
        if resume_path == "latest":
            latest_ckpt = find_latest_checkpoint(args.output_dir)
            if latest_ckpt is None:
                logger.warning("No checkpoint found to resume from; starting fresh.")
                resume_path = None
            else:
                resume_path = os.path.join(args.output_dir, latest_ckpt)

        if resume_path:
            logger.info("Resuming from checkpoint: %s", resume_path)
            accelerator.load_state(resume_path)
            state_path = os.path.join(resume_path, "training_state.json")
            if os.path.exists(state_path):
                with open(state_path, "r") as f:
                    state = json.load(f)
                starting_epoch = int(state.get("epoch", 0))
                global_step = int(state.get("global_step", starting_epoch * num_update_steps_per_epoch))
                logger.info("Resumed state: epoch=%d, global_step=%d", starting_epoch, global_step)
            else:
                # Fallback: parse epoch from checkpoint directory name
                ckpt_name = os.path.basename(resume_path)
                parts = ckpt_name.split("-")
                epoch_num = 0
                for p in reversed(parts):
                    if p.isdigit():
                        epoch_num = int(p)
                        break
                starting_epoch = epoch_num
                global_step = starting_epoch * num_update_steps_per_epoch
                logger.info("Resumed from directory name: epoch=%d, global_step=%d", starting_epoch, global_step)

    # Recalculate max_train_steps based on remaining epochs
    remaining_steps = (args.num_train_epochs - starting_epoch) * num_update_steps_per_epoch
    logger.info(
        "Starting from epoch %d/%d (global_step=%d). Remaining steps: %d",
        starting_epoch, args.num_train_epochs, global_step, remaining_steps,
    )

    # ──────────────────────────────────────────────────────────────
    # Pre-training validation step (detect issues early before
    # committing to a long training run)
    # ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Running PRE-TRAINING validation to detect issues early...")
    logger.info("=" * 60)
    pre_val_result = run_validation_or_test(
        transformer,
        noise_scheduler,
        vae,
        text_encoder,
        tokenizer,
        args,
        accelerator,
        epoch=0,
        is_final_validation=False,
        weight_dtype=weight_dtype,
        vae_scale_factor=vae_scale_factor,
        latents_bn_mean=latents_bn_mean,
        latents_bn_std=latents_bn_std,
    )
    if accelerator.is_main_process:
        print("###VALIDATION_COMPLETE### epoch=0 (pre-training)")
    logger.info("Pre-training validation complete. Proceeding to training...")

    # Aggressively free GPU memory before training starts
    del pre_val_result
    gc.collect()
    torch.cuda.empty_cache()

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=global_step,
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(starting_epoch, args.num_train_epochs):
        transformer.train()
        logger.info("Starting epoch %d/%d (global_step=%d)", epoch + 1, args.num_train_epochs, global_step)
        epoch_loss_sum = 0.0
        epoch_steps = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                try:
                    flow_loss, static_loss, loss = compute_batch_losses(
                        batch,
                        args,
                        accelerator,
                        vae,
                        text_encoder,
                        tokenizer,
                        transformer,
                        noise_scheduler,
                        weight_dtype,
                        vae_scale_factor,
                        latents_bn_mean,
                        latents_bn_std,
                    )
                except RuntimeError as e:
                    logger.error("RuntimeError during batch loss computation: %s", e)
                    for key in [
                        "pixel_values",
                        "original_image_pixel_values",
                        "initial_edit_region_mask_pixel_values",
                        "final_edit_region_mask_pixel_values",
                    ]:
                        t = batch.get(key)
                        if isinstance(t, torch.Tensor):
                            logger.error("%s: shape=%s dtype=%s device=%s", key, tuple(t.shape), t.dtype, t.device)
                        else:
                            logger.error("%s: not a tensor (type=%s)", key, type(t))

                    try:
                        run_runtime_diagnostics(accelerator.device)
                    except Exception as e2:
                        logger.error("Runtime diagnostics failed: %s", e2)
                    raise
                
                # Detect NaN/Inf losses before backward to avoid cascading CUDA errors
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error("NaN/Inf loss detected at epoch %d step %d: flow=%s static=%s total=%s",
                                 epoch, step, flow_loss.item(), static_loss.item(), loss.item())
                    # Skip this batch to avoid corrupting optimizer state
                    optimizer.zero_grad()
                    continue

                # Print per-step losses (flow, SSC, total)
                if accelerator.is_main_process:
                    logger.info(
                        "[Epoch %d/%d] Step %d/%d | flow_loss=%.6f | ssc_loss=%.6f | total_loss=%.6f",
                        epoch + 1, args.num_train_epochs, step + 1, len(train_dataloader),
                        flow_loss.item(), static_loss.item(), loss.item(),
                    )

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss_sum += loss.detach().item()
                epoch_steps += 1
                
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

        # Flush any pending async CUDA errors before epoch-end operations
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Log epoch summary
        avg_epoch_loss = epoch_loss_sum / max(epoch_steps, 1)
        logger.info(
            "Epoch %d/%d complete — avg_loss=%.6f (%d steps, global_step=%d)",
            epoch + 1, args.num_train_epochs, avg_epoch_loss, epoch_steps, global_step,
        )
        if accelerator.is_main_process and args.report_to == "tensorboard":
            for tracker in accelerator.trackers:
                if hasattr(tracker, "writer"):
                    tracker.writer.add_scalar("epoch/avg_loss", avg_epoch_loss, epoch + 1)

        # Epoch end: checkpoint + validation
        if accelerator.is_main_process:
            if (epoch + 1) % args.checkpointing_epochs == 0:
                save_checkpoint(
                    accelerator,
                    args.output_dir,
                    epoch + 1,
                    global_step,
                    args.checkpoints_total_limit,
                    num_update_steps_per_epoch=num_update_steps_per_epoch,
                )

        if (epoch + 1) % args.validation_epochs == 0:
            val_result = run_validation_or_test(
                transformer,
                noise_scheduler,
                vae,
                text_encoder,
                tokenizer,
                args,
                accelerator,
                epoch + 1,
                is_final_validation=False,
                weight_dtype=weight_dtype,
                vae_scale_factor=vae_scale_factor,
                latents_bn_mean=latents_bn_mean,
                latents_bn_std=latents_bn_std,
            )
            if accelerator.is_main_process:
                print(f"###VALIDATION_COMPLETE### epoch={epoch + 1}")

    # Final test after training completes
    test_result = run_validation_or_test(
        transformer,
        noise_scheduler,
        vae,
        text_encoder,
        tokenizer,
        args,
        accelerator,
        args.num_train_epochs,
        is_final_validation=True,
        weight_dtype=weight_dtype,
        vae_scale_factor=vae_scale_factor,
        latents_bn_mean=latents_bn_mean,
        latents_bn_std=latents_bn_std,
    )
    if accelerator.is_main_process:
        print("###TEST_COMPLETE###")
            
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="flux-2-klein-lora")
    parser.add_argument("--edited_image_column", type=str, default="edited_image")
    parser.add_argument("--original_image_column", type=str, default="original_image")
    parser.add_argument("--initial_edit_region_mask_column", type=str, default="initial_edit_region_mask")
    parser.add_argument("--final_edit_region_mask_column", type=str, default="final_edit_region_mask")
    parser.add_argument("--edit_instruction_column", type=str, default="edit_instruction")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=75)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpointing_epochs", type=int, default=1)
    parser.add_argument("--checkpoints_total_limit", type=int, default=75)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=4)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_layers", type=str, default=None)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--weighting_scheme", type=str, default="none")
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--train_text_encoder", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--validation_epochs", type=int, default=1)
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--max_val_samples", type=int, default=15)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--eval_viz_num", type=int, default=10)
    parser.add_argument("--val_batch_size", type=int, default=150)
    parser.add_argument("--test_batch_size", type=int, default=150)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument(
        "--original_only_test_outputs",
        action="store_true",
        help="Generate outputs with only original image + prompt (no masks, no metrics/losses).",
    )
    parser.add_argument(
        "--original_only_split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split used by --original_only_test_outputs.",
    )
    parser.add_argument(
        "--original_only_max_samples",
        type=int,
        default=15,
        help="Maximum number of samples for --original_only_test_outputs.",
    )
    parser.add_argument(
        "--original_only_output_subdir",
        type=str,
        default="original_only_visualization",
        help="Subdirectory under output_dir to store original-only outputs.",
    )

    # Debugging / compatibility flags
    parser.add_argument("--debug_cuda", action="store_true", help="Set CUDA_LAUNCH_BLOCKING=1 for synchronous CUDA traces")
    parser.add_argument("--force_fp32", action="store_true", help="Force fp32 computation (override mixed precision)")
    parser.add_argument("--disable_8bit", action="store_true", help="Disable 8-bit optimizers (bitsandbytes)")
    parser.add_argument("--disable_tf32", action="store_true", help="Disable TF32 matmul/cudnn flags")

    # Static loss args
    parser.add_argument("--t_rec_max", type=float, default=0.35)
    parser.add_argument("--p_rec_fraction", type=float, default=1.0)
    parser.add_argument("--lambda_static", type=float, default=7.0)
    
    args = parser.parse_args()
    main(args)
