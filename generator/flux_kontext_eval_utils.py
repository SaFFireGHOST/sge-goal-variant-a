import os
import shutil
import math
import contextlib
import copy
import numpy as np
from PIL import Image
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import load_image
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3
)

from pipeline_flux_kontext_multi_imgs import FluxKontextPipeline

# Optional imports (graceful fallback)
try:
    import lpips
    LPIPS_AVAILABLE = True
except Exception:
    LPIPS_AVAILABLE = False

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except Exception:
    CLIP_AVAILABLE = False

# Global cache for CLIP and LPIPS models to avoid reloading
_CLIP_MODEL_CACHE = {}
_LPIPS_MODEL_CACHE = {}


def build_enhanced_prompt(edit_instruction: str) -> str:
    """
    Build an enhanced prompt for the Flux Kontext model that clearly describes
    the task, inputs, and expected output for robotic manipulation image editing.
    
    The edit instruction is placed FIRST to ensure it's included in CLIP's 77-token
    window, while detailed context follows for T5's longer sequence handling.
    
    Args:
        edit_instruction: The short edit instruction describing the manipulation
                         (e.g., "put the broccoli in the pot")
    
    Returns:
        Enhanced prompt string with full task description
    """
    # Edit instruction comes FIRST so CLIP (77 tokens) captures the key action
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


# helper conversions
def pil_to_numpy(img):
    """PIL.Image -> float32 ndarray in [0,1], shape (H,W,3)"""
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

def numpy_to_torch(img_np):
    """(H,W,3) [0,1] -> torch.tensor (1,3,H,W) in [-1,1] for LPIPS"""
    t = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0)
    # [-1,1]
    t = t * 2.0 - 1.0
    return t

def crop_to_mask_bbox(img_np, mask_np, pad=4):
    """Crop image and mask to tight bbox containing mask (if any). Returns cropped image and mask."""
    ys, xs = np.where(mask_np > 0.5)
    if len(xs) == 0 or len(ys) == 0:
        return None, None
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, mask_np.shape[1] - 1)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, mask_np.shape[0] - 1)
    return img_np[y0 : y1 + 1, x0 : x1 + 1, :], mask_np[y0 : y1 + 1, x0 : x1 + 1]

def safe_resize_pair(a_np, b_np):
    """Resize b_np to a_np's HxW if shapes differ (uses PIL bilinear)."""
    if a_np is None or b_np is None:
        return a_np, b_np
    if a_np.shape[:2] == b_np.shape[:2]:
        return a_np, b_np
    a_h, a_w = a_np.shape[:2]
    b_img = Image.fromarray((b_np * 255).astype(np.uint8)).convert("RGB")
    b_img = b_img.resize((a_w, a_h), resample=Image.BILINEAR)
    b_np_resized = np.asarray(b_img).astype(np.float32) / 255.0
    return a_np, b_np_resized

def compute_clip_score_batch(images_pil, texts, device="cpu", clip_model=None, clip_processor=None):
    """
    Compute CLIP score for list of PIL images vs list of texts (same length).
    Returns list of floats (score per sample). If CLIP not available, returns None.
    
    Optionally accepts pre-loaded clip_model and clip_processor to avoid reloading.
    """
    global _CLIP_MODEL_CACHE
    
    if not CLIP_AVAILABLE:
        return None
    
    # Use cached or provided model
    if clip_model is None or clip_processor is None:
        cache_key = device
        if cache_key not in _CLIP_MODEL_CACHE:
            _CLIP_MODEL_CACHE[cache_key] = {
                'model': CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval(),
                'processor': CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            }
        clip_model = _CLIP_MODEL_CACHE[cache_key]['model']
        clip_processor = _CLIP_MODEL_CACHE[cache_key]['processor']
    
    outputs = []
    batch_size = 16  # Increased batch size for efficiency
    for i in range(0, len(images_pil), batch_size):
        batch_imgs = images_pil[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]
        # some pipelines may return None images; replace with blank if needed
        imgs = [img if (img is not None) else Image.new("RGB", (224, 224), (0, 0, 0)) for img in batch_imgs]
        inputs = clip_processor(text=batch_texts, images=imgs, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_embeds = clip_model.get_image_features(**{k: inputs[k] for k in ["pixel_values"]})
            text_embeds = clip_model.get_text_features(**{k: inputs[k] for k in ["input_ids", "attention_mask"]})
            # normalize
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            sims = (image_embeds * text_embeds).sum(dim=-1).cpu().numpy()  # cosine
            # CLIPScore formulation (torchmetrics / CLIPScore uses 100*max(cos,0))
            sims = np.maximum(sims, 0.0) * 100.0
            outputs.extend(sims.tolist())
    return outputs


def get_lpips_model(device="cuda"):
    """Get or create cached LPIPS model on the specified device."""
    global _LPIPS_MODEL_CACHE
    
    if not LPIPS_AVAILABLE:
        return None
    
    if device not in _LPIPS_MODEL_CACHE:
        try:
            _LPIPS_MODEL_CACHE[device] = lpips.LPIPS(net="alex").to(device).eval()
        except Exception:
            return None
    
    return _LPIPS_MODEL_CACHE[device]

def compute_eval_metrics(
    predicted_images,
    ground_truth_images,
    original_images,
    initial_edit_region_masks,
    final_edit_region_masks,
    prompts_list,
    is_final_validation,
    logger
):
    """
    Compute a set of evaluation metrics over the supplied lists of PIL images / masks / prompts.

    All inputs are lists of equal length. Images are expected as PIL.Image or numpy arrays.
    Returns an aggregated dict of metrics (mean and std where applicable).
    """
    n = len(predicted_images)
    assert n == len(ground_truth_images) == len(original_images) == len(initial_edit_region_masks) == len(final_edit_region_masks) == len(prompts_list), "All input lists must have equal length"

    per_sample = {
        "psnr": [],
        "ssim": [],
        "lpips": [],
        "masked_psnr": [],
        "masked_ssim": [],
        "masked_lpips": [],
        "unmasked_psnr": [],
        "unmasked_ssim": [],
        "preservation_changed_fraction": [],  # fraction of unmasked pixels changed > threshold
        "clip_score_pred": [],
        "clip_score_gt": [],
        "clip_delta_pred_minus_gt": [],
    }

    # Determine device for metric computation
    metric_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get cached LPIPS model on GPU for faster computation
    lpips_alex = get_lpips_model(metric_device) if LPIPS_AVAILABLE else None
    if lpips_alex is None and LPIPS_AVAILABLE:
        logger.warning("LPIPS available but failed to initialize")

    # compute CLIP scores (batch) if transformers available - uses cached model
    clip_pred_scores = None
    clip_gt_scores = None
    if CLIP_AVAILABLE:
        try:
            clip_pred_scores = compute_clip_score_batch(predicted_images, prompts_list, device=metric_device)
            clip_gt_scores = compute_clip_score_batch(ground_truth_images, prompts_list, device=metric_device)
        except Exception as e:
            logger.warning(f"Failed to compute CLIP scores in batch: {e}")
            clip_pred_scores = None
            clip_gt_scores = None

    # iterate each sample
    for i in range(n):
        pred = predicted_images[i]
        gt = ground_truth_images[i]
        orig = original_images[i]
        init_mask = initial_edit_region_masks[i]
        final_mask = final_edit_region_masks[i]
        prompt = prompts_list[i]

        def _mask_to_numpy(x):
            """Convert PIL.Image or numpy mask to float32 numpy array in range [0,1]."""
            if x is None:
                return None
            if isinstance(x, Image.Image):
                x = np.array(x)
            x = x.astype(np.float32)
            if x.max() > 1.0:
                x /= 255.0
            return x

        def _preserve_format_union(init_mask, final_mask):
            """Compute union of two masks while preserving original format."""
            if init_mask is None and final_mask is None:
                return None
            if init_mask is None:
                return final_mask
            if final_mask is None:
                return init_mask

            # Remember original format
            was_pil = isinstance(init_mask, Image.Image)
            was_rgb = (
                isinstance(init_mask, Image.Image) and init_mask.mode == "RGB"
            ) or (isinstance(init_mask, np.ndarray) and init_mask.ndim == 3)

            # Convert to [0,1]
            init_np = _mask_to_numpy(init_mask)
            final_np = _mask_to_numpy(final_mask)

            # If RGB, take one channel (since all channels are equal)
            if init_np.ndim == 3:
                init_np = init_np[..., 0]
            if final_np.ndim == 3:
                final_np = final_np[..., 0]

            # Compute union
            union_np = np.clip(init_np + final_np, 0, 1)
            union_np = (union_np > 0.5).astype(np.float32)

            # Restore same range
            if (isinstance(init_mask, np.ndarray) and init_mask.max() > 1.0) or (
                isinstance(init_mask, Image.Image)
                and (np.array(init_mask).max() > 1.0)
            ):
                union_np = union_np * 255.0

            # Restore same channel structure
            if was_rgb:
                union_np = np.stack([union_np] * 3, axis=-1)

            # Return same type as input
            if was_pil:
                # Convert to same mode
                if was_rgb:
                    union_np = union_np.astype(np.uint8)
                    return Image.fromarray(union_np, mode="RGB")
                else:
                    union_np = union_np.astype(np.uint8)
                    return Image.fromarray(union_np, mode="L")
            else:
                return union_np.astype(np.uint8 if union_np.max() > 1 else np.float32)

        mask = _preserve_format_union(init_mask, final_mask)

        # convert to numpy [0,1]
        pred_np = pil_to_numpy(pred) if pred is not None else None
        gt_np = pil_to_numpy(gt) if gt is not None else None
        orig_np = pil_to_numpy(orig) if orig is not None else None

        # Robust mask conversion -> binary float array [0,1]
        if mask is None:
            mask_np = None
        else:
            # Convert PIL -> numpy array, keep numpy arrays as-is
            if isinstance(mask, np.ndarray):
                m_arr = mask.astype(np.float32)
            else:
                # ensure RGB or grayscale converted to array (H,W,3) or (H,W)
                m_arr = np.asarray(mask.convert("RGB")).astype(np.float32)

            # If 3 channels, reduce to single channel. Use max over channels (robust to small differences)
            if m_arr.ndim == 3:
                m_single = np.max(m_arr[..., :3], axis=2)
            else:
                m_single = m_arr

            # Normalize if values are in 0-255 range
            if m_single.max() > 1.0:
                m_single = m_single / 255.0

            # Threshold to binary mask
            mask_np = (m_single > 0.5).astype(np.float32)

        # ensure pred and gt same size (resize pred to gt if mismatch)
        # This is the correct approach: we resize the prediction to match ground truth dimensions
        if pred_np is None or gt_np is None:
            # skip sample if missing
            logger.warning(f"Missing predicted or ground-truth image for sample {i}; skipping metrics for this sample.")
            continue
        # Resize prediction to ground truth size (not vice versa)
        gt_np, pred_np = safe_resize_pair(gt_np, pred_np)  # Resizes pred_np to gt_np's size
        gt_np, orig_np = safe_resize_pair(gt_np, orig_np)  # Resizes orig_np to gt_np's size
        h, w = gt_np.shape[:2]

        # if mask exists and different shape, resize mask to gt size (nearest)
        if mask_np is not None and mask_np.shape[:2] != (h, w):
            mask_img = Image.fromarray((mask_np * 255).astype(np.uint8)).resize((w, h), resample=Image.NEAREST)
            mask_np = (np.asarray(mask_img).astype(np.float32) / 255.0) > 0.5
            mask_np = mask_np.astype(np.float32)

        # --- PSNR (global) ---
        try:
            psnr_val = float(peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0))
        except Exception:
            # fallback MSE-based
            mse = float(np.mean((gt_np - pred_np) ** 2))
            psnr_val = 10.0 * math.log10(1.0 / (mse + 1e-12))
        per_sample["psnr"].append(psnr_val)

        # --- SSIM (global) ---
        try:
            ssim_val = float(structural_similarity(gt_np, pred_np, multichannel=True, data_range=1.0))
        except Exception:
            ssim_val = float(0.0)
        per_sample["ssim"].append(ssim_val)

        # --- LPIPS (global) ---
        if lpips_alex is not None:
            try:
                # LPIPS expects [-1,1] and torch tensors
                with torch.no_grad():
                    # move to cpu/cuda consistent with lpips_alex device
                    device_lpips = next(lpips_alex.parameters()).device
                    t_pred = numpy_to_torch(pred_np).to(device_lpips)
                    t_gt = numpy_to_torch(gt_np).to(device_lpips)
                    lpips_val = float(lpips_alex.forward(t_gt, t_pred).cpu().numpy().reshape(-1)[0])
                per_sample["lpips"].append(lpips_val)
            except Exception as e:
                logger.warning(f"LPIPS failure on sample {i}: {e}")
                per_sample["lpips"].append(float("nan"))
        else:
            per_sample["lpips"].append(float("nan"))

        # --- Masked metrics (edited region) ---
        if mask_np is not None and mask_np.sum() > 0:
            # crop to mask bbox for LPIPS & local metrics
            bbox_crop_pred, bbox_mask = crop_to_mask_bbox(pred_np, mask_np)
            bbox_crop_gt, _ = crop_to_mask_bbox(gt_np, mask_np)
            # PSNR in mask (compute on cropped region or masked pixels)
            try:
                # option: compute PSNR on *masked pixels* on full image
                masked_pred = pred_np * mask_np[..., None]
                masked_gt = gt_np * mask_np[..., None]
                # compute MSE only over masked pixels
                denom = mask_np.sum() * 3.0
                mse_masked = float(((masked_gt - masked_pred) ** 2).sum() / max(denom, 1.0))
                masked_psnr = 10.0 * math.log10(1.0 / (mse_masked + 1e-12))
            except Exception:
                masked_psnr = float("nan")
            per_sample["masked_psnr"].append(masked_psnr)

            # SSIM on masked (use crop if available)
            try:
                if bbox_crop_gt is not None and bbox_crop_pred is not None:
                    masked_ssim = float(structural_similarity(bbox_crop_gt, bbox_crop_pred, multichannel=True, data_range=1.0))
                else:
                    masked_ssim = float(structural_similarity(gt_np * mask_np[..., None], pred_np * mask_np[..., None], multichannel=True, data_range=1.0))
            except Exception:
                masked_ssim = float("nan")
            per_sample["masked_ssim"].append(masked_ssim)

            # LPIPS in mask: crop to bbox and compute LPIPS if possible
            if lpips_alex is not None and bbox_crop_gt is not None and bbox_crop_pred is not None:
                try:
                    t1 = numpy_to_torch(bbox_crop_gt)
                    t2 = numpy_to_torch(bbox_crop_pred)
                    device_lpips = next(lpips_alex.parameters()).device
                    t1 = t1.to(device_lpips); t2 = t2.to(device_lpips)
                    with torch.no_grad():
                        lpips_masked_val = float(lpips_alex.forward(t1, t2).cpu().numpy().reshape(-1)[0])
                    per_sample["masked_lpips"].append(lpips_masked_val)
                except Exception as e:
                    logger.warning(f"Masked LPIPS failure on sample {i}: {e}")
                    per_sample["masked_lpips"].append(float("nan"))
            else:
                per_sample["masked_lpips"].append(float("nan"))
        else:
            # no edit region -> mark as nan
            per_sample["masked_psnr"].append(float("nan"))
            per_sample["masked_ssim"].append(float("nan"))
            per_sample["masked_lpips"].append(float("nan"))

        # --- Unmasked/preservation region metrics ---
        if mask_np is not None:
            inv_mask = 1.0 - mask_np
            if inv_mask.sum() > 0:
                try:
                    masked_pred_un = pred_np * inv_mask[..., None]
                    masked_gt_un = gt_np * inv_mask[..., None]
                    denom_un = inv_mask.sum() * 3.0
                    mse_un = float(((masked_gt_un - masked_pred_un) ** 2).sum() / max(denom_un, 1.0))
                    unmasked_psnr = 10.0 * math.log10(1.0 / (mse_un + 1e-12))
                except Exception:
                    unmasked_psnr = float("nan")
                per_sample["unmasked_psnr"].append(unmasked_psnr)

                # SSIM for unmasked region: crop to whole image using mask applied
                try:
                    unmasked_ssim = float(structural_similarity(gt_np * inv_mask[..., None], pred_np * inv_mask[..., None], multichannel=True, data_range=1.0))
                except Exception:
                    unmasked_ssim = float("nan")
                per_sample["unmasked_ssim"].append(unmasked_ssim)

                # fraction of unmasked pixels changed above small threshold (preservation measure)
                try:
                    per_pixel_l1 = np.mean(np.abs(orig_np - pred_np), axis=-1)  # HxW
                    # threshold slightly above quantization noise
                    changed = (per_pixel_l1 * (inv_mask > 0.5)) > 0.02
                    changed_frac = changed.sum() / max((inv_mask > 0.5).sum(), 1)
                except Exception:
                    changed_frac = float("nan")
                per_sample["preservation_changed_fraction"].append(float(changed_frac))
            else:
                per_sample["unmasked_psnr"].append(float("nan"))
                per_sample["unmasked_ssim"].append(float("nan"))
                per_sample["preservation_changed_fraction"].append(float("nan"))
        else:
            # if no mask provided, skip unmasked eval
            per_sample["unmasked_psnr"].append(float("nan"))
            per_sample["unmasked_ssim"].append(float("nan"))
            per_sample["preservation_changed_fraction"].append(float("nan"))

        # CLIP scores (from precomputed batch if available)
        if clip_pred_scores is not None and clip_gt_scores is not None:
            try:
                cp = float(clip_pred_scores[i])
                cg = float(clip_gt_scores[i])
            except Exception:
                cp = float("nan"); cg = float("nan")
        else:
            # fallback: try to compute single sample via transformers if available
            if CLIP_AVAILABLE:
                try:
                    device = "cpu"
                    if torch.cuda.is_available():
                        device = "cuda"
                    cp_arr = compute_clip_score_batch([pred], [prompt], device=device)
                    cg_arr = compute_clip_score_batch([gt], [prompt], device=device)
                    cp = cp_arr[0] if cp_arr else float("nan")
                    cg = cg_arr[0] if cg_arr else float("nan")
                except Exception:
                    cp = float("nan"); cg = float("nan")
            else:
                cp = float("nan"); cg = float("nan")
        per_sample["clip_score_pred"].append(cp)
        per_sample["clip_score_gt"].append(cg)
        per_sample["clip_delta_pred_minus_gt"].append(cp - cg if (not (math.isnan(cp) or math.isnan(cg))) else float("nan"))

    # Aggregate (mean and std where numeric)
    def safe_stats(lst):
        arr = np.array([x for x in lst if (x is not None and (not (isinstance(x, float) and math.isnan(x))))], dtype=float)
        if arr.size == 0:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

    aggregated = {}
    for k, v in per_sample.items():
        stats = safe_stats(v)
        aggregated[f"{k}_mean"] = stats["mean"]
        aggregated[f"{k}_std"] = stats["std"]

    # Log metrics with appropriate prefix
    prefix = "test" if is_final_validation else "val"
    # human-friendly logging
    log_items = {
        f"{prefix}/psnr_mean": aggregated["psnr_mean"],
        f"{prefix}/ssim_mean": aggregated["ssim_mean"],
        f"{prefix}/lpips_mean": aggregated["lpips_mean"],
        f"{prefix}/masked_psnr_mean": aggregated["masked_psnr_mean"],
        f"{prefix}/masked_ssim_mean": aggregated["masked_ssim_mean"],
        f"{prefix}/masked_lpips_mean": aggregated["masked_lpips_mean"],
        f"{prefix}/unmasked_psnr_mean": aggregated["unmasked_psnr_mean"],
        f"{prefix}/unmasked_ssim_mean": aggregated["unmasked_ssim_mean"],
        f"{prefix}/preservation_changed_fraction_mean": aggregated["preservation_changed_fraction_mean"],
        f"{prefix}/clip_score_pred_mean": aggregated["clip_score_pred_mean"],
        f"{prefix}/clip_score_gt_mean": aggregated["clip_score_gt_mean"],
        f"{prefix}/clip_delta_mean": aggregated["clip_delta_pred_minus_gt_mean"],
    }

    # Log each metric
    logger.info(f"[{prefix.upper()}] Evaluation aggregated metrics:")
    for k, v in log_items.items():
        logger.info(f"{k}: {v:.6f}" if (isinstance(v, float) and (not math.isnan(v))) else f"{k}: {v}")

    # Return aggregated metrics dict (user code can also log/store them)
    return aggregated

def run_validation_or_test(
    pipeline,
    args,
    accelerator,
    is_final_validation: bool,
    epoch,
    torch_dtype,
    dataloader,
    dataset,
    logger,
    skip_loss_computation: bool = True,  # Skip loss computation to save time
    num_inference_steps: int = 20,  # Reduced from default 28 for faster validation
    max_val_samples: int = None,  # Limit samples for pipeline inference only (loss computed on all samples)
):
    """
    Runs validation or final test with optimized performance.

    Parameters
    ----------
    pipeline : FluxKontextPipeline-like object (must provide .encode_prompt and .__call__)
    args : argparse-like object (uses fields referenced in the training script)
    accelerator : huggingface Accelerator instance
    is_final_validation : bool (True = testing)
    epoch : int or None (None when testing)
    torch_dtype : torch.dtype used for pipeline/device
    dataloader : PyTorch DataLoader for batched loss computation
    dataset : Hugging Face dataset (iterable of samples). Columns:
              args.original_image_column, args.initial_edit_region_mask_column,
              args.final_edit_region_mask_column, args.edit_instruction_column,
              args.edited_image_column
    logger : logging.Logger instance
    skip_loss_computation : bool - Skip expensive loss computation for intermediate validation
    num_inference_steps : int - Number of diffusion steps (default 20 for faster validation)
    max_val_samples : int or None - Limit samples for pipeline inference only (loss is always computed on all samples)
    
    Returns
    -------
    dict with {
        'mean_flow_loss': float,
        'mean_static_loss': float, 
        'mean_total_loss': float,
        'metrics': <whatever compute_eval_metrics returns or None>
    }
    """
    
    # For intermediate validation, default max_val_samples for inference
    if not is_final_validation and max_val_samples is None:
        max_val_samples = min(50, len(dataset))  # Default: 50 samples for intermediate validation inference
    
    logger.info(f"[{'TEST' if is_final_validation else 'VAL'}] Running {'full' if is_final_validation else 'fast'} evaluation...")
    if not is_final_validation:
        logger.info(f"[VAL] Using {max_val_samples} samples, {num_inference_steps} inference steps for speed")

    # local helpers
    def _maybe_load_image(x):
        # Accept PIL.Image, path string, numpy array, or HF-style dict
        # Always returns RGB to avoid VAE channel mismatch (masks may be grayscale)
        if x is None:
            return None
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        if isinstance(x, str):
            try:
                img = load_image(x)
                return img.convert("RGB")
            except Exception:
                return Image.open(x).convert("RGB")
        if isinstance(x, dict) and "image" in x:
            return _maybe_load_image(x["image"])
        # if numpy array
        try:
            import numpy as _np
            if isinstance(x, _np.ndarray):
                return Image.fromarray(x.astype("uint8")).convert("RGB")
        except Exception:
            pass
        # Last resort: try to convert if it's a PIL-like object
        if hasattr(x, 'convert'):
            return x.convert("RGB")
        return x

    device = None
    try:
        device = accelerator.device
    except Exception:
        device = None

    # Extract sub-modules
    text_encoder_one, text_encoder_two = pipeline.text_encoder, pipeline.text_encoder_2
    vae = pipeline.vae
    transformer = pipeline.transformer
    
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels

    # Ensure model in eval mode
    try:
        transformer.eval()
    except Exception:
        pass
    try:
        if args.train_text_encoder:
            text_encoder_one.eval()
    except Exception:
        pass

    # ========================================================================
    # Compute loss: flow loss + static loss, batched for efficiency
    # Loss is computed on ALL samples in the dataloader (no max_val_samples limit)
    # ========================================================================
    mean_flow_loss = float("nan")
    mean_static_loss = float("nan")
    mean_total_loss = float("nan")
    flow_losses = []
    static_losses = []
    total_losses = []
    
    if not skip_loss_computation or is_final_validation:
        # Only compute loss for final test or if explicitly requested
        logger.info(f"[{'TEST' if is_final_validation else 'VAL'}] Computing validation loss (flow + static) on ALL dataloader samples...")
        
        # Load scheduler for loss computation
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        
        def unwrap_model(model):
            model = accelerator.unwrap_model(model)
            model = model._orig_mod if is_compiled_module(model) else model
            return model
        
        def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
            sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
            schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
            timesteps = timesteps.to(accelerator.device)
            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma
        
        def decode_latents(latents):
            """
            latents : torch.Tensor (B, C_lat, H_lat, W_lat) in the same scaled form as 'model_input'
            returns : torch.Tensor (B, C_img, H_img, W_img) in image-space (float32, not necessarily [0,1])
            """
            # Undo the encoding scaling/shift
            lat_to_decode = latents / vae_config_scaling_factor + vae_config_shift_factor

            # Make latents the same dtype as the VAE parameters (handles fp32 / bf16 / fp16 safely)
            vae_param_dtype = next(vae.parameters()).dtype
            lat_to_decode = lat_to_decode.to(dtype=vae_param_dtype)

            # Decode with VAE (handle different return types)
            dec_out = vae.decode(lat_to_decode)
            if hasattr(dec_out, "sample"):
                img = dec_out.sample
            elif isinstance(dec_out, (tuple, list)):
                img = dec_out[0]
            else:
                img = dec_out

            # We want the image in float32 for losses / logging, cast back to float32 here:
            img = img.to(dtype=torch.float32)
            return img
        
        # Static loss hyperparameters (same as training)
        t_rec_max = getattr(args, "t_rec_max", 0.35)
        p_rec_fraction = getattr(args, "p_rec_fraction", 1.0)
        LAMBDA_STATIC = getattr(args, "lambda_static", 7.0)

        with torch.no_grad():
            loss_dataloader = dataloader
            total_batches = len(dataloader)
            
            for batch_idx, batch in enumerate(tqdm(loss_dataloader, desc="Computing loss (flow + static)", total=total_batches)):
                # encode prompts
                prompts = batch["prompts"]
                prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(prompts, prompt_2=None)
                if device is not None:
                    try:
                        prompt_embeds = prompt_embeds.to(device)
                        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
                        text_ids = text_ids.to(device)
                    except Exception:
                        pass

                device = accelerator.device if accelerator is not None else None

                pixel_values = (
                    batch["pixel_values"].to(device=device, non_blocking=True, dtype=vae.dtype)
                    if "pixel_values" in batch and batch["pixel_values"] is not None
                    else None
                )
                original_image_pixel_values = (
                    batch["original_image_pixel_values"].to(device=device, non_blocking=True, dtype=vae.dtype)
                    if "original_image_pixel_values" in batch and batch["original_image_pixel_values"] is not None
                    else None
                )
                initial_edit_region_mask_pixel_values = (
                    batch["initial_edit_region_mask_pixel_values"].to(device=device, non_blocking=True, dtype=vae.dtype)
                    if "initial_edit_region_mask_pixel_values" in batch and batch["initial_edit_region_mask_pixel_values"] is not None
                    else None
                )
                final_edit_region_mask_pixel_values = (
                    batch["final_edit_region_mask_pixel_values"].to(device=device, non_blocking=True, dtype=vae.dtype)
                    if "final_edit_region_mask_pixel_values" in batch and batch["final_edit_region_mask_pixel_values"] is not None
                    else None
                )

                # VAE encode
                if args.vae_encode_mode == "sample":
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    original_image_model_input = vae.encode(original_image_pixel_values).latent_dist.sample()
                    initial_edit_region_mask_model_input = vae.encode(initial_edit_region_mask_pixel_values).latent_dist.sample()
                    final_edit_region_mask_model_input = vae.encode(final_edit_region_mask_pixel_values).latent_dist.sample()
                else:
                    model_input = vae.encode(pixel_values).latent_dist.mode()
                    original_image_model_input = vae.encode(original_image_pixel_values).latent_dist.mode()
                    initial_edit_region_mask_model_input = vae.encode(initial_edit_region_mask_pixel_values).latent_dist.mode()
                    final_edit_region_mask_model_input = vae.encode(final_edit_region_mask_pixel_values).latent_dist.mode()

                # Preprocessing
                model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                model_input = model_input.to(dtype=torch_dtype)
                original_image_model_input = (original_image_model_input - vae_config_shift_factor) * vae_config_scaling_factor
                original_image_model_input = original_image_model_input.to(dtype=torch_dtype)
                initial_edit_region_mask_model_input = (initial_edit_region_mask_model_input - vae_config_shift_factor) * vae_config_scaling_factor
                initial_edit_region_mask_model_input = initial_edit_region_mask_model_input.to(dtype=torch_dtype)
                final_edit_region_mask_model_input = (final_edit_region_mask_model_input - vae_config_shift_factor) * vae_config_scaling_factor
                final_edit_region_mask_model_input = final_edit_region_mask_model_input.to(dtype=torch_dtype)

                vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

                # Prepare latent IDs
                latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    model_input.shape[0], model_input.shape[2] // 2, model_input.shape[3] // 2,
                    accelerator.device, model_input.dtype,
                )
                original_image_latents_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    original_image_model_input.shape[0], original_image_model_input.shape[2] // 2,
                    original_image_model_input.shape[3] // 2, accelerator.device, original_image_model_input.dtype,
                )
                original_image_latents_ids[..., 0] = 1
                latent_image_ids = torch.cat([latent_image_ids, original_image_latents_ids], dim=0)

                initial_edit_region_mask_latents_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    initial_edit_region_mask_model_input.shape[0], initial_edit_region_mask_model_input.shape[2] // 2,
                    initial_edit_region_mask_model_input.shape[3] // 2, accelerator.device, initial_edit_region_mask_model_input.dtype,
                )
                initial_edit_region_mask_latents_ids[..., 0] = 2
                latent_image_ids = torch.cat([latent_image_ids, initial_edit_region_mask_latents_ids], dim=0)

                final_edit_region_mask_latents_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    final_edit_region_mask_model_input.shape[0], final_edit_region_mask_model_input.shape[2] // 2,
                    final_edit_region_mask_model_input.shape[3] // 2, accelerator.device, final_edit_region_mask_model_input.dtype,
                )
                final_edit_region_mask_latents_ids[..., 0] = 3
                latent_image_ids = torch.cat([latent_image_ids, final_edit_region_mask_latents_ids], dim=0)

                # Sample noise
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme, batch_size=bsz,
                    logit_mean=args.logit_mean, logit_std=args.logit_std, mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # Pack latents
                packed_noisy_model_input = FluxKontextPipeline._pack_latents(
                    noisy_model_input, batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2], width=model_input.shape[3],
                )
                orig_inp_shape = packed_noisy_model_input.shape

                packed_original_image_input = FluxKontextPipeline._pack_latents(
                    original_image_model_input, batch_size=original_image_model_input.shape[0],
                    num_channels_latents=original_image_model_input.shape[1],
                    height=original_image_model_input.shape[2], width=original_image_model_input.shape[3],
                )
                packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_original_image_input], dim=1)

                packed_initial_edit_region_mask_input = FluxKontextPipeline._pack_latents(
                    initial_edit_region_mask_model_input, batch_size=initial_edit_region_mask_model_input.shape[0],
                    num_channels_latents=initial_edit_region_mask_model_input.shape[1],
                    height=initial_edit_region_mask_model_input.shape[2], width=initial_edit_region_mask_model_input.shape[3],
                )
                packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_initial_edit_region_mask_input], dim=1)

                packed_final_edit_region_mask_input = FluxKontextPipeline._pack_latents(
                    final_edit_region_mask_model_input, batch_size=final_edit_region_mask_model_input.shape[0],
                    num_channels_latents=final_edit_region_mask_model_input.shape[1],
                    height=final_edit_region_mask_model_input.shape[2], width=final_edit_region_mask_model_input.shape[3],
                )
                packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_final_edit_region_mask_input], dim=1)

                # Guidance
                guidance = None
                try:
                    has_guidance = unwrap_model(transformer).config.guidance_embeds
                except Exception:
                    try:
                        has_guidance = getattr(transformer.config, "guidance_embeds", False)
                    except Exception:
                        has_guidance = False

                if has_guidance:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])

                # Model prediction
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, : orig_inp_shape[1]]
                model_pred_latents = FluxKontextPipeline._unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                # Compute flow loss (same as training)
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                target = noise - model_input
                flow_loss_per_example = torch.mean(
                    (weighting.float() * (model_pred_latents.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1,
                )
                flow_loss = flow_loss_per_example.mean()

                # -------------------------
                # Static-scene consistency loss (ensures unmasked region stays identical to original image)
                # Same computation as in training
                # -------------------------
                # predicted_clean_latents is latent-space one-step recon:
                predicted_clean_latents = noise - model_pred_latents  # shape: (B, C_lat, H_lat, W_lat)

                # compute per-sample t-values from sigmas
                t_vals = sigmas.view(sigmas.shape[0], -1)[:, 0].detach()  # shape (B,)

                # select only those samples we consider "easy" to reconstruct (sigmas < t_rec_max)
                rec_mask = t_vals < t_rec_max  # boolean mask length B
                rec_idx = rec_mask.nonzero(as_tuple=False).view(-1)

                static_loss = torch.tensor(0.0, device=model_input.device, dtype=model_input.dtype)
                # Only compute heavy ops if there are any eligible samples
                if rec_idx.numel() > 0:
                    # optionally subsample eligible indices so we don't decode entire batch
                    keep_n = max(1, int(p_rec_fraction * rec_idx.numel()))
                    if rec_idx.numel() > keep_n:
                        perm = torch.randperm(rec_idx.numel(), device=rec_idx.device)
                        rec_idx = rec_idx[perm[:keep_n]]

                    # gather predicted / original latents and mask latents
                    pred_sel_lat = predicted_clean_latents[rec_idx]
                    orig_sel_lat = original_image_model_input[rec_idx]
                    initial_mask_sel_lat = initial_edit_region_mask_model_input[rec_idx]
                    final_mask_sel_lat = final_edit_region_mask_model_input[rec_idx]

                    # decode to image-space using VAE decode
                    pred_sel_img = decode_latents(pred_sel_lat)
                    orig_sel_img = decode_latents(orig_sel_lat)

                    initial_mask_sel_img = decode_latents(initial_mask_sel_lat)
                    # normalize mask -> [0,1] per-sample
                    initial_m_min = initial_mask_sel_img.amin(dim=[1, 2, 3], keepdim=True)
                    initial_m_max = initial_mask_sel_img.amax(dim=[1, 2, 3], keepdim=True)
                    initial_mask_normalized = (initial_mask_sel_img - initial_m_min) / (initial_m_max - initial_m_min + 1e-8)
                    initial_mask_normalized = torch.clamp(initial_mask_normalized, 0.0, 1.0)

                    final_mask_sel_img = decode_latents(final_mask_sel_lat)
                    # normalize mask -> [0,1] per-sample
                    final_m_min = final_mask_sel_img.amin(dim=[1, 2, 3], keepdim=True)
                    final_m_max = final_mask_sel_img.amax(dim=[1, 2, 3], keepdim=True)
                    final_mask_normalized = (final_mask_sel_img - final_m_min) / (final_m_max - final_m_min + 1e-8)
                    final_mask_normalized = torch.clamp(final_mask_normalized, 0.0, 1.0)

                    # Take union of both normalized masks (pixelwise maximum)
                    mask_normalized = torch.maximum(initial_mask_normalized, final_mask_normalized)
                    mask_normalized = torch.clamp(mask_normalized, 0.0, 1.0)

                    # channel-mean to get single-channel probabilities
                    mask_chan_mean = mask_normalized.mean(dim=1, keepdim=True)  # (k,1,H,W)

                    # unmasked region probability
                    unmasked_region = (1.0 - mask_chan_mean).detach()

                    # compute squared difference between pred and original only on unmasked region
                    sq_diff = (pred_sel_img - orig_sel_img) ** 2
                    masked_sq_diff = sq_diff * unmasked_region
                    static_loss = masked_sq_diff.mean()

                # --- combine losses (same as training) ---
                total_loss = flow_loss + (LAMBDA_STATIC * static_loss)

                # Accumulate losses
                flow_losses.append(flow_loss.detach().cpu().item())
                static_losses.append(static_loss.detach().cpu().item())
                total_losses.append(total_loss.detach().cpu().item())

        # Compute means
        mean_flow_loss = float(np.mean(flow_losses)) if len(flow_losses) > 0 else float("nan")
        mean_static_loss = float(np.mean(static_losses)) if len(static_losses) > 0 else float("nan")
        mean_total_loss = float(np.mean(total_losses)) if len(total_losses) > 0 else float("nan")
    
    # Log loss
    if is_final_validation:
        logger.info(f"[TEST] Mean flow loss = {mean_flow_loss:.6f}, Mean static loss = {mean_static_loss:.6f}, Mean total loss = {mean_total_loss:.6f}")
        try:
            accelerator.log({
                "test/mean_flow_loss": mean_flow_loss,
                "test/mean_static_loss": mean_static_loss,
                "test/mean_total_loss": mean_total_loss,
            })
        except Exception:
            pass
    else:
        logger.info(f"[VAL] Epoch {epoch} - Mean flow loss = {mean_flow_loss:.6f}, Mean static loss = {mean_static_loss:.6f}, Mean total loss = {mean_total_loss:.6f} (skipped={skip_loss_computation})")
        try:
            accelerator.log({
                "validation/mean_flow_loss": mean_flow_loss,
                "validation/mean_static_loss": mean_static_loss,
                "validation/mean_total_loss": mean_total_loss,
            })
        except Exception:
            pass

    # ========================================================================
    # 2) Run pipeline inference to collect predictions and compute metrics
    # ========================================================================
    predicted_images = []
    ground_truth_images = []
    original_images = []
    initial_edit_region_masks = []
    final_edit_region_masks = []
    prompts_list = []

    # Prepare visualization folder and overwrite if exists
    viz_root = (
        os.path.join(args.output_dir, "test_visualization")
        if is_final_validation
        else os.path.join(args.output_dir, "validation_visualization")
    )
    # Overwrite entire folder each run to meet requirement
    if os.path.exists(viz_root):
        try:
            shutil.rmtree(viz_root)
        except Exception:
            pass
    os.makedirs(viz_root, exist_ok=True)

    # Move pipeline to device/dtype and disable progress bar where possible
    try:
        if device is not None and torch_dtype is not None:
            pipeline = pipeline.to(device, dtype=torch_dtype)
        elif device is not None:
            pipeline = pipeline.to(device)
    except Exception:
        pass
    try:
        pipeline.set_progress_bar_config(disable=True)
    except Exception:
        pass

    # generator from seed if available
    seed = getattr(args, "seed", None)
    try:
        if seed is not None and device is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        elif seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None
    except Exception:
        generator = None

    # autocast context (best-effort)
    try:
        device_type = device.type if device is not None else None
        autocast_ctx = torch.autocast(device_type) if device_type is not None else contextlib.nullcontext()
    except Exception:
        autocast_ctx = contextlib.nullcontext()

    # Pre-warm metric models (CLIP and LPIPS) to avoid loading during metrics computation
    metric_device = "cuda" if torch.cuda.is_available() else "cpu"
    if CLIP_AVAILABLE:
        try:
            _ = compute_clip_score_batch([Image.new("RGB", (64, 64))], ["test"], device=metric_device)
            logger.info(f"[{'TEST' if is_final_validation else 'VAL'}] CLIP model pre-warmed on {metric_device}")
        except Exception:
            pass
    if LPIPS_AVAILABLE:
        try:
            _ = get_lpips_model(metric_device)
            logger.info(f"[{'TEST' if is_final_validation else 'VAL'}] LPIPS model pre-warmed on {metric_device}")
        except Exception:
            pass

    # Determine total samples to process
    total_samples = len(dataset)
    if max_val_samples is not None:
        total_samples = min(max_val_samples, total_samples)
    
    logger.info(f"[{'TEST' if is_final_validation else 'VAL'}] Running inference on {total_samples} samples with {num_inference_steps} steps...")

    # iterate HF dataset with progress bar
    desc = "Testing" if is_final_validation else f"Validation (epoch {epoch})"
    for i, sample in enumerate(tqdm(dataset, desc=desc, total=total_samples)):
        # Stop if we've processed enough samples
        if i >= total_samples:
            break
            
        # load images and prompt
        try:
            edit_instruction = sample[args.edit_instruction_column]
        except Exception:
            edit_instruction = sample.get(args.edit_instruction_column, None)
        
        # Build enhanced prompt from the short edit instruction
        prompt = build_enhanced_prompt(edit_instruction) if edit_instruction else None

        orig_image = _maybe_load_image(sample.get(args.original_image_column))
        initial_edit_region_mask = _maybe_load_image(sample.get(args.initial_edit_region_mask_column))
        final_edit_region_mask = _maybe_load_image(sample.get(args.final_edit_region_mask_column))
        gt_image = _maybe_load_image(sample.get(args.edited_image_column))

        # prepare call args
        call_args = {}
        call_args["original_image"] = orig_image
        call_args["initial_edit_region_mask"] = initial_edit_region_mask
        call_args["final_edit_region_mask"] = final_edit_region_mask
        call_args["num_inference_steps"] = num_inference_steps  # Use reduced steps for faster inference
        
        # Get ground truth dimensions to request same size from pipeline
        # This attempts to make the pipeline output the same size as the ground truth
        gt_height, gt_width = None, None
        if gt_image is not None:
            try:
                gt_width, gt_height = gt_image.size  # PIL Image size is (width, height)
            except Exception:
                pass
        
        # Pass height and width to pipeline to try to get matching size
        # Also disable _auto_resize so the pipeline doesn't resize to PREFERRED_KONTEXT_RESOLUTIONS
        # And set max_area to preserve our requested dimensions
        if gt_height is not None and gt_width is not None:
            call_args["height"] = gt_height
            call_args["width"] = gt_width
            call_args["_auto_resize"] = False  # Prevent resizing to preferred Kontext resolutions
            call_args["max_area"] = gt_height * gt_width  # Set max_area to ground truth area to prevent scaling

        # Precompute embeddings with pipeline.encode_prompt() (no grad)
        with torch.no_grad():
            try:
                prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(prompt, prompt_2=None)
                if device is not None:
                    try:
                        prompt_embeds = prompt_embeds.to(device)
                    except Exception:
                        pass
                    try:
                        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
                    except Exception:
                        pass
            except Exception:
                # If pipeline.encode_prompt signature differs, fall back to passing raw prompt
                prompt_embeds = None
                pooled_prompt_embeds = None
                text_ids = None

        # call pipeline with embeddings if available
        with autocast_ctx:
            try:
                out = pipeline(
                    **call_args,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    generator=generator,
                )
            except Exception:
                # fallback: call with raw prompt
                out = pipeline(**call_args, prompt=prompt, generator=generator)

        # extract image from pipeline output (same logic as snippet)
        try:
            if hasattr(out, "images"):
                pred_img = out.images[0]
            elif isinstance(out, (list, tuple)):
                candidate = out[0]
                if isinstance(candidate, (list, tuple)):
                    pred_img = candidate[0]
                else:
                    pred_img = candidate
            else:
                pred_img = out
        except Exception:
            pred_img = None

        # Retry once if pipeline returned None or a zero image (to save time, don't loop forever)
        if pred_img is None or (hasattr(pred_img, '__array__') and np.array(pred_img).max() == 0):
            try:
                logger.warning(f"Warning: pipeline returned None or zero image for sample {i}, retrying once...")
                out = pipeline(
                    **call_args,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    generator=generator,
                )
                if hasattr(out, "images"):
                    pred_img = out.images[0]
                elif isinstance(out, (list, tuple)):
                    candidate = out[0]
                    if isinstance(candidate, (list, tuple)):
                        pred_img = candidate[0]
                    else:
                        pred_img = candidate
                else:
                    pred_img = out
            except Exception:
                pass  # Give up on retry

        # Append predicted and ground-truth (convert to PIL if needed)
        if not isinstance(pred_img, Image.Image):
            try:
                pred_img = _maybe_load_image(pred_img)
            except Exception:
                pass

        predicted_images.append(pred_img)
        ground_truth_images.append(gt_image)
        original_images.append(orig_image)
        initial_edit_region_masks.append(initial_edit_region_mask)
        final_edit_region_masks.append(final_edit_region_mask)
        # Use original short edit instruction for CLIP score metrics (not enhanced prompt)
        prompts_list.append(edit_instruction if edit_instruction else "")

        # Save visualization for first args.eval_viz_num samples
        try:
            eval_viz_num = int(getattr(args, "eval_viz_num", 10))
        except Exception:
            eval_viz_num = 10

        if i < eval_viz_num:
            sample_dir = os.path.join(viz_root, f"{i}")
            os.makedirs(sample_dir, exist_ok=True)

            # Save original image, final mask, initial mask, prompt, and predicted image
            try:
                if orig_image is not None:
                    orig_image.save(os.path.join(sample_dir, "original_image.png"))
            except Exception:
                pass
            try:
                if final_edit_region_mask is not None:
                    final_edit_region_mask.save(os.path.join(sample_dir, "final_edit_region_mask.png"))
            except Exception:
                pass
            try:
                if initial_edit_region_mask is not None:
                    initial_edit_region_mask.save(os.path.join(sample_dir, "initial_edit_region_mask.png"))
            except Exception:
                pass
            try:
                if pred_img is not None:
                    # Resize predicted image to match ground truth size if they differ
                    # This ensures consistent visualization with the other images
                    pred_img_to_save = pred_img
                    if gt_image is not None:
                        try:
                            gt_w, gt_h = gt_image.size
                            pred_w, pred_h = pred_img.size
                            if (pred_w, pred_h) != (gt_w, gt_h):
                                pred_img_to_save = pred_img.resize((gt_w, gt_h), resample=Image.BILINEAR)
                        except Exception:
                            pass
                    pred_img_to_save.save(os.path.join(sample_dir, "predicted_image.png"))
            except Exception:
                pass
            try:
                if gt_image is not None:
                    gt_image.save(os.path.join(sample_dir, "ground_truth_image.png"))
            except Exception:
                pass
            # save prompt text (enhanced prompt used for model)
            try:
                with open(os.path.join(sample_dir, "prompt.txt"), "w", encoding="utf-8") as pf:
                    pf.write(str(prompt))
            except Exception:
                pass
            # save original edit instruction separately
            try:
                with open(os.path.join(sample_dir, "edit_instruction.txt"), "w", encoding="utf-8") as ef:
                    ef.write(str(edit_instruction) if edit_instruction else "")
            except Exception:
                pass

    # 3) Compute eval metrics
    metrics = None
    try:
        metrics = compute_eval_metrics(
            predicted_images,
            ground_truth_images,
            original_images,
            initial_edit_region_masks,
            final_edit_region_masks,
            prompts_list,
            is_final_validation,
            logger
        )
    except Exception as e:
        logger.warning(f"compute_eval_metrics failed: {e}")
        metrics = None

    # final logging for evaluation metrics + return
    if metrics is not None:
        if is_final_validation:
            logger.info(f"[TEST] Eval metrics: {metrics}")
            try:
                # safe accelerator.log
                accelerator.log({f"test/{k}": v for k, v in metrics.items()})
            except Exception:
                pass
        else:
            logger.info(f"[VAL] Epoch {epoch} Eval metrics: {metrics}")
            try:
                accelerator.log({f"validation/{k}": v for k, v in metrics.items()})
            except Exception:
                pass

    # If this is an intermediate validation (not final test), restore original
    # training modes so that calling code can continue training uninterrupted.
    # We attempt to call .train() on the same modules we set to eval before.
    if not is_final_validation:
        try:
            transformer.train()
        except Exception:
            pass
        try:
            if args.train_text_encoder:
                text_encoder_one.train()
        except Exception:
            pass
    
    return {
        "mean_flow_loss": mean_flow_loss,
        "mean_static_loss": mean_static_loss,
        "mean_total_loss": mean_total_loss,
        "metrics": metrics
    }

if __name__ == "__main__":
    import logging
    from types import SimpleNamespace
    from accelerate import Accelerator
    import torch
    from PIL import Image

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("flux_eval_driver")

    # Minimal args required by run_validation_or_test and Accelerator setup
    args = SimpleNamespace(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-Kontext-dev",
        revision=None,
        variant=None,
        dataset_name="RiddhiCh/dummy-image-editing-dataset",
        dataset_config_name=None,
        original_image_column="original_image",
        final_edit_region_mask_column="final_edit_region_mask",
        edited_image_column="edited_image",
        initial_edit_region_mask_column="initial_edit_region_mask",
        edit_instruction_column="edit_instruction",
        cache_dir=None,
        output_dir="/home/riddhi/LGIE-ER/eval_code_testing",
        logging_dir="logs",
        seed=42,
        train_text_encoder=False,
        eval_viz_num=1,
        repeats=1,
        center_crop=False,
        final_edit_region_mask_probs=[0.1, 0.9, 0.0],
        cache_latents=False,
        vae_encode_mode="mode",
        weighting_scheme="none",
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
        guidance_scale=3.5,
        random_flip=False,
    # Fields used for Accelerator construction
    gradient_accumulation_steps=1,
    # Match typical finetuning runs: use fp16 mixed precision so weights load smaller on GPU.
    mixed_precision="fp16",
    report_to="tensorboard",
    )

    # Construct Accelerator exactly like the finetuning script to match behavior
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS to match training script behavior
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Determine weight dtype as in the training script
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    # Load the HuggingFace dataset requested by the user
    from datasets import load_dataset

    hf_dataset = load_dataset("RiddhiCh/dummy-image-editing-dataset", cache_dir=args.cache_dir)

    # Trim every split to at most 10 samples to speed up dataset creation and evaluation
    trimmed_dataset = {}
    try:
        for split_name in hf_dataset.keys():
            split_ds = hf_dataset[split_name]
            n = min(10, len(split_ds))
            trimmed_dataset[split_name] = split_ds.select(range(n))
    except Exception:
        # If select fails for any reason, fall back to slicing via list conversion
        for split_name in hf_dataset.keys():
            split_ds = hf_dataset[split_name]
            n = min(10, len(split_ds))
            trimmed_dataset[split_name] = split_ds.select(list(range(n)))

    # Load the pipeline (no LoRA adapters)
    try:
        pipeline = FluxKontextPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
            device_map=str(accelerator.device),
        )
    except Exception as e:
        logger.exception("Failed to load pipeline; aborting driver.")
        raise

    # Import DreamBoothDataset and collate_fn from training script to build PyTorch dataloaders
    try:
        from eval_code_testing.flux_kontext_lora_finetuning_testing import DreamBoothDataset, collate_fn
    except Exception as e:
        logger.exception("Failed to import DreamBoothDataset from training script")
        raise

    # Build DreamBoothDataset instances for validation and test splits and corresponding dataloaders
    buckets = [(512, 512)]
    val_dataset = DreamBoothDataset(
        split="validation",
        pipeline=pipeline,
        accelerator=accelerator,
        pipeline_args=None,
        torch_dtype=weight_dtype,
        repeats=1,
        center_crop=False,
        buckets=buckets,
        args=args,
        dataset=trimmed_dataset
    )
    test_dataset = DreamBoothDataset(
        split="test",
        pipeline=pipeline,
        accelerator=accelerator,
        pipeline_args=None,
        torch_dtype=weight_dtype,
        repeats=1,
        center_crop=False,
        buckets=buckets,
        args=args,
        dataset=trimmed_dataset
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, collate_fn=lambda examples: collate_fn(examples), num_workers=0
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, collate_fn=lambda examples: collate_fn(examples), num_workers=0
    )

    # Run intermediate validation (is_final_validation=False)
    try:
        logger.info("Running intermediate validation (is_final_validation=False)")
        res = run_validation_or_test(
            pipeline=pipeline,
            args=args,
            accelerator=accelerator,
            is_final_validation=False,
            epoch=0,
            torch_dtype=weight_dtype,
            dataloader=val_dataloader,
            dataset=trimmed_dataset["validation"],
            logger=logger,
        )
        logger.info(f"Intermediate validation result: {res}")
    except Exception:
        logger.exception("Intermediate validation failed")

    # Run final test (is_final_validation=True)
    try:
        logger.info("Running final test (is_final_validation=True)")
        res = run_validation_or_test(
            pipeline=pipeline,
            args=args,
            accelerator=accelerator,
            is_final_validation=True,
            epoch=None,
            torch_dtype=weight_dtype,
            dataloader=test_dataloader,
            dataset=trimmed_dataset["test"],
            logger=logger,
        )
        logger.info(f"Final test result: {res}")
    except Exception:
        logger.exception("Final test failed")
