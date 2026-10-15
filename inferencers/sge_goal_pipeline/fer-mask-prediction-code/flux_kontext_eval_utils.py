import os
import shutil
import math
import contextlib
import copy
import numpy as np
from PIL import Image
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import load_image
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3
)

from pipeline_flux_kontext_multi_imgs import FluxKontextPipeline

# try optional imports for stronger metrics; fall back gracefully
try:
    from skimage.metrics import hausdorff_distance as sk_hausdorff
    SKIMAGE_HAUSDORFF = True
except Exception:
    SKIMAGE_HAUSDORFF = False

try:
    from scipy.ndimage import distance_transform_edt
    SCIPY_DTED = True
except Exception:
    SCIPY_DTED = False

# -------------------------
# Helper functions
# -------------------------
def pil_to_numpy(img):
    """PIL.Image -> float32 ndarray in [0,1], shape (H,W,3). Returns None for None."""
    if img is None:
        return None
    if isinstance(img, np.ndarray):
        arr = img
    else:
        img = img.convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr

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


def resize_pred_to_gt(pred_np, gt_np):
    """
    Resize prediction array to match ground truth dimensions.
    This is the correct direction for fair evaluation - predictions should match GT size.
    Uses bilinear interpolation for smooth resizing.
    
    Args:
        pred_np: Prediction as float32 ndarray in [0,1], shape (H,W,3)
        gt_np: Ground truth as float32 ndarray in [0,1], shape (H,W,3)
    
    Returns:
        (pred_np_resized, gt_np) - prediction resized to GT dimensions, GT unchanged
    """
    if pred_np is None or gt_np is None:
        return pred_np, gt_np
    if pred_np.shape[:2] == gt_np.shape[:2]:
        return pred_np, gt_np
    gt_h, gt_w = gt_np.shape[:2]
    pred_img = Image.fromarray((pred_np * 255).astype(np.uint8)).convert("RGB")
    pred_img = pred_img.resize((gt_w, gt_h), resample=Image.BILINEAR)
    pred_np_resized = np.asarray(pred_img).astype(np.float32) / 255.0
    return pred_np_resized, gt_np

def binarize_mask_from_rgb(mask_img, threshold=0.5):
    """
    Convert mask-like image (PIL or numpy RGB) to binary mask (HxW) dtype float32 in {0,1}.
    Uses max channel, normalizes from 0..255 -> 0..1 if needed, and thresholds.
    """
    if mask_img is None:
        return None
    if isinstance(mask_img, np.ndarray):
        m = mask_img.astype(np.float32)
    else:
        m = np.asarray(mask_img.convert("RGB")).astype(np.float32)
    # collapse channels by max (robust to colored masks)
    if m.ndim == 3:
        m_single = np.max(m[..., :3], axis=2)
    else:
        m_single = m
    if m_single.max() > 1.0:
        m_single = m_single / 255.0
    bin_mask = (m_single > threshold).astype(np.float32)
    return bin_mask

def compute_bbox(mask):
    """
    Returns bbox tuple (y0,y1,x0,x1) inclusive coordinates. If empty, returns None.
    """
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return None
    return (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))

def bbox_iou(b1, b2):
    """IoU of two bboxes in (y0,y1,x0,x1) or None -> 0.0 if either None."""
    if b1 is None or b2 is None:
        return 0.0
    y0 = max(b1[0], b2[0]); y1 = min(b1[1], b2[1])
    x0 = max(b1[2], b2[2]); x1 = min(b1[3], b2[3])
    if (y1 < y0) or (x1 < x0):
        inter = 0
    else:
        inter = (y1 - y0 + 1) * (x1 - x0 + 1)
    area1 = (b1[1] - b1[0] + 1) * (b1[3] - b1[2] + 1)
    area2 = (b2[1] - b2[0] + 1) * (b2[3] - b2[2] + 1)
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union

def centroid(mask):
    """Return (y, x) centroid of binary mask; return None if empty."""
    ys, xs = np.nonzero(mask > 0.5)
    if len(ys) == 0:
        return None
    return (ys.mean(), xs.mean())

def safe_div(a, b):
    return a / b if b != 0 else float("nan")

def compute_confusion_from_masks(pred_mask, gt_mask):
    """
    pred_mask, gt_mask: binary float arrays {0,1}
    Returns TP, FP, TN, FN counts (ints)
    """
    pred_bool = pred_mask > 0.5
    gt_bool = gt_mask > 0.5
    TP = int(np.logical_and(pred_bool, gt_bool).sum())
    FP = int(np.logical_and(pred_bool, np.logical_not(gt_bool)).sum())
    FN = int(np.logical_and(np.logical_not(pred_bool), gt_bool).sum())
    TN = int(np.logical_and(np.logical_not(pred_bool), np.logical_not(gt_bool)).sum())
    return TP, FP, TN, FN

def mask_mae(pred, gt):
    """Mean absolute error per pixel (range 0..1)."""
    return float(np.mean(np.abs(pred - gt)))

def compute_hausdorff(pred, gt):
    """
    Returns Hausdorff distance in pixel units if possible.
    If skimage hausdorff available, use it. Otherwise, try distance transform based symmetric directed distances if scipy available.
    Otherwise return NaN.
    """
    pred_bool = (pred > 0.5).astype(np.uint8)
    gt_bool = (gt > 0.5).astype(np.uint8)
    # if either mask empty, return nan
    if pred_bool.sum() == 0 or gt_bool.sum() == 0:
        return float("nan")
    if SKIMAGE_HAUSDORFF:
        try:
            return float(sk_hausdorff(pred_bool, gt_bool))
        except Exception:
            pass
    # fallback using distance transforms (directed distances)
    if SCIPY_DTED:
        try:
            # distances from background to nearest foreground in other mask
            dt_gt = distance_transform_edt(1 - gt_bool)
            dt_pred = distance_transform_edt(1 - pred_bool)
            # for each foreground pixel in pred, distance to nearest gt foreground:
            d1 = dt_gt[pred_bool.astype(bool)]
            d2 = dt_pred[gt_bool.astype(bool)]
            if d1.size == 0 or d2.size == 0:
                return float("nan")
            haus = max(d1.max(), d2.max())
            return float(haus)
        except Exception:
            pass
    return float("nan")

def boundary_f1(pred, gt, tol=3):
    """
    Compute boundary F1 score between binary masks pred/gt.
    tol = tolerance in pixels; if scipy distance transform available we compute boundary pixels and match within tol.
    Returns F1 in [0,1] or NaN if either mask empty.
    """
    pred_bool = (pred > 0.5).astype(np.uint8)
    gt_bool = (gt > 0.5).astype(np.uint8)
    if pred_bool.sum() == 0 and gt_bool.sum() == 0:
        return 1.0
    if pred_bool.sum() == 0 or gt_bool.sum() == 0:
        return float("nan")
    # boundary extraction via XOR of mask and its erosion (simple morphological approach)
    try:
        # use simple 3x3 erosion via convolution without extra deps
        from scipy.ndimage import binary_erosion
        pred_eroded = binary_erosion(pred_bool, structure=np.ones((3,3)))
        gt_eroded = binary_erosion(gt_bool, structure=np.ones((3,3)))
        pred_boundary = pred_bool ^ pred_eroded
        gt_boundary = gt_bool ^ gt_eroded
    except Exception:
        # fallback: approximate boundary as mask XOR mask shifted (approx)
        pred_boundary = pred_bool.copy()
        gt_boundary = gt_bool.copy()
        # we will still use distance transform matching if available
    # if distance transform available, compute matches
    if SCIPY_DTED:
        dt_gt = distance_transform_edt(1 - gt_boundary)
        dt_pred = distance_transform_edt(1 - pred_boundary)
        # true positives on pred boundary: pred boundary pixels within tol of gt boundary
        pred_matches = (dt_gt[pred_boundary.astype(bool)] <= tol).sum()
        gt_matches = (dt_pred[gt_boundary.astype(bool)] <= tol).sum()
        prec = safe_div(pred_matches, pred_boundary.sum())
        rec = safe_div(gt_matches, gt_boundary.sum())
        if math.isnan(prec) or math.isnan(rec) or (prec + rec) == 0:
            return float("nan")
        return 2.0 * prec * rec / (prec + rec)
    else:
        # Without distance transform, compute simple overlap-of-boundaries (very strict)
        inter = (pred_boundary & gt_boundary).sum()
        prec = safe_div(inter, pred_boundary.sum())
        rec = safe_div(inter, gt_boundary.sum())
        if math.isnan(prec) or math.isnan(rec) or (prec + rec) == 0:
            return float("nan")
        return 2.0 * prec * rec / (prec + rec)

# -------------------------
# Main compute function
# -------------------------
def compute_eval_metrics(
    predicted_images,
    ground_truth_images,
    is_final_validation,
    logger,
    mask_threshold=0.5,
    boundary_tol=3,
):
    """
    Evaluate predicted masks (provided as images) against ground-truth masks.

    Inputs:
      - predicted_images: list of PIL or numpy images representing predicted masks (RGB or grayscale).
      - ground_truth_images: list of PIL or numpy images representing GT masks (RGB or grayscale).
      - is_final_validation: bool
      - logger: logging.Logger
      - mask_threshold: float threshold for binarizing masks (0..1)
      - boundary_tol: tolerance in pixels for boundary matching

    Returns:
      - aggregated dict with mean/std for each metric (keys like "mask_iou_mean", "mask_iou_std", ...)
    """
    n = len(predicted_images)
    assert n == len(ground_truth_images), "Input lists must match length"

    per_sample = {
        "iou": [],
        "dice": [],
        "precision": [],
        "recall": [],
        "pixel_accuracy": [],
        "mae": [],
        "pred_area": [],
        "gt_area": [],
        "area_ratio": [],  # pred_area / (gt_area + eps)
        "centroid_distance": [],  # Euclidean distance in pixels (nan if missing)
        "bbox_iou": [],
        "hausdorff": [],
        "boundary_f1": [],
    }

    for i in range(n):
        pred_img = predicted_images[i]
        gt_img = ground_truth_images[i]

        pred_np = pil_to_numpy(pred_img) if pred_img is not None else None
        gt_np = pil_to_numpy(gt_img) if gt_img is not None else None

        if pred_np is None or gt_np is None:
            logger.warning(f"[MASK METRICS] missing pred or gt for sample {i}; skipping sample.")
            # append NaNs to keep lengths consistent
            for k in per_sample:
                per_sample[k].append(float("nan"))
            continue

        # Resize prediction to GT size (correct direction for fair evaluation)
        # GT is the ground truth, so prediction should be resized to match it
        pred_np, gt_np = resize_pred_to_gt(pred_np, gt_np)
        h, w = gt_np.shape[:2]

        # Convert to binary masks
        pred_mask = binarize_mask_from_rgb((pred_np * 255).astype(np.uint8), threshold=mask_threshold)
        gt_mask = binarize_mask_from_rgb((gt_np * 255).astype(np.uint8), threshold=mask_threshold)

        # Final shape check (should always pass after resize_pred_to_gt)
        if pred_mask.shape != gt_mask.shape:
            # Fallback: resize prediction mask to GT mask dimensions (using NEAREST for binary masks)
            logger.warning(f"[Sample {i}] Mask shape mismatch after resize: pred={pred_mask.shape}, gt={gt_mask.shape}. Applying fallback resize.")
            pred_img_resized = Image.fromarray((pred_mask * 255).astype(np.uint8)).resize((w, h), resample=Image.NEAREST)
            pred_mask = (np.asarray(pred_img_resized).astype(np.float32) / 255.0) > 0.5
            pred_mask = pred_mask.astype(np.float32)

        # compute confusion
        TP, FP, TN, FN = compute_confusion_from_masks(pred_mask, gt_mask)
        total = TP + FP + TN + FN

        # pixel accuracy
        pixel_acc = safe_div(TP + TN, total) if total > 0 else float("nan")
        # precision and recall
        precision = safe_div(TP, (TP + FP))
        recall = safe_div(TP, (TP + FN))
        # dice (F1)
        dice = safe_div(2 * TP, (2 * TP + FP + FN))
        # iou
        iou = safe_div(TP, (TP + FP + FN))
        # mae
        mae_v = mask_mae(pred_mask, gt_mask)
        # areas
        pred_area = float(pred_mask.sum())
        gt_area = float(gt_mask.sum())
        area_ratio = safe_div(pred_area, gt_area) if gt_area > 0 else float("nan")

        # centroid distance in pixels
        c_pred = centroid(pred_mask)
        c_gt = centroid(gt_mask)
        if c_pred is None or c_gt is None:
            centroid_dist = float("nan")
        else:
            centroid_dist = float(math.hypot(c_pred[0] - c_gt[0], c_pred[1] - c_gt[1]))

        # bbox iou
        bbox_pred = compute_bbox(pred_mask)
        bbox_gt = compute_bbox(gt_mask)
        b_iou = bbox_iou(bbox_pred, bbox_gt)

        # hausdorff (pixels)
        hdist = compute_hausdorff(pred_mask, gt_mask)

        # boundary f1
        b_f1 = boundary_f1(pred_mask, gt_mask, tol=boundary_tol)

        # append
        per_sample["iou"].append(float(iou) if (not math.isnan(iou)) else float("nan"))
        per_sample["dice"].append(float(dice) if (not math.isnan(dice)) else float("nan"))
        per_sample["precision"].append(float(precision) if (not math.isnan(precision)) else float("nan"))
        per_sample["recall"].append(float(recall) if (not math.isnan(recall)) else float("nan"))
        per_sample["pixel_accuracy"].append(float(pixel_acc) if (not math.isnan(pixel_acc)) else float("nan"))
        per_sample["mae"].append(float(mae_v))
        per_sample["pred_area"].append(pred_area)
        per_sample["gt_area"].append(gt_area)
        per_sample["area_ratio"].append(float(area_ratio) if (not math.isnan(area_ratio)) else float("nan"))
        per_sample["centroid_distance"].append(float(centroid_dist) if (not math.isnan(centroid_dist)) else float("nan"))
        per_sample["bbox_iou"].append(float(b_iou))
        per_sample["hausdorff"].append(float(hdist) if (not math.isnan(hdist)) else float("nan"))
        per_sample["boundary_f1"].append(float(b_f1) if (not math.isnan(b_f1)) else float("nan"))

    # Aggregate per metric (mean, std) while safely ignoring NaNs
    def safe_stats(lst):
        arr = np.array([x for x in lst if (x is not None and not (isinstance(x, float) and math.isnan(x)))], dtype=float)
        if arr.size == 0:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

    aggregated = {}
    for k, v in per_sample.items():
        stats = safe_stats(v)
        aggregated[f"mask_{k}_mean"] = stats["mean"]
        aggregated[f"mask_{k}_std"] = stats["std"]

    # Logging with prefix
    prefix = "test" if is_final_validation else "val"
    logger.info(f"[{prefix.upper()}] Mask-prediction aggregated metrics:")
    # human-friendly selection of keys to log
    keys_to_log = [
        ("mask_iou_mean", "mask_iou_mean"),
        ("mask_dice_mean", "mask_dice_mean"),
        ("mask_precision_mean", "mask_precision_mean"),
        ("mask_recall_mean", "mask_recall_mean"),
        ("mask_pixel_accuracy_mean", "mask_pixel_accuracy_mean"),
        ("mask_mae_mean", "mask_mae_mean"),
        ("mask_centroid_distance_mean", "mask_centroid_distance_mean"),
        ("mask_bbox_iou_mean", "mask_bbox_iou_mean"),
        ("mask_hausdorff_mean", "mask_hausdorff_mean"),
        ("mask_boundary_f1_mean", "mask_boundary_f1_mean"),
    ]
    for key_display, key in keys_to_log:
        v = aggregated.get(key, float("nan"))
        try:
            logger.info(f"{prefix}/{key}: {v:.6f}")
        except Exception:
            logger.info(f"{prefix}/{key}: {v}")

    # Optionally, return both aggregated and per-sample (so caller can save per-sample arrays)
    return {"aggregated": aggregated, "per_sample": per_sample}

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
    max_samples: int = -1,
    skip_loss: bool = False,
):
    """
    Runs validation or final test.

    Parameters
    ----------
    pipeline : FluxKontextPipeline-like object (must provide .encode_prompt and .__call__)
    args : argparse-like object (uses fields referenced in the training script)
    accelerator : huggingface Accelerator instance
    is_final_validation : bool (True = testing)
    epoch : int or None (None when testing)
    torch_dtype : torch.dtype used for pipeline/device
    dataloader : PyTorch DataLoader (batch_size can be > 1 for faster loss computation)
    dataset : Hugging Face dataset (iterable of samples). Columns:
              args.original_image_column, args.initial_edit_region_mask_column,
              args.final_edit_region_mask_column, args.edit_instruction_column,
              args.edited_image_column
    logger : logging.Logger instance
    max_samples : int, optional
        Maximum number of samples to process. -1 means all samples. Default is -1.
    skip_loss : bool, optional
        If True, skip loss computation to speed up validation/testing. Default is False.
    Returns
    -------
    dict with {
        'mean_flow_loss': float,
        'mean_dice_loss': float,
        'mean_bce_loss': float,
        'mean_chan_consistency': float,
        'mean_binarize_reg': float,
        'mean_total_loss': float,
        'metrics': <whatever compute_eval_metrics returns or None>
    }
    """

    # local helpers
    def _maybe_load_image(x):
        # Accept PIL.Image, path string, numpy array, or HF-style dict
        # Always returns an RGB PIL Image (3 channels) or None.
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
        # Last resort: try converting whatever it is
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        return x

    # 1) Compute per-sample loss over dataloader (mimic training forward)
    # Skip this entire section if skip_loss is True
    mean_total_loss = float("nan")
    
    device = None
    try:
        device = accelerator.device
    except Exception:
        device = None

    # Extract sub-modules needed for both loss computation and inference
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

    if not skip_loss:
        # Load scheduler for loss computation
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        
        # Helper functions 
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
            returns : torch.Tensor (B, C_img, H_img, W_img) in image-space (pixel range depends on VAE)
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

        # --- hyperparams for auxiliary mask loss (same as training) ---
        t_rec_max = getattr(args, "t_rec_max", 0.35)  # apply mask losses only when sigmas < t_rec_max
        p_rec_fraction = getattr(args, "p_rec_fraction", 1.0)  # fraction of eligible samples to decode
        LAMBDA_DICE = getattr(args, "lambda_dice", 0)
        LAMBDA_BCE = getattr(args, "lambda_bce", 0)
        LAMBDA_CHAN = getattr(args, "lambda_channel", 25)
        LAMBDA_BINARIZE = getattr(args, "lambda_binarize", 0)

        # Lists to track all 5 loss types
        flow_losses = []
        dice_losses = []
        bce_losses = []
        chan_consistency_losses = []
        binarize_reg_losses = []

        with torch.no_grad():
            for batch in dataloader:
                # Follow training forward exactly (batch_size=1 expected)
                # encode prompts
                prompts = batch["prompts"]
                # re-use pipeline.encode_prompt (precomputed cached embeddings are not used here)
                prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(prompts, prompt_2=None)
                # move embeddings to device if available
                if device is not None:
                    try:
                        prompt_embeds = prompt_embeds.to(device)
                    except Exception:
                        pass
                    try:
                        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
                    except Exception:
                        pass
                    try:
                        text_ids = text_ids.to(device)
                    except Exception:
                        pass

                # latents (use same vae encoding logic as training)
                if args.cache_latents:
                    # If cache_latents were used in training, the dataloader might already provide
                    # latents; but we keep same codepath as non-cached for safety.
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

                # choose sample vs mode
                if args.vae_encode_mode == "sample":
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    original_image_model_input = vae.encode(original_image_pixel_values).latent_dist.sample()
                    initial_edit_region_mask_model_input = vae.encode(initial_edit_region_mask_pixel_values).latent_dist.sample()
                else:
                    model_input = vae.encode(pixel_values).latent_dist.mode()
                    original_image_model_input = vae.encode(original_image_pixel_values).latent_dist.mode()
                    initial_edit_region_mask_model_input = vae.encode(initial_edit_region_mask_pixel_values).latent_dist.mode()

                # same preprocessing as training
                model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                model_input = model_input.to(dtype=torch_dtype)
                original_image_model_input = (original_image_model_input - vae_config_shift_factor) * vae_config_scaling_factor
                original_image_model_input = original_image_model_input.to(dtype=torch_dtype)
                initial_edit_region_mask_model_input = (initial_edit_region_mask_model_input - vae_config_shift_factor) * vae_config_scaling_factor
                initial_edit_region_mask_model_input = initial_edit_region_mask_model_input.to(dtype=torch_dtype)

                vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

                latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    accelerator.device,
                    model_input.dtype,
                )

                original_image_latents_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    original_image_model_input.shape[0],
                    original_image_model_input.shape[2] // 2,
                    original_image_model_input.shape[3] // 2,
                    accelerator.device,
                    original_image_model_input.dtype,
                )
                original_image_latents_ids[..., 0] = 1
                latent_image_ids = torch.cat([latent_image_ids, original_image_latents_ids], dim=0)
                
                initial_edit_region_mask_latents_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    initial_edit_region_mask_model_input.shape[0],
                    initial_edit_region_mask_model_input.shape[2] // 2,
                    initial_edit_region_mask_model_input.shape[3] // 2,
                    accelerator.device,
                    initial_edit_region_mask_model_input.dtype,
                )
                initial_edit_region_mask_latents_ids[..., 0] = 2
                latent_image_ids = torch.cat([latent_image_ids, initial_edit_region_mask_latents_ids], dim=0)

                # sample noise
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = FluxKontextPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )
                orig_inp_shape = packed_noisy_model_input.shape

                packed_original_image_input = FluxKontextPipeline._pack_latents(
                    original_image_model_input,
                    batch_size=original_image_model_input.shape[0],
                    num_channels_latents=original_image_model_input.shape[1],
                    height=original_image_model_input.shape[2],
                    width=original_image_model_input.shape[3],
                )
                packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_original_image_input], dim=1)
                
                packed_initial_edit_region_mask_input = FluxKontextPipeline._pack_latents(
                    initial_edit_region_mask_model_input,
                    batch_size=initial_edit_region_mask_model_input.shape[0],
                    num_channels_latents=initial_edit_region_mask_model_input.shape[1],
                    height=initial_edit_region_mask_model_input.shape[2],
                    width=initial_edit_region_mask_model_input.shape[3],
                )
                packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_initial_edit_region_mask_input], dim=1)

                # guidance handling
                guidance = None
                try:
                    has_guidance = unwrap_model(transformer).config.guidance_embeds
                except Exception:
                    # fallback if unwrap_model isn't available in scope
                    try:
                        has_guidance = getattr(transformer.config, "guidance_embeds", False)
                    except Exception:
                        has_guidance = False

                if has_guidance:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])

                # model prediction - keep same scaling of timestep as in training
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

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                target = noise - model_input

                # flow matching loss (flow_loss)
                flow_loss_per_example = torch.mean(
                    (weighting.float() * (model_pred_latents.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                flow_loss = flow_loss_per_example.mean()
                flow_losses.append(flow_loss.detach().cpu().item())

                # -------------------------
                # Mask-specific auxiliary losses (apply only for easy t's)
                # -------------------------
                # predicted_clean_latents is latent-space one-step recon:
                predicted_clean_latents = noise - model_pred_latents  # shape: (B, C_latent, H_latent, W_latent)

                # compute per-sample t-values from sigmas
                t_vals = sigmas.view(sigmas.shape[0], -1)[:, 0].detach()  # shape (B,)

                # select only those samples we consider "easy" to reconstruct (sigmas < t_rec_max)
                rec_mask = t_vals < t_rec_max  # boolean mask length B
                rec_idx = rec_mask.nonzero(as_tuple=False).view(-1)

                # Only compute heavy ops if there are any eligible samples
                if rec_idx.numel() > 0:
                    # optionally subsample eligible indices so we don't decode entire batch
                    keep_n = max(1, int(p_rec_fraction * rec_idx.numel()))
                    if rec_idx.numel() > keep_n:
                        perm = torch.randperm(rec_idx.numel(), device=rec_idx.device)
                        rec_idx = rec_idx[perm[:keep_n]]

                    # gather selected predicted / gt latents
                    pred_sel_lat = predicted_clean_latents[rec_idx]  # (k, C_lat, H_lat, W_lat)
                    gt_sel_lat = model_input[rec_idx]  # (k, C_lat, H_lat, W_lat)

                    # decode to image-space using VAE decode
                    pred_sel_img = decode_latents(pred_sel_lat)  # (k, C_img, H_img, W_img)
                    gt_sel_img = decode_latents(gt_sel_lat)  # (k, C_img, H_img, W_img)

                    # normalize GT -> [0,1] per-sample
                    gt_min = gt_sel_img.amin(dim=[1, 2, 3], keepdim=True)
                    gt_max = gt_sel_img.amax(dim=[1, 2, 3], keepdim=True)
                    gt_normalized = (gt_sel_img - gt_min) / (gt_max - gt_min + 1e-8)

                    # predicted probabilities via sigmoid (image-space)
                    pred_probs = torch.sigmoid(pred_sel_img)  # (k, C_img, H, W)
                    pred_probs_chan_mean = pred_probs.mean(dim=1, keepdim=True)  # (k,1,H,W)
                    gt_mask = gt_normalized.mean(dim=1, keepdim=True)  # (k,1,H,W)

                    # Dice loss (smooth)
                    smooth = 1e-6
                    intersection = (pred_probs_chan_mean * gt_mask).sum(dim=[1, 2, 3])
                    denom = pred_probs_chan_mean.sum(dim=[1, 2, 3]) + gt_mask.sum(dim=[1, 2, 3])
                    dice_score = (2.0 * intersection + smooth) / (denom + smooth)
                    dice_loss = 1.0 - dice_score.mean()
                    dice_losses.append(dice_loss.detach().cpu().item())

                    # BCE loss (pixel-wise)
                    bce_loss = torch.nn.functional.binary_cross_entropy(
                        pred_probs_chan_mean, gt_mask, reduction="mean"
                    )
                    bce_losses.append(bce_loss.detach().cpu().item())

                    # Channel-consistency loss on image-space (encourages R=G=B)
                    if pred_sel_img.shape[1] >= 3:
                        r, g, b = (
                            pred_sel_img[:, 0:1, :, :],
                            pred_sel_img[:, 1:2, :, :],
                            pred_sel_img[:, 2:3, :, :],
                        )
                        chan_consistency = ((r - g) ** 2 + (r - b) ** 2 + (g - b) ** 2).mean()
                    else:
                        chan_consistency = torch.tensor(0.0, device=pred_sel_img.device, dtype=pred_sel_img.dtype)
                    chan_consistency_losses.append(chan_consistency.detach().cpu().item())

                    # Binarization regularizer: p*(1-p)
                    binarize_reg = (pred_probs_chan_mean * (1.0 - pred_probs_chan_mean)).mean()
                    binarize_reg_losses.append(binarize_reg.detach().cpu().item())

    # mean losses and logging
    if not skip_loss:
        mean_flow_loss = float(np.mean(flow_losses)) if len(flow_losses) > 0 else float("nan")
        mean_dice_loss = float(np.mean(dice_losses)) if len(dice_losses) > 0 else float("nan")
        mean_bce_loss = float(np.mean(bce_losses)) if len(bce_losses) > 0 else float("nan")
        mean_chan_consistency = float(np.mean(chan_consistency_losses)) if len(chan_consistency_losses) > 0 else float("nan")
        mean_binarize_reg = float(np.mean(binarize_reg_losses)) if len(binarize_reg_losses) > 0 else float("nan")
        
        # Compute total loss with same weighting as training:
        # total_loss = flow_loss + (LAMBDA_BCE * bce + LAMBDA_DICE * dice + LAMBDA_CHAN * chan + LAMBDA_BINARIZE * binarize)
        mean_mask_loss = (
            LAMBDA_BCE * mean_bce_loss
            + LAMBDA_DICE * mean_dice_loss
            + LAMBDA_CHAN * mean_chan_consistency
            + LAMBDA_BINARIZE * mean_binarize_reg
        ) if not (np.isnan(mean_bce_loss) or np.isnan(mean_dice_loss) or np.isnan(mean_chan_consistency) or np.isnan(mean_binarize_reg)) else 0.0
        mean_total_loss = mean_flow_loss + mean_mask_loss

        # Logging: different message for validation vs final test
        if is_final_validation:
            logger.info(f"\n[TEST] Loss breakdown over dataloader:")
            logger.info(f"  flow_loss:        {mean_flow_loss:.6f}")
            logger.info(f"  dice_loss:        {mean_dice_loss:.6f}")
            logger.info(f"  bce_loss:         {mean_bce_loss:.6f}")
            logger.info(f"  chan_consistency: {mean_chan_consistency:.6f}")
            logger.info(f"  binarize_reg:     {mean_binarize_reg:.6f}")
            logger.info(f"  total_loss:       {mean_total_loss:.6f}")
            try:
                accelerator.log({
                    "test/flow_loss": mean_flow_loss,
                    "test/dice_loss": mean_dice_loss,
                    "test/bce_loss": mean_bce_loss,
                    "test/chan_consistency": mean_chan_consistency,
                    "test/binarize_reg": mean_binarize_reg,
                    "test/total_loss": mean_total_loss,
                })
            except Exception:
                pass
        else:
            logger.info(f"\n[VAL] Epoch {epoch} - Loss breakdown over dataloader:")
            logger.info(f"  flow_loss:        {mean_flow_loss:.6f}")
            logger.info(f"  dice_loss:        {mean_dice_loss:.6f}")
            logger.info(f"  bce_loss:         {mean_bce_loss:.6f}")
            logger.info(f"  chan_consistency: {mean_chan_consistency:.6f}")
            logger.info(f"  binarize_reg:     {mean_binarize_reg:.6f}")
            logger.info(f"  total_loss:       {mean_total_loss:.6f}")
            try:
                accelerator.log({
                    "validation/flow_loss": mean_flow_loss,
                    "validation/dice_loss": mean_dice_loss,
                    "validation/bce_loss": mean_bce_loss,
                    "validation/chan_consistency": mean_chan_consistency,
                    "validation/binarize_reg": mean_binarize_reg,
                    "validation/total_loss": mean_total_loss,
                })
            except Exception:
                pass
    else:
        mean_flow_loss = float("nan")
        mean_dice_loss = float("nan")
        mean_bce_loss = float("nan")
        mean_chan_consistency = float("nan")
        mean_binarize_reg = float("nan")
        mean_total_loss = float("nan")
        logger.info(f"\n[{'TEST' if is_final_validation else 'VAL'}] Skipping loss computation as requested.")

    # 2) Run pipeline over the HF dataset (one sample at a time) to collect preds and ground truth,
    #    and save visualization for the first args.eval_viz_num samples.
    predicted_images = []
    ground_truth_images = []

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

    # iterate HF dataset
    total_samples = len(dataset) if hasattr(dataset, "__len__") else None
    effective_max = max_samples if max_samples > 0 else (total_samples if total_samples is not None else float('inf'))
    logger.info(f"\n[{'TEST' if is_final_validation else 'VAL'}] Running pipeline inference on {effective_max if effective_max != float('inf') else 'all'} samples...")
    
    for i, sample in enumerate(dataset):
        # Check if we've reached the max_samples limit
        if max_samples > 0 and i >= max_samples:
            logger.info(f"Reached max_samples limit ({max_samples}), stopping inference.")
            break
        
        # load images and prompt
        try:
            prompt = sample[args.edit_instruction_column]
        except Exception:
            prompt = sample.get(args.edit_instruction_column, None)

        orig_image = _maybe_load_image(sample.get(args.original_image_column))
        init_er_mask = _maybe_load_image(sample.get(args.initial_edit_region_mask_column))
        gt_image = _maybe_load_image(sample.get(args.edited_image_column))

        # Get target dimensions from ground truth image to ensure pipeline outputs matching size
        target_height, target_width = None, None
        if gt_image is not None:
            try:
                target_width, target_height = gt_image.size  # PIL uses (width, height)
            except Exception:
                pass
        # Fallback to original image dimensions if GT not available
        if (target_height is None or target_width is None) and orig_image is not None:
            try:
                target_width, target_height = orig_image.size
            except Exception:
                pass

        # prepare call args, copying pipeline_args if present
        call_args = {}
        pipeline_args = {"prompt": None}  # avoid mutating user's pipeline_args if passed externally
        # If user expects other pipeline args, they can pass them via args or outer scope; keep minimal here.
        # Avoid passing raw prompt string to pipeline kwargs as in sample snippet
        call_args["original_image"] = orig_image
        call_args["initial_edit_region_mask"] = init_er_mask
        
        # Pass target dimensions to pipeline to generate output at the correct size
        # Use _auto_resize=False to prevent pipeline from selecting a different resolution
        # Set max_area to match GT area to prevent the pipeline from scaling up dimensions
        if target_height is not None and target_width is not None:
            call_args["height"] = target_height
            call_args["width"] = target_width
            call_args["_auto_resize"] = False
            call_args["max_area"] = target_height * target_width  # Prevent scaling to default 1024² area

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

        # Retry loop if pipeline returned None or a zero image
        while True:
            try:
                if pred_img is None or np.array(pred_img).max() == 0:
                    logger.warning("Warning: pipeline returned None or a zero image, retrying...")
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
                else:
                    break
            except Exception:
                break

        # Append predicted and ground-truth (convert to PIL if needed)
        if not isinstance(pred_img, Image.Image):
            try:
                pred_img = _maybe_load_image(pred_img)
            except Exception:
                pass

        # Resize prediction to match GT dimensions if they differ (for fair metric computation)
        # This handles cases where pipeline couldn't output exact target dimensions
        if pred_img is not None and gt_image is not None:
            try:
                pred_w, pred_h = pred_img.size
                gt_w, gt_h = gt_image.size
                if pred_w != gt_w or pred_h != gt_h:
                    logger.info(f"[Sample {i}] Resizing prediction from ({pred_w}x{pred_h}) to GT size ({gt_w}x{gt_h})")
                    pred_img = pred_img.resize((gt_w, gt_h), resample=Image.BILINEAR)
            except Exception as resize_err:
                logger.warning(f"[Sample {i}] Failed to resize prediction to GT size: {resize_err}")

        predicted_images.append(pred_img)
        ground_truth_images.append(gt_image)

        # Save visualization for first args.eval_viz_num samples
        try:
            eval_viz_num = int(getattr(args, "eval_viz_num", 10))
        except Exception:
            eval_viz_num = 10

        if i < eval_viz_num:
            sample_dir = os.path.join(viz_root, f"{i}")
            os.makedirs(sample_dir, exist_ok=True)

            # Save original, current, mask, prompt, and predicted image
            try:
                if orig_image is not None:
                    orig_image.save(os.path.join(sample_dir, "original_image.png"))
            except Exception:
                pass
            try:
                if init_er_mask is not None:
                    init_er_mask.save(os.path.join(sample_dir, "initial_edit_region_mask.png"))
            except Exception:
                pass
            try:
                if pred_img is not None:
                    # pred_img has already been resized to GT dimensions above, so just save it
                    pred_img.save(os.path.join(sample_dir, "predicted_final_edit_region_mask.png"))
            except Exception:
                pass
            try:
                if gt_image is not None:
                    gt_image.save(os.path.join(sample_dir, "ground_truth_final_edit_region_mask.png"))
            except Exception:
                pass
            # save prompt text
            try:
                with open(os.path.join(sample_dir, "prompt.txt"), "w", encoding="utf-8") as pf:
                    pf.write(str(prompt))
            except Exception:
                pass

    # 3) Compute eval metrics
    metrics = None
    try:
        metrics = compute_eval_metrics(
            predicted_images,
            ground_truth_images,
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
        "mean_dice_loss": mean_dice_loss,
        "mean_bce_loss": mean_bce_loss,
        "mean_chan_consistency": mean_chan_consistency,
        "mean_binarize_reg": mean_binarize_reg,
        "mean_total_loss": mean_total_loss,
        "metrics": metrics,
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
        edited_image_column="edit_region_mask",
        edit_instruction_column="edit_instruction",
        cache_dir=None,
        output_dir="/home/riddhi/LGIE-ER/eval_code_testing",
        logging_dir="logs",
        seed=42,
        train_text_encoder=False,
        eval_viz_num=1,
        repeats=1,
        center_crop=False,
        current_image_probs=[0.1, 0.9, 0.0],
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
