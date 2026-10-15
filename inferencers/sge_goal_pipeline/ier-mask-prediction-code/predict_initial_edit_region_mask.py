"""
Initial Edit Region (IER) Mask Prediction Pipeline — V2.

Multi-stage pipeline for predicting binary masks of dynamic objects in robotic
manipulation images.  Dramatically reduces false positives compared to V1 by
adding CLIP-based verification and multi-model detection.

Pipeline stages:
  1. Instruction parsing — extract dynamic object names from the
     manipulation instruction.
  2. Multi-model detection — Florence-2 phrase grounding + open-vocabulary
     detection, AND Grounding DINO open-set detection.
  3. Label-to-object assignment — fuzzy match each detection label to a
     target object.
  4. Per-object NMS — deduplicate overlapping boxes for the same object.
  5. BLIP-2 VQA verification — crop each candidate region, use VQA to
     verify object identity, and discard false positives.
  6. SAM segmentation — generate precise pixel-level masks from verified
     bounding boxes.
  7. Morphological cleanup — fill holes, remove tiny components.

Models (all from HuggingFace Hub):
  * Florence-2-large      — phrase grounding & open-vocab detection
  * Grounding DINO base   — open-set object detection
  * BLIP-2 (OPT-2.7b)    — visual question answering verification
  * SAM ViT-Huge          — segment-anything segmentation
"""

import os
import re
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from scipy import ndimage


# =============================================================================
# Instruction Parsing Utilities
# =============================================================================

def clean_instruction_text(text: str) -> str:
    """
    Clean encoded identifiers in instructions.

    In robotic manipulation datasets, objects may have encoded names like
    'white1fbox' (meaning 'white box #1f'). This function inserts spaces
    around digit sequences embedded within words to normalize them.

    Examples:
        'white1fbox' -> 'white fbox'
        'box2table'  -> 'box table'
    """
    text = re.sub(r'(?<=[a-zA-Z])\d+(?=[a-zA-Z])', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_dynamic_objects(instruction: str) -> list[str]:
    """
    Extract the names/descriptions of dynamic objects from a robotic
    manipulation instruction.

    Returns a list of object-name strings suitable for use as text queries.

    Examples:
        "The robotic arm removes the spoon, the banana, and the blue cube
        from the scene"
        -> ["spoon", "banana", "blue cube"]

        "close white1fbox flap"
        -> ["white fbox flap"]

        "turn lever vertical to front"
        -> ["lever"]
    """
    instruction = instruction.strip()
    instruction_lower = clean_instruction_text(instruction).lower()

    # Remove agent references
    agent_patterns = [
        r'\bthe robotic arm\b', r'\ba robotic arm\b',
        r'\bthe robot\b', r'\ba robot\b', r'\brobot\b',
        r'\brobotic arm\b', r'\bthe arm\b',
    ]
    for p in agent_patterns:
        instruction_lower = re.sub(p, '', instruction_lower).strip()

    # Remove trailing scene/location phrases
    trailing_patterns = [
        r'\s+from\s+the\s+scene\s*$',
        r'\s+from\s+scene\s*$',
        r'\s+off\s+the\s+table\s*$',
        r'\s+off\s+table\s*$',
        r'\s+away\s*$',
    ]
    for p in trailing_patterns:
        instruction_lower = re.sub(p, '', instruction_lower).strip()

    # Common manipulation verbs (as regex patterns)
    verb_patterns = [
        r'removes?', r'picks?\s*up', r'places?', r'moves?',
        r'pushes?', r'pulls?', r'closes?', r'opens?',
        r'turns?', r'flips?', r'rotates?', r'slides?',
        r'lifts?', r'lowers?', r'grasps?', r'grabs?',
        r'drags?', r'drops?', r'puts?', r'takes?',
        r'presses?', r'folds?', r'unfolds?',
        r'squeezes?', r'twists?', r'stacks?', r'unstacks?',
        r'wipes?', r'sweeps?', r'scoops?',
    ]

    for vp in verb_patterns:
        match = re.search(rf'(?:^|\s)({vp})\s+(.*)', instruction_lower)
        if match:
            objects_str = match.group(2).strip()

            # Remove directional / state modifiers at the end
            dir_patterns = [
                r'\s+(?:vertical(?:ly)?|horizontal(?:ly)?)\b.*$',
                r'\s+to\s+(?:the\s+)?(?:left|right|front|back|top|bottom|center|side|middle)\b.*$',
                r'\s+(upwards?|downwards?|forwards?|backwards?|sideways)\b.*$',
                r'\s+(up|down)\s*$',
                r'\s+on\s+top\s+of\b.*$',
                r'\s+next\s+to\b.*$',
                r'\s+near\b.*$',
                r'\s+towards?\b.*$',
                r'\s+into\b.*$',
                r'\s+inside\b.*$',
            ]
            for dp in dir_patterns:
                objects_str = re.sub(dp, '', objects_str).strip()

            # Split comma-separated lists (handle "and")
            objects_str = re.sub(r',?\s+and\s+', ', ', objects_str)
            objects = [o.strip() for o in objects_str.split(',')]

            # Clean each object: remove leading articles
            cleaned = []
            for obj in objects:
                obj = re.sub(r'^(the|a|an)\s+', '', obj).strip()
                if obj and len(obj) > 0:
                    cleaned.append(obj)

            if cleaned:
                return cleaned

    # Fallback: remove stop words and verbs, return remainder
    stop = {
        'the', 'a', 'an', 'to', 'from', 'in', 'on', 'at', 'by', 'with',
        'into', 'onto', 'is', 'are', 'scene', 'it', 'its', 'this', 'that',
        'and', 'or', 'of', 'for',
    }
    verb_words = {
        'remove', 'removes', 'pick', 'picks', 'place', 'places',
        'move', 'moves', 'push', 'pushes', 'pull', 'pulls',
        'close', 'closes', 'open', 'opens', 'turn', 'turns',
        'flip', 'flips', 'rotate', 'rotates', 'slide', 'slides',
        'lift', 'lifts', 'lower', 'lowers', 'grasp', 'grasps',
        'grab', 'grabs', 'drag', 'drags', 'drop', 'drops',
        'put', 'puts', 'take', 'takes', 'press', 'presses',
        'fold', 'folds', 'unfold', 'unfolds', 'squeeze', 'squeezes',
        'twist', 'twists', 'up',
    }

    words = [
        w for w in instruction_lower.split()
        if w not in stop and w not in verb_words
    ]
    if words:
        return [' '.join(words)]
    return [clean_instruction_text(instruction.lower())]


# =============================================================================
# Box Utilities
# =============================================================================

def compute_iou(box_a, box_b):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def nms_per_category(
    boxes: list[list[float]],
    scores: list[float],
    labels: list[str],
    iou_threshold: float = 0.5,
) -> tuple[list[list[float]], list[float], list[str]]:
    """
    Apply Non-Maximum Suppression per object category.

    Groups boxes by their assigned label, then within each group keeps only
    the highest-scoring box when multiple boxes overlap above *iou_threshold*.

    Returns filtered (boxes, scores, labels).
    """
    if not boxes:
        return [], [], []

    # Group by label
    groups: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        groups[label].append(i)

    keep_indices: list[int] = []

    for _label, indices in groups.items():
        # Sort by score descending
        indices = sorted(indices, key=lambda i: scores[i], reverse=True)
        suppressed = set()
        for i, idx_i in enumerate(indices):
            if idx_i in suppressed:
                continue
            keep_indices.append(idx_i)
            for idx_j in indices[i + 1:]:
                if idx_j in suppressed:
                    continue
                if compute_iou(boxes[idx_i], boxes[idx_j]) > iou_threshold:
                    suppressed.add(idx_j)

    # Sort by original order
    keep_indices.sort()
    return (
        [boxes[i] for i in keep_indices],
        [scores[i] for i in keep_indices],
        [labels[i] for i in keep_indices],
    )


def match_label_to_object(label: str, target_objects: list[str]) -> str | None:
    """
    Fuzzy-match a detected label string to one of the target object names.

    Returns the best matching target object, or None if no match is found.
    """
    label_lower = label.lower().strip()

    # Exact match
    for obj in target_objects:
        if label_lower == obj:
            return obj

    # Substring containment (either direction)
    for obj in target_objects:
        if obj in label_lower or label_lower in obj:
            return obj

    # Word overlap
    label_words = set(label_lower.split())
    best_obj = None
    best_overlap = 0
    for obj in target_objects:
        obj_words = set(obj.split())
        overlap = len(label_words & obj_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_obj = obj

    if best_overlap > 0:
        return best_obj

    return None


# =============================================================================
# Mask Post-processing
# =============================================================================

def clean_mask(mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """
    Clean a binary mask: fill small holes and remove tiny components.
    """
    mask_filled = ndimage.binary_fill_holes(mask).astype(np.uint8)
    labeled, num_features = ndimage.label(mask_filled)
    for i in range(1, num_features + 1):
        component = labeled == i
        if np.sum(component) < min_area:
            mask_filled[component] = 0
    return mask_filled


# =============================================================================
# Multi-Stage Predictor
# =============================================================================

class InitialEditRegionMaskPredictor:
    """
    Predicts initial edit region masks using a multi-stage pipeline:

      Florence-2 (phrase grounding)
       + Grounding DINO (open-set detection)
       -> per-object NMS
       -> BLIP-2 VQA verification
       -> SAM segmentation
       -> morphological cleanup

    This pipeline is far more accurate than using Grounding DINO alone because:
    1. Florence-2 grounds phrases in context (understands what the sentence
       refers to, not just individual words).
    2. BLIP-2 VQA verification eliminates false positives by asking
       "Is this a {object}?" for each cropped detection — semantic
       yes/no understanding rather than fragile cosine thresholds.
    3. NMS prevents the same physical object from being counted multiple
       times.
    """

    def __init__(
        self,
        device: str = "cuda",
        # Model IDs
        florence_model_id: str = "microsoft/Florence-2-large",
        gdino_model_id: str = "IDEA-Research/grounding-dino-base",
        blip2_model_id: str = "Salesforce/blip2-opt-2.7b",
        sam_model_id: str = "facebook/sam-vit-huge",
        # Grounding DINO thresholds
        gdino_box_threshold: float = 0.35,
        gdino_text_threshold: float = 0.30,
    ):
        self.device = device
        self.gdino_box_threshold = gdino_box_threshold
        self.gdino_text_threshold = gdino_text_threshold

        # ---- Florence-2 ----
        print(f"  Loading Florence-2 ({florence_model_id})...")
        from transformers import (
            AutoModelForCausalLM,
            AutoProcessor as Florence2AutoProcessor,
        )
        self.florence_processor = Florence2AutoProcessor.from_pretrained(
            florence_model_id, trust_remote_code=True,
        )
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            florence_model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(device)
        self.florence_model.eval()
        print("    Florence-2 loaded.")

        # ---- Grounding DINO ----
        print(f"  Loading Grounding DINO ({gdino_model_id})...")
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )
        self.gdino_processor = AutoProcessor.from_pretrained(gdino_model_id)
        self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            gdino_model_id, torch_dtype=torch.float32,
        ).to(device)
        self.gdino_model.eval()
        print("    Grounding DINO loaded.")

        # ---- BLIP-2 ----
        print(f"  Loading BLIP-2 ({blip2_model_id})...")
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        self.blip2_processor = Blip2Processor.from_pretrained(blip2_model_id)
        self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
            blip2_model_id,
            torch_dtype=torch.float16,
        ).to(device)
        self.blip2_model.eval()
        print("    BLIP-2 loaded.")

        # ---- SAM ----
        print(f"  Loading SAM ({sam_model_id})...")
        from transformers import SamModel, SamProcessor
        self.sam_processor = SamProcessor.from_pretrained(sam_model_id)
        self.sam_model = SamModel.from_pretrained(
            sam_model_id, torch_dtype=torch.float32,
        ).to(device)
        self.sam_model.eval()
        print("    SAM loaded.")

        print("  All models loaded successfully.\n")

    # -----------------------------------------------------------------
    # Florence-2 detection
    # -----------------------------------------------------------------
    def detect_with_florence2(
        self,
        image: Image.Image,
        instruction: str,
        target_objects: list[str],
    ) -> tuple[list[list[float]], list[float], list[str]]:
        """
        Use Florence-2 for:
          a) CAPTION_TO_PHRASE_GROUNDING - grounds phrases in the
             instruction to bounding boxes.
          b) OPEN_VOCABULARY_DETECTION - detects each target object by
             name.

        Returns (boxes, scores, assigned_labels).
        """
        all_boxes: list[list[float]] = []
        all_scores: list[float] = []
        all_labels: list[str] = []

        w, h = image.size

        # --- (a) Caption-to-phrase grounding ---
        task_cpg = "<CAPTION_TO_PHRASE_GROUNDING>"
        prompt_cpg = task_cpg + instruction
        inputs = self.florence_processor(
            text=prompt_cpg, images=image, return_tensors="pt",
        ).to(self.device, torch.float16)

        with torch.no_grad():
            generated_ids = self.florence_model.generate(
                **inputs, max_new_tokens=1024, num_beams=3,
            )
        generated_text = self.florence_processor.batch_decode(
            generated_ids, skip_special_tokens=False,
        )[0]
        parsed = self.florence_processor.post_process_generation(
            generated_text, task=task_cpg, image_size=(w, h),
        )

        cpg_result = parsed.get(task_cpg, {})
        if 'bboxes' in cpg_result and 'labels' in cpg_result:
            for bbox, label in zip(cpg_result['bboxes'], cpg_result['labels']):
                matched_obj = match_label_to_object(label, target_objects)
                if matched_obj is not None:
                    x1, y1, x2, y2 = bbox
                    # Clamp to image bounds
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    if (x2 - x1) > 2 and (y2 - y1) > 2:
                        all_boxes.append([x1, y1, x2, y2])
                        all_scores.append(0.90)
                        all_labels.append(matched_obj)

        print(f"    Florence-2 CPG: {len(all_boxes)} detections")

        # --- (b) Open vocabulary detection per object ---
        task_ovd = "<OPEN_VOCABULARY_DETECTION>"
        count_before = len(all_boxes)
        for obj in target_objects:
            prompt_ovd = task_ovd + obj
            inputs = self.florence_processor(
                text=prompt_ovd, images=image, return_tensors="pt",
            ).to(self.device, torch.float16)

            with torch.no_grad():
                generated_ids = self.florence_model.generate(
                    **inputs, max_new_tokens=1024, num_beams=3,
                )
            generated_text = self.florence_processor.batch_decode(
                generated_ids, skip_special_tokens=False,
            )[0]
            parsed = self.florence_processor.post_process_generation(
                generated_text, task=task_ovd, image_size=(w, h),
            )

            ovd_result = parsed.get(task_ovd, {})
            if 'bboxes' in ovd_result:
                for bbox in ovd_result['bboxes']:
                    x1, y1, x2, y2 = bbox
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    if (x2 - x1) > 2 and (y2 - y1) > 2:
                        all_boxes.append([x1, y1, x2, y2])
                        all_scores.append(0.85)
                        all_labels.append(obj)

        print(
            f"    Florence-2 OVD: "
            f"{len(all_boxes) - count_before} additional detections"
        )

        return all_boxes, all_scores, all_labels

    # -----------------------------------------------------------------
    # Florence-2 standard object detection (for recovery)
    # -----------------------------------------------------------------
    def detect_with_florence2_od(
        self,
        image: Image.Image,
        target_objects: list[str],
    ) -> tuple[list[list[float]], list[float], list[str]]:
        """
        Use Florence-2 standard object detection (<OD>) to find all
        recognisable objects, then match them to target objects.

        This is useful for recovery because <OD> recognises common COCO
        categories ("spoon", "cup", "knife", etc.) that open-set
        detectors sometimes miss.
        """
        w, h = image.size
        task = "<OD>"
        inputs = self.florence_processor(
            text=task, images=image, return_tensors="pt",
        ).to(self.device, torch.float16)

        with torch.no_grad():
            generated_ids = self.florence_model.generate(
                **inputs, max_new_tokens=1024, num_beams=3,
            )
        generated_text = self.florence_processor.batch_decode(
            generated_ids, skip_special_tokens=False,
        )[0]
        parsed = self.florence_processor.post_process_generation(
            generated_text, task=task, image_size=(w, h),
        )

        od_result = parsed.get(task, {})
        boxes, scores, labels = [], [], []
        if 'bboxes' in od_result and 'labels' in od_result:
            for bbox, label in zip(od_result['bboxes'], od_result['labels']):
                matched_obj = match_label_to_object(label, target_objects)
                if matched_obj is not None:
                    x1, y1, x2, y2 = bbox
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    if (x2 - x1) > 2 and (y2 - y1) > 2:
                        boxes.append([x1, y1, x2, y2])
                        scores.append(0.80)
                        labels.append(matched_obj)

        return boxes, scores, labels

    # -----------------------------------------------------------------
    # Florence-2 dense region caption (for recovery)
    # -----------------------------------------------------------------
    def detect_with_florence2_dense_caption(
        self,
        image: Image.Image,
        target_objects: list[str],
    ) -> tuple[list[list[float]], list[float], list[str]]:
        """
        Use Florence-2 ``<DENSE_REGION_CAPTION>`` to caption ALL salient
        regions, then match captions to target objects.

        This is the most robust recovery strategy because it does not
        rely on the detector knowing the object name in advance.
        Instead, it finds all salient regions and describes them,
        then we check if any description matches a target.
        """
        w, h = image.size
        task = "<DENSE_REGION_CAPTION>"
        inputs = self.florence_processor(
            text=task, images=image, return_tensors="pt",
        ).to(self.device, torch.float16)

        with torch.no_grad():
            generated_ids = self.florence_model.generate(
                **inputs, max_new_tokens=1024, num_beams=3,
            )
        generated_text = self.florence_processor.batch_decode(
            generated_ids, skip_special_tokens=False,
        )[0]
        parsed = self.florence_processor.post_process_generation(
            generated_text, task=task, image_size=(w, h),
        )

        drc_result = parsed.get(task, {})
        boxes, scores, labels = [], [], []
        if 'bboxes' in drc_result and 'labels' in drc_result:
            for bbox, caption in zip(
                drc_result['bboxes'], drc_result['labels'],
            ):
                matched_obj = match_label_to_object(
                    caption, target_objects,
                )
                if matched_obj is not None:
                    x1, y1, x2, y2 = bbox
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    if (x2 - x1) > 2 and (y2 - y1) > 2:
                        boxes.append([x1, y1, x2, y2])
                        scores.append(0.80)
                        labels.append(matched_obj)
                        print(
                            f"        DRC matched: '{caption}' -> "
                            f"'{matched_obj}'"
                        )

        return boxes, scores, labels

    # -----------------------------------------------------------------
    # Florence-2 region proposals (class-agnostic, for recovery)
    # -----------------------------------------------------------------
    def get_region_proposals(
        self,
        image: Image.Image,
    ) -> list[list[float]]:
        """
        Use Florence-2 ``<REGION_PROPOSAL>`` to obtain class-agnostic
        bounding box proposals for **all** salient objects in the image.

        Returns a list of ``[x1, y1, x2, y2]`` boxes.  These are not
        filtered by any class — the caller must verify each one.
        """
        w, h = image.size
        task = "<REGION_PROPOSAL>"
        inputs = self.florence_processor(
            text=task, images=image, return_tensors="pt",
        ).to(self.device, torch.float16)

        with torch.no_grad():
            generated_ids = self.florence_model.generate(
                **inputs, max_new_tokens=1024, num_beams=3,
            )
        generated_text = self.florence_processor.batch_decode(
            generated_ids, skip_special_tokens=False,
        )[0]
        parsed = self.florence_processor.post_process_generation(
            generated_text, task=task, image_size=(w, h),
        )

        rp_result = parsed.get(task, {})
        proposals: list[list[float]] = []
        if 'bboxes' in rp_result:
            for bbox in rp_result['bboxes']:
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                if (x2 - x1) > 2 and (y2 - y1) > 2:
                    proposals.append([x1, y1, x2, y2])

        return proposals

    # -----------------------------------------------------------------
    # Florence-2 referring expression segmentation (for recovery)
    # -----------------------------------------------------------------
    def detect_with_referring_expression(
        self,
        image: Image.Image,
        expression: str,
    ) -> list[list[float]]:
        """
        Use Florence-2 ``<REFERRING_EXPRESSION_SEGMENTATION>`` to ground
        a natural-language expression to a region.

        This uses a completely different mechanism from OVD — it
        interprets the expression in the full image context and directly
        produces a segmentation polygon.  We convert the polygon to a
        bounding box.

        Returns a list of ``[x1, y1, x2, y2]`` boxes (usually 0 or 1).
        """
        w, h = image.size
        task = "<REFERRING_EXPRESSION_SEGMENTATION>"
        prompt = task + expression
        inputs = self.florence_processor(
            text=prompt, images=image, return_tensors="pt",
        ).to(self.device, torch.float16)

        with torch.no_grad():
            generated_ids = self.florence_model.generate(
                **inputs, max_new_tokens=1024, num_beams=3,
            )
        generated_text = self.florence_processor.batch_decode(
            generated_ids, skip_special_tokens=False,
        )[0]
        parsed = self.florence_processor.post_process_generation(
            generated_text, task=task, image_size=(w, h),
        )

        result = parsed.get(task, {})
        boxes: list[list[float]] = []

        # Result may contain 'polygons' — convert to bounding boxes
        polygons = result.get('polygons', [])
        for polygon_group in polygons:
            # Each polygon_group is a list of [x,y,x,y,...] coords
            if not polygon_group:
                continue
            coords = polygon_group
            if isinstance(coords[0], list):
                # Nested: list of polygons
                for poly in coords:
                    xs = [poly[i] for i in range(0, len(poly), 2)]
                    ys = [poly[i] for i in range(1, len(poly), 2)]
                    if xs and ys:
                        bx = [
                            max(0, min(xs)), max(0, min(ys)),
                            min(w, max(xs)), min(h, max(ys)),
                        ]
                        if (bx[2] - bx[0]) > 2 and (bx[3] - bx[1]) > 2:
                            boxes.append(bx)
            else:
                # Flat list of coordinates
                xs = [coords[i] for i in range(0, len(coords), 2)]
                ys = [coords[i] for i in range(1, len(coords), 2)]
                if xs and ys:
                    bx = [
                        max(0, min(xs)), max(0, min(ys)),
                        min(w, max(xs)), min(h, max(ys)),
                    ]
                    if (bx[2] - bx[0]) > 2 and (bx[3] - bx[1]) > 2:
                        boxes.append(bx)

        return boxes

    # -----------------------------------------------------------------
    # Grounding DINO detection
    # -----------------------------------------------------------------
    def detect_with_gdino(
        self,
        image: Image.Image,
        target_objects: list[str],
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> tuple[list[list[float]], list[float], list[str]]:
        """
        Detect objects with Grounding DINO.

        Returns (boxes, scores, assigned_labels) where each label is
        matched to the best target object.
        """
        if box_threshold is None:
            box_threshold = self.gdino_box_threshold
        if text_threshold is None:
            text_threshold = self.gdino_text_threshold

        text_prompt = ". ".join(target_objects) + "."

        inputs = self.gdino_processor(
            images=image, text=text_prompt, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.gdino_model(**inputs)

        target_sizes = [image.size[::-1]]  # (H, W)
        results = self.gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]

        raw_labels = results.get("text_labels", results.get("labels", []))
        raw_boxes = (
            results["boxes"].tolist() if len(results["boxes"]) > 0 else []
        )
        raw_scores = (
            results["scores"].tolist() if len(results["scores"]) > 0 else []
        )

        boxes, scores, labels = [], [], []
        for box, score, label in zip(raw_boxes, raw_scores, raw_labels):
            matched_obj = match_label_to_object(label, target_objects)
            if matched_obj is not None:
                boxes.append(box)
                scores.append(score)
                labels.append(matched_obj)

        return boxes, scores, labels

    # -----------------------------------------------------------------
    # BLIP-2 VQA verification
    # -----------------------------------------------------------------
    def _blip2_ask(self, crop: Image.Image, question: str) -> str:
        """
        Ask BLIP-2 a question about a cropped image region.

        Uses the BLIP-2 VQA prompt format: ``Question: {q} Answer:``.
        Returns the generated answer string (lowercased, stripped).
        """
        # Ensure RGB
        crop = crop.convert("RGB")

        prompt = f"Question: {question} Answer:"

        inputs = self.blip2_processor(
            images=crop, text=prompt, return_tensors="pt",
        ).to(self.device, torch.float16)

        with torch.no_grad():
            generated_ids = self.blip2_model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
            )
        answer = self.blip2_processor.batch_decode(
            generated_ids, skip_special_tokens=True,
        )[0].strip().lower()
        return answer

    def verify_with_blip2(
        self,
        image: Image.Image,
        boxes: list[list[float]],
        labels: list[str],
        target_objects: list[str],
    ) -> tuple[list[list[float]], list[float], list[str]]:
        """
        Verify each detection using BLIP-2 Visual Question Answering.

        Fully open-vocabulary — no hardcoded synonyms or keyword lists.

        For each bounding box crop, asks BLIP-2 two complementary
        questions and combines the evidence:

          1. **Constrained choice** — "Which of the following best
             describes this object: {obj1}, {obj2}, …, or none of
             these?"  Forces the model to pick from the target objects
             or explicitly reject.
          2. **Open caption** — "What is the main object in this
             image?"  Then asks BLIP-2 *itself* whether the caption
             and any target object refer to the same thing (semantic
             matching via VQA, not string matching).

        A detection is accepted if *either* question yields a match.

        Returns (verified_boxes, confidence_scores, verified_labels).
        """
        if not boxes:
            return [], [], []

        w, h = image.size

        # Build the constrained-choice option string once
        choices_str = ", ".join(target_objects) + ", or none of these"

        verified_boxes: list[list[float]] = []
        verified_scores: list[float] = []
        verified_labels: list[str] = []

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1

            # 10 % padding
            pad_x = bw * 0.10
            pad_y = bh * 0.10
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)

            crop = image.crop((int(cx1), int(cy1), int(cx2), int(cy2)))
            if crop.size[0] < 5 or crop.size[1] < 5:
                continue

            # ---- Strategy 1: constrained choice --------------------
            choice_q = (
                "Which of the following best describes the main object "
                f"in this image: {choices_str}? "
                "Answer with just the object name or none."
            )
            choice_ans = self._blip2_ask(crop, choice_q)

            matched_obj = self._match_answer_to_target(
                choice_ans, target_objects,
            )
            if matched_obj is not None:
                verified_boxes.append(box)
                verified_scores.append(1.0)
                verified_labels.append(matched_obj)
                print(
                    f"      BLIP-2 verified (choice): '{label}' -> "
                    f"'{matched_obj}' (answer='{choice_ans}')"
                )
                continue

            # ---- Strategy 2: open caption + text matching -----------
            caption = self._blip2_ask(
                crop,
                "What is the main object in this image? "
                "Describe it briefly.",
            )

            # 2a: try direct text matching (substring / word-overlap)
            matched_obj = self._match_answer_to_target(
                caption, target_objects,
            )
            if matched_obj is not None:
                verified_boxes.append(box)
                verified_scores.append(0.95)
                verified_labels.append(matched_obj)
                print(
                    f"      BLIP-2 verified (caption-text): '{label}'"
                    f" -> '{matched_obj}' (caption='{caption}')"
                )
                continue

            # 2b: ask a second constrained-choice question, this time
            #     with the caption as additional context
            ctx_q = (
                f"This object was described as \"{caption}\". "
                "Which of the following is it most likely: "
                f"{choices_str}? "
                "Answer with just the object name or none."
            )
            ctx_ans = self._blip2_ask(crop, ctx_q)
            matched_obj = self._match_answer_to_target(
                ctx_ans, target_objects,
            )
            if matched_obj is not None:
                verified_boxes.append(box)
                verified_scores.append(0.85)
                verified_labels.append(matched_obj)
                print(
                    f"      BLIP-2 verified (caption-choice): "
                    f"'{label}' -> '{matched_obj}' "
                    f"(caption='{caption}', answer='{ctx_ans}')"
                )
                continue

            # ---- Rejected ------------------------------------------
            print(
                f"      BLIP-2 rejected: label='{label}', "
                f"choice='{choice_ans}', caption='{caption}', "
                f"ctx_choice='{ctx_ans}'"
            )

        return verified_boxes, verified_scores, verified_labels

    @staticmethod
    def _match_answer_to_target(
        answer: str,
        target_objects: list[str],
    ) -> str | None:
        """
        Check whether the free-text *answer* from BLIP-2 refers to one
        of the *target_objects*.

        Uses simple substring containment (both directions) which
        generalises to any vocabulary without hardcoded synonym lists.
        Returns the matched target object name, or ``None``.
        """
        answer = answer.strip().lower()

        # Reject explicit "none" answers
        if "none" in answer:
            return None

        # Exact match
        for obj in target_objects:
            if answer == obj:
                return obj

        # Substring containment (either direction)
        for obj in target_objects:
            if obj in answer or answer in obj:
                return obj

        # Word-overlap: for multi-word targets (e.g. "blue cube"),
        # require that the *majority* of content words match to avoid
        # spurious matches on a single shared adjective like "blue".
        stop = {"a", "an", "the", "of", "in", "on", "is", "it", "this"}
        answer_words = set(answer.split()) - stop
        for obj in target_objects:
            obj_words = set(obj.split()) - stop
            if not obj_words:
                continue
            overlap = obj_words & answer_words
            # Require >50% of the target's content words to match
            if len(overlap) > len(obj_words) / 2:
                return obj

        return None

    # -----------------------------------------------------------------
    # SAM segmentation
    # -----------------------------------------------------------------
    def segment_with_sam(
        self,
        image: Image.Image,
        boxes: list[list[float]],
    ) -> np.ndarray:
        """
        Segment objects using SAM with bounding box prompts.

        Returns a combined binary mask (uint8, values 0 or 1).
        """
        h, w = image.size[1], image.size[0]

        if not boxes:
            return np.zeros((h, w), dtype=np.uint8)

        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for box in boxes:
            input_boxes = [[box]]  # [[[x1, y1, x2, y2]]]

            inputs = self.sam_processor(
                image, input_boxes=input_boxes, return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.sam_model(**inputs)

            masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0]  # (1, 3, H, W)

            iou_scores = outputs.iou_scores.cpu().numpy()[0]  # (1, 3)
            best_idx = np.argmax(iou_scores[0])
            mask = masks[0, best_idx].numpy().astype(np.uint8)
            combined_mask = np.maximum(combined_mask, mask)

        return combined_mask

    # -----------------------------------------------------------------
    # Full prediction pipeline
    # -----------------------------------------------------------------
    def predict_mask(
        self,
        image: Image.Image,
        instruction: str,
    ) -> np.ndarray:
        """
        Predict the initial edit region mask for a given image + instruction.

        Full multi-stage pipeline:
          1. Parse instruction -> target objects
          2. Florence-2 + Grounding DINO -> candidate boxes
          3. Per-object NMS -> deduplicated boxes
          4. BLIP-2 VQA verification -> verified boxes
          5. SAM -> precise mask
          6. Morphological cleanup
        """
        w, h = image.size

        # == Stage 1: Parse instruction ===============================
        target_objects = extract_dynamic_objects(instruction)
        print(f"  Target objects: {target_objects}")

        # == Stage 2: Multi-model detection ===========================
        all_boxes: list[list[float]] = []
        all_scores: list[float] = []
        all_labels: list[str] = []

        # 2a. Florence-2
        print("  [Stage 2a] Florence-2 detection...")
        try:
            f2_boxes, f2_scores, f2_labels = self.detect_with_florence2(
                image, instruction, target_objects,
            )
        except Exception as exc:
            print(f"    WARNING: Florence-2 detection failed: {exc}")
            f2_boxes, f2_scores, f2_labels = [], [], []
        all_boxes.extend(f2_boxes)
        all_scores.extend(f2_scores)
        all_labels.extend(f2_labels)

        # 2b. Grounding DINO at default threshold
        print("  [Stage 2b] Grounding DINO detection...")
        gd_boxes, gd_scores, gd_labels = self.detect_with_gdino(
            image, target_objects,
        )
        all_boxes.extend(gd_boxes)
        all_scores.extend(gd_scores)
        all_labels.extend(gd_labels)
        print(
            f"    Grounding DINO: {len(gd_boxes)} detections "
            f"(threshold={self.gdino_box_threshold})"
        )

        # 2c. Grounding DINO at lower threshold (only for missing objects)
        found_objects = set(all_labels)
        missing_objects = [o for o in target_objects if o not in found_objects]
        if missing_objects:
            gd_boxes2, gd_scores2, gd_labels2 = self.detect_with_gdino(
                image, target_objects,
                box_threshold=0.20, text_threshold=0.15,
            )
            for b, s, lbl in zip(gd_boxes2, gd_scores2, gd_labels2):
                if lbl in missing_objects:
                    all_boxes.append(b)
                    all_scores.append(s)
                    all_labels.append(lbl)
            print(
                f"    Grounding DINO (lower threshold) added boxes for "
                f"missing: {missing_objects}"
            )

        if not all_boxes:
            print("  WARNING: No detections from any model. Empty mask.")
            return np.zeros((h, w), dtype=np.uint8)

        print(f"  Total raw candidates: {len(all_boxes)}")

        # == Stage 3: Per-object NMS ==================================
        print("  [Stage 3] Per-object NMS...")
        nms_boxes, nms_scores, nms_labels = nms_per_category(
            all_boxes, all_scores, all_labels, iou_threshold=0.5,
        )
        print(f"    After NMS: {len(nms_boxes)} boxes")
        for lbl, sc, bx in zip(nms_labels, nms_scores, nms_boxes):
            print(
                f"      '{lbl}' score={sc:.3f} "
                f"box=[{bx[0]:.0f},{bx[1]:.0f},{bx[2]:.0f},{bx[3]:.0f}]"
            )

        # == Stage 4: BLIP-2 VQA verification ========================
        print("  [Stage 4] BLIP-2 VQA verification...")
        v_boxes, v_scores, v_labels = self.verify_with_blip2(
            image, nms_boxes, nms_labels, target_objects,
        )
        print(f"    After BLIP-2 verification: {len(v_boxes)} boxes")
        for lbl, sc in zip(v_labels, v_scores):
            print(f"      '{lbl}' score={sc:.2f}")

        # == Stage 4b: Targeted recovery for missing objects ==========
        verified_objects = set(v_labels)
        missing_after_verify = [
            o for o in target_objects if o not in verified_objects
        ]
        if missing_after_verify and v_boxes:
            # Multi-model recovery: try several detection strategies
            # and verify all candidates with BLIP-2.
            print(
                f"  [Stage 4b] Targeted recovery for missing: "
                f"{missing_after_verify}"
            )
            for miss_obj in missing_after_verify:
                rec_boxes: list[list[float]] = []
                rec_scores: list[float] = []
                rec_labels: list[str] = []

                # Strategy A: Florence-2 standard OD (detects common
                # categories like COCO classes)
                try:
                    od_b, od_s, od_l = self.detect_with_florence2_od(
                        image, [miss_obj],
                    )
                except Exception as exc:
                    print(f"      WARNING: Florence-2 OD failed: {exc}")
                    od_b, od_s, od_l = [], [], []
                rec_boxes.extend(od_b)
                rec_scores.extend(od_s)
                rec_labels.extend(od_l)
                print(
                    f"      Florence-2 OD for '{miss_obj}': "
                    f"{len(od_b)} candidates"
                )

                # Strategy B: Florence-2 OVD for the single object
                task_ovd = "<OPEN_VOCABULARY_DETECTION>"
                prompt_ovd = task_ovd + miss_obj
                w_img, h_img = image.size
                ovd_count = 0
                try:
                    inputs_ovd = self.florence_processor(
                        text=prompt_ovd, images=image, return_tensors="pt",
                    ).to(self.device, torch.float16)
                    with torch.no_grad():
                        gen_ids = self.florence_model.generate(
                            **inputs_ovd, max_new_tokens=1024, num_beams=3,
                        )
                    gen_text = self.florence_processor.batch_decode(
                        gen_ids, skip_special_tokens=False,
                    )[0]
                    parsed_ovd = self.florence_processor.post_process_generation(
                        gen_text, task=task_ovd,
                        image_size=(w_img, h_img),
                    )
                    ovd_result = parsed_ovd.get(task_ovd, {})
                    if 'bboxes' in ovd_result:
                        for bbox in ovd_result['bboxes']:
                            x1, y1, x2, y2 = bbox
                            x1 = max(0, min(x1, w_img))
                            y1 = max(0, min(y1, h_img))
                            x2 = max(0, min(x2, w_img))
                            y2 = max(0, min(y2, h_img))
                            if (x2 - x1) > 2 and (y2 - y1) > 2:
                                rec_boxes.append([x1, y1, x2, y2])
                                rec_scores.append(0.80)
                                rec_labels.append(miss_obj)
                                ovd_count += 1
                except Exception as exc:
                    print(f"      WARNING: Florence-2 OVD failed: {exc}")
                print(
                    f"      Florence-2 OVD for '{miss_obj}': "
                    f"{ovd_count} candidates"
                )

                # Strategy C: GDINO at ultra-low threshold
                gd_b, gd_s, gd_l = self.detect_with_gdino(
                    image, [miss_obj],
                    box_threshold=0.05, text_threshold=0.05,
                )
                rec_boxes.extend(gd_b)
                rec_scores.extend(gd_s)
                rec_labels.extend(gd_l)
                print(
                    f"      GDINO (ultra-low 0.05) for '{miss_obj}': "
                    f"{len(gd_b)} candidates"
                )

                # Strategy D: Florence-2 dense region caption
                try:
                    drc_b, drc_s, drc_l = (
                        self.detect_with_florence2_dense_caption(
                            image, [miss_obj],
                        )
                    )
                except Exception as exc:
                    print(f"      WARNING: Florence-2 DRC failed: {exc}")
                    drc_b, drc_s, drc_l = [], [], []
                rec_boxes.extend(drc_b)
                rec_scores.extend(drc_s)
                rec_labels.extend(drc_l)
                print(
                    f"      Florence-2 DRC for '{miss_obj}': "
                    f"{len(drc_b)} candidates"
                )

                # Strategy E: Florence-2 referring expression
                ref_phrases = [miss_obj, f"the {miss_obj}"]
                ref_boxes_all: list[list[float]] = []
                for phrase in ref_phrases:
                    try:
                        ref_bx = self.detect_with_referring_expression(
                            image, phrase,
                        )
                        ref_boxes_all.extend(ref_bx)
                    except Exception as exc:
                        print(f"      WARNING: Florence-2 RefExp failed: {exc}")
                for rb in ref_boxes_all:
                    rec_boxes.append(rb)
                    rec_scores.append(0.75)
                    rec_labels.append(miss_obj)
                print(
                    f"      Florence-2 RefExp for '{miss_obj}': "
                    f"{len(ref_boxes_all)} candidates"
                )

                if not rec_boxes:
                    print(f"      No candidates found for '{miss_obj}'")
                    continue

                # De-duplicate recovery candidates via NMS
                rec_boxes, rec_scores, rec_labels = nms_per_category(
                    rec_boxes, rec_scores, rec_labels, iou_threshold=0.5,
                )

                # Verify against ALL target objects for proper
                # discrimination, then filter to only the missing one.
                rv_boxes, rv_scores, rv_labels = self.verify_with_blip2(
                    image, rec_boxes, rec_labels, target_objects,
                )
                filtered = [
                    (b, s, l)
                    for b, s, l in zip(rv_boxes, rv_scores, rv_labels)
                    if l == miss_obj
                ]
                rv_boxes = [x[0] for x in filtered]
                rv_scores = [x[1] for x in filtered]
                rv_labels = [x[2] for x in filtered]
                if rv_boxes:
                    v_boxes.extend(rv_boxes)
                    v_scores.extend(rv_scores)
                    v_labels.extend(rv_labels)
                    print(
                        f"      Recovered {len(rv_boxes)} box(es) for "
                        f"'{miss_obj}'"
                    )
                    continue

                # Strategy E: class-agnostic region proposals.
                # Get ALL object proposals and verify each with BLIP-2.
                # This is the last resort — doesn't rely on any
                # detector recognising the object by name.
                print(
                    f"      Trying region proposals for '{miss_obj}'..."
                )
                try:
                    proposals = self.get_region_proposals(image)
                except Exception as exc:
                    print(f"      WARNING: Florence-2 proposals failed: {exc}")
                    proposals = []

                # Exclude proposals that heavily overlap with already
                # verified boxes (those are known other objects)
                novel_proposals = []
                for prop in proposals:
                    overlaps_existing = False
                    for vb in v_boxes:
                        if compute_iou(prop, vb) > 0.5:
                            overlaps_existing = True
                            break
                    if not overlaps_existing:
                        novel_proposals.append(prop)

                print(
                    f"      {len(proposals)} proposals, "
                    f"{len(novel_proposals)} novel (non-overlapping)"
                )

                if novel_proposals:
                    prop_labels = [miss_obj] * len(novel_proposals)
                    rp_boxes, rp_scores, rp_labels = self.verify_with_blip2(
                        image,
                        novel_proposals,
                        prop_labels,
                        target_objects,
                    )
                    rp_filtered = [
                        (b, s, l)
                        for b, s, l in zip(rp_boxes, rp_scores, rp_labels)
                        if l == miss_obj
                    ]
                    if rp_filtered:
                        for b, s, l in rp_filtered:
                            v_boxes.append(b)
                            v_scores.append(s)
                            v_labels.append(l)
                        print(
                            f"      Region proposals recovered "
                            f"{len(rp_filtered)} box(es) for "
                            f"'{miss_obj}'"
                        )
                    else:
                        print(
                            f"      Could not verify any detection for "
                            f"'{miss_obj}' — skipping"
                        )
                else:
                    print(
                        f"      Could not verify any detection for "
                        f"'{miss_obj}' — skipping"
                    )

        if not v_boxes:
            # Fallback: if BLIP-2 rejected everything, use the single
            # top NMS detection per object
            print(
                "    BLIP-2 rejected all -> falling back to top NMS "
                "detections"
            )
            seen = set()
            for b, s, lbl in zip(nms_boxes, nms_scores, nms_labels):
                if lbl not in seen:
                    v_boxes.append(b)
                    v_scores.append(s)
                    v_labels.append(lbl)
                    seen.add(lbl)

        # == Stage 5: SAM segmentation ================================
        print(
            f"  [Stage 5] SAM segmentation on "
            f"{len(v_boxes)} verified regions..."
        )
        mask = self.segment_with_sam(image, v_boxes)

        # == Stage 6: Post-process ====================================
        mask = clean_mask(mask, min_area=50)

        return mask

    # -----------------------------------------------------------------
    # Process all test cases
    # -----------------------------------------------------------------
    def process_test_cases(
        self,
        data_dir: str,
        output_filename: str = "initial_edit_region_mask.jpg",
    ) -> dict:
        """
        Process all test cases in the given data directory.

        Each subdirectory should contain:
          - original_image.{jpg,jpeg,png,bmp,webp}
          - edit_instruction.txt

        Saves the predicted binary mask as ``output_filename`` inside each
        test-case subdirectory.
        """
        subdirs = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
            and not d.startswith('.')
        ])

        print(f"{'=' * 60}")
        print(f"Processing {len(subdirs)} test cases from: {data_dir}")
        print(f"{'=' * 60}\n")

        results = {}

        for subdir in subdirs:
            case_dir = os.path.join(data_dir, subdir)
            print(f"--- Test Case: {subdir} ---")

            # Read instruction
            instruction_path = os.path.join(case_dir, "edit_instruction.txt")
            if not os.path.exists(instruction_path):
                print("  WARNING: No edit_instruction.txt. Skipping.\n")
                continue
            with open(instruction_path, 'r') as f:
                instruction = f.read().strip()
            print(f'  Instruction: "{instruction}"')

            # Find original image
            image_path = None
            for ext in ['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tiff']:
                candidate = os.path.join(case_dir, f"original_image.{ext}")
                if os.path.exists(candidate):
                    image_path = candidate
                    break

            if image_path is None:
                for f_name in os.listdir(case_dir):
                    if (
                        f_name.lower().endswith(
                            ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
                        )
                        and 'mask' not in f_name.lower()
                        and f_name != output_filename
                    ):
                        image_path = os.path.join(case_dir, f_name)
                        break

            if image_path is None:
                print("  WARNING: No original image found. Skipping.\n")
                continue

            # Load image
            image = Image.open(image_path).convert("RGB")
            iw, ih = image.size
            print(f"  Image: {os.path.basename(image_path)} ({iw}x{ih})")

            # Predict mask
            mask = self.predict_mask(image, instruction)

            # Save mask as grayscale JPEG (0 or 255)
            mask_uint8 = (mask * 255).astype(np.uint8)
            mask_image = Image.fromarray(mask_uint8, mode='L')
            output_path = os.path.join(case_dir, output_filename)
            mask_image.save(output_path, quality=95)
            print(f"  Saved: {output_path}")

            # Statistics
            total_pixels = mask.size
            active_pixels = int(np.sum(mask > 0))
            coverage = (
                100.0 * active_pixels / total_pixels
                if total_pixels > 0
                else 0
            )
            print(
                f"  Mask coverage: {active_pixels}/{total_pixels} "
                f"pixels ({coverage:.1f}%)\n"
            )

            results[subdir] = {
                "instruction": instruction,
                "image_size": (iw, ih),
                "active_pixels": active_pixels,
                "total_pixels": total_pixels,
                "coverage_pct": round(coverage, 2),
            }

        print(f"{'=' * 60}")
        print(f"Completed processing {len(results)} test cases.")
        print(f"{'=' * 60}")

        return results


# =============================================================================
# CLI entry-point
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Predict initial edit region masks "
            "(V2 multi-stage pipeline)."
        ),
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to directory containing test-case subdirectories.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on (cuda or cpu).",
    )
    parser.add_argument(
        "--output_filename", type=str,
        default="initial_edit_region_mask.jpg",
        help="Output mask filename.",
    )
    args = parser.parse_args()

    predictor = InitialEditRegionMaskPredictor(device=args.device)
    predictor.process_test_cases(
        args.data_dir, output_filename=args.output_filename,
    )


if __name__ == "__main__":
    main()
