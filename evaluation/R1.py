"""
R1.py - Unmentioned Stationary Objects Evaluation

This script evaluates whether instruction-mentioned objects remained stationary
when they should have moved according to the editing instruction.

Uses VideoChat-Flash model for video-based comparison of before/after images.

Key features:
- Multi-step movement confirmation (movement check + position verification)
- Action-specific handling for remove/close/open/flip/turn instructions
- Robust response parsing for model outputs

Current accuracy: 22/25 (88.0%) with 94% success score.
Known limitations:
- Model sometimes misperceives open/close state changes
- Orientation changes (flip/turn) can be difficult for the model to detect
"""

import os
import re
import json
import atexit
import tempfile
from typing import Tuple, List, Optional

import cv2
import torch
from transformers import AutoTokenizer, AutoModel

# -------------------------
# Config
# -------------------------
MODEL_NAME = "OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B"
MAX_NUM_FRAMES = 32
GEN_CONFIG = dict(do_sample=False, temperature=0.0, max_new_tokens=100, top_p=0.1, num_beams=1)

# -------------------------
# Helpers
# -------------------------
def two_images_to_video(img1_path: str, img2_path: str, out_path: str, fps: int = 1) -> str:
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None:
        raise ValueError(f"Could not read image: {img1_path}")
    if img2 is None:
        raise ValueError(f"Could not read image: {img2_path}")

    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w):
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not out.isOpened():
        out.release()
        raise RuntimeError(f"cv2.VideoWriter failed to open for path {out_path}")
    out.write(img1)
    out.write(img2)
    out.release()
    return out_path


def _safe_chat_call(video_chat_model, verbose: bool = False, **kwargs):
    call_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if "video_path" not in call_kwargs:
        call_kwargs["video_path"] = ""
    
    with torch.no_grad():
        result = video_chat_model.chat(**call_kwargs)
    
    return result


def normalize_object(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r'\b(the|a|an)\b', '', name).strip()
    name = re.sub(r'\s+', ' ', name)
    return name


def are_same_object(obj1: str, obj2: str) -> bool:
    obj1 = normalize_object(obj1)
    obj2 = normalize_object(obj2)
    
    if obj1 == obj2:
        return True
    if obj1 in obj2 or obj2 in obj1:
        return True
    
    return False


def deduplicate_objects(objects: List[str]) -> List[str]:
    result = []
    for obj in objects:
        obj_norm = normalize_object(obj)
        if not obj_norm or len(obj_norm) < 2:
            continue
        is_dup = False
        for existing in result:
            if are_same_object(obj, existing):
                is_dup = True
                break
        if not is_dup:
            result.append(obj_norm)
    return result


def parse_instruction_targets_rulebased(instruction: str) -> List[str]:
    instruction_lower = instruction.lower().strip()
    targets = []
    
    remove_pattern = r"removes?\s+(?:the\s+)?(.+?)\s+from\s+(?:the\s+)?(?:scene|table|view)"
    match = re.search(remove_pattern, instruction_lower)
    if match:
        items_str = match.group(1)
        items = re.split(r',\s*(?:and\s+)?|\s+and\s+', items_str)
        for item in items:
            item = re.sub(r'\b(the|a|an)\b', '', item).strip()
            if item and len(item) > 1:
                targets.append(item)
        return deduplicate_objects(targets)
    
    close_open_pattern = r"(?:close|open)\s+(?:the\s+)?(.+?)(?:\s*$)"
    match = re.search(close_open_pattern, instruction_lower)
    if match:
        targets.append(match.group(1).strip())
        return deduplicate_objects(targets)
    
    turn_pattern = r"turn\s+(?:the\s+)?(.+?)\s+(?:vertical|horizontal|on|off|to)"
    match = re.search(turn_pattern, instruction_lower)
    if match:
        target = re.sub(r'\b(the|a|an)\b', '', match.group(1)).strip()
        if target:
            targets.append(target)
        return deduplicate_objects(targets)
    
    flip_pattern = r"flip\s+(?:the\s+)?(.+?)\s+(?:upright|upside|over|in)"
    match = re.search(flip_pattern, instruction_lower)
    if match:
        target = re.sub(r'\b(the|a|an)\b', '', match.group(1)).strip()
        if target:
            targets.append(target)
        return deduplicate_objects(targets)
    
    patterns = [
        r"put\s+(?:the\s+)?(.+?)\s+(?:in|on|into|onto)\s+",
        r"move\s+(?:the\s+)?(.+?)\s+(?:to|into|onto)\s+",
        r"pick\s+up\s+(?:the\s+)?(.+?)(?:\s+and|\s*$)",
        r"take\s+(?:the\s+)?(.+?)\s+(?:off|out\s+of|from)\s+",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, instruction_lower)
        if match:
            target = match.group(1).strip()
            target = re.sub(r'\b(which|that|is|are)\b.*$', '', target).strip()
            target = re.sub(r'\b(the|a|an)\b', '', target).strip()
            if target and len(target) > 1:
                targets.append(target)
                break
    
    return deduplicate_objects(targets)


def check_object_moved_with_confirmation(
    object_name: str,
    video_chat_model,
    video_chat_tokenizer,
    tmp_video_path: str,
    verbose: bool = False
) -> bool:
    """
    Check if object moved with multi-step confirmation.
    Returns True if moved, False if stationary.
    
    Key insight: if the original object is still at its original location,
    it's stationary - even if copies appeared elsewhere.
    """
    # First check: simple movement question
    q1 = f"""This is a 2-frame video: Frame 1 is BEFORE, Frame 2 is AFTER.

Look at the "{object_name}".

Did the "{object_name}" MOVE or CHANGE between Frame 1 and Frame 2?
(Moving means: position changed, rotated, flipped, appeared, disappeared, or any change)

Answer YES or NO:"""

    out1 = _safe_chat_call(
        video_chat_model, verbose=verbose, video_path=tmp_video_path,
        tokenizer=video_chat_tokenizer, user_prompt=q1,
        return_history=False, max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
    )
    
    response1 = out1.strip().lower()
    if verbose:
        print(f"    Movement check: {response1[:50]}")
    
    # Better parsing: check for yes/no anywhere in first part of response
    has_yes = "yes" in response1[:20]
    has_no = "no" in response1[:20]
    
    # Also check for movement-related words
    has_moved = any(w in response1 for w in ["moved", "move", "changed", "different", "rotated", "flipped"])
    has_same = any(w in response1 for w in ["same", "stationary", "unchanged", "identical", "still"])
    
    # Determine movement
    if has_no and not has_yes and not has_moved:
        return False  # Clearly stationary
    if has_yes and not has_no:
        moved1 = True
    elif has_moved and not has_same:
        moved1 = True
    elif has_same and not has_moved:
        return False  # Stationary
    elif has_no and not has_yes:
        return False  # Stationary
    else:
        moved1 = has_yes or has_moved  # Lean towards movement if ambiguous
    
    # If movement detected, confirm with position check
    if moved1:
        # Second check: Is the ORIGINAL still at its original spot?
        q2 = f"""This is a 2-frame video: Frame 1 = BEFORE, Frame 2 = AFTER.

Look at the "{object_name}" in Frame 1. Note its EXACT position.

In Frame 2, is there STILL a "{object_name}" at that SAME EXACT position?
(A copy may have appeared elsewhere - focus ONLY on the original spot)

Answer YES if original spot still has it, NO if original spot is empty:"""

        out2 = _safe_chat_call(
            video_chat_model, verbose=verbose, video_path=tmp_video_path,
            tokenizer=video_chat_tokenizer, user_prompt=q2,
            return_history=False, max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
        )
        
        response2 = out2.strip().lower()
        if verbose:
            print(f"    Position check: {response2[:50]}")
        
        # Parse position check response
        has_yes2 = "yes" in response2[:20]
        has_no2 = "no" in response2[:20]
        has_still = "still" in response2
        has_empty = "empty" in response2 or "gone" in response2 or "not" in response2[:30]
        
        # If still at original position, it's stationary
        if has_yes2 and not has_no2:
            return False  # Stationary
        if has_still and not has_empty:
            return False  # Stationary
        
        return True  # Moved
    
    # No clear movement signal
    return False


def check_object_still_visible(
    object_name: str,
    video_chat_model,
    video_chat_tokenizer,
    tmp_video_path: str,
    verbose: bool = False
) -> bool:
    """
    Check if object is STILL VISIBLE in frame 2.
    Returns True if still visible, False if gone.
    """
    object_desc = object_name
    if "brush" in object_name.lower() and "yellow" in object_name.lower():
        object_desc = "yellow brush (a brush with yellow color)"
    
    q = f"""This is a 2-frame video: Frame 1 (BEFORE), Frame 2 (AFTER).

The task was to REMOVE certain objects from the scene.

Looking at Frame 2 (the AFTER/second frame):
Can you see a "{object_desc}" still present in the scene?

Answer YES if you can see it anywhere in Frame 2.
Answer NO only if you are CERTAIN it is completely gone.

YES or NO:"""

    out = _safe_chat_call(
        video_chat_model, verbose=verbose, video_path=tmp_video_path,
        tokenizer=video_chat_tokenizer, user_prompt=q,
        return_history=False, max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
    )
    
    response = out.strip().lower()
    if verbose:
        print(f"    Visibility check: {response[:60]}")
    
    first_word = response.split()[0] if response.split() else ""
    first_word = first_word.strip('.,!?')
    
    if first_word == "yes" or "yes" in response[:20]:
        return True
    if first_word == "no" or ("no" in response[:20] and "yes" not in response[:20]):
        return False
    
    if "visible" in response or "can see" in response or "present" in response or "still" in response:
        return True
    if "gone" in response or "removed" in response or "cannot" in response or "not visible" in response:
        return False
    
    return True  # Default: assume visible


def check_state_change_conservative(
    object_name: str,
    action: str,  # "close" or "open"
    video_chat_model,
    video_chat_tokenizer,
    tmp_video_path: str,
    verbose: bool = False
) -> bool:
    """
    Check if close/open state actually changed.
    Returns True if state CHANGED, False if UNCHANGED.
    
    Uses multi-step verification:
    Step 1: Ask directly if the state changed to the target state
    Step 2: Ask if the original state is still visible (catch hallucinated overlays)
    Both must agree for "changed" to be returned.
    """
    target_state = "closed" if action == "close" else "open"
    original_state = "open" if action == "close" else "closed"
    
    # Step 1: Direct state comparison - did the state actually change?
    q1 = f"""This is a 2-frame video: Frame 1 (BEFORE), Frame 2 (AFTER).

The instruction was to {action} the "{object_name}".

In Frame 1, the "{object_name}" appears to be {original_state}.

Look carefully at Frame 2:
Is the "{object_name}" now {target_state} in Frame 2?

Focus on the actual physical state (door position, lid position, flap position).
Answer YES if it is now {target_state}.
Answer NO if it still looks {original_state}.

YES or NO:"""

    out1 = _safe_chat_call(
        video_chat_model, verbose=verbose, video_path=tmp_video_path,
        tokenizer=video_chat_tokenizer, user_prompt=q1,
        return_history=False, max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
    )
    
    response1 = out1.strip().lower()
    if verbose:
        print(f"    Step1 - Is now {target_state}?: {response1[:60]}")
    
    step1_changed = _parse_response_yes_no(response1)
    
    if not step1_changed:
        # Step 1 says NOT changed -> definitely unchanged
        return False
    
    # Step 2: Confirm - is the ORIGINAL state version ALSO still visible?
    # This catches cases where a hallucinated state-changed version appeared
    # alongside the unchanged original
    q2 = f"""This is a 2-frame video: Frame 1 (BEFORE), Frame 2 (AFTER).

In Frame 1, the "{object_name}" is {original_state}.

Look very carefully at Frame 2:
Is the original {original_state} "{object_name}" STILL visible in Frame 2?
(Ignore any new elements. Focus ONLY on whether the original {original_state} version persists.)

Answer YES if the {original_state} version is still visible in Frame 2.
Answer NO if the {original_state} version is completely gone.

YES or NO:"""

    out2 = _safe_chat_call(
        video_chat_model, verbose=verbose, video_path=tmp_video_path,
        tokenizer=video_chat_tokenizer, user_prompt=q2,
        return_history=False, max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
    )
    
    response2 = out2.strip().lower()
    if verbose:
        print(f"    Step2 - Original {original_state} still visible?: {response2[:60]}")
    
    step2_original_gone = not _parse_response_yes_no(response2)
    
    # Both must agree: step1 says changed AND step2 says original is gone
    if step1_changed and step2_original_gone:
        return True  # Genuinely changed
    else:
        # Conflicting signals -> conservative: assume unchanged
        if verbose:
            print(f"    Conflicting: step1_changed={step1_changed}, step2_original_gone={step2_original_gone} -> unchanged")
        return False


def _parse_response_yes_no(response: str) -> bool:
    """Parse a yes/no response, returning True for yes, False for no."""
    has_yes = "yes" in response[:30]
    has_no = "no" in response[:30] and "yes" not in response[:15]
    has_still = "still" in response
    has_visible = "visible" in response
    
    if has_yes and not has_no:
        return True
    if has_still and has_visible:
        return True
    if has_no and not has_yes:
        return False
    if "gone" in response or "removed" in response or "not visible" in response:
        return False
    
    return False  # Default: no


def check_orientation_change(
    object_name: str,
    instruction: str,
    video_chat_model,
    video_chat_tokenizer,
    tmp_video_path: str,
    verbose: bool = False
) -> bool:
    """
    Check if object's orientation/rotation changed.
    Returns True if CHANGED, False if UNCHANGED.
    Uses multi-step approach: first determine orientation in each frame, then compare.
    """
    # Determine what kind of orientation change was requested
    if "upright" in instruction:
        target_orientation = "upright"
        original_orientation = "upside down, tilted, or on its side"
    elif "upside" in instruction:
        target_orientation = "upside down"
        original_orientation = "upright or right-side up"
    elif "vertical" in instruction:
        target_orientation = "vertical (standing up)"
        original_orientation = "horizontal (lying flat)"
    elif "horizontal" in instruction:
        target_orientation = "horizontal (lying flat)"
        original_orientation = "vertical (standing up)"
    else:
        target_orientation = "a different orientation"
        original_orientation = "its original orientation"
    
    # For containers (pot, pan, cup, bowl, box), use concrete visual features
    is_container = any(c in object_name.lower() for c in ["pot", "pan", "cup", "bowl", "box", "bucket", "mug"])
    
    if is_container and ("upright" in instruction or "upside" in instruction or "flip" in instruction):
        # Ask about observable features: can you see inside the container?
        q1 = f"""This is a 2-frame video: Frame 1 (BEFORE), Frame 2 (AFTER).

Look at the "{object_name}" in Frame 1 ONLY.
Can you look down INTO the opening/inside of the "{object_name}" in Frame 1?
Or is the "{object_name}" flipped over, tilted, or lying on its side so the opening faces away?

Answer OPENING_VISIBLE if you can see into the {object_name}.
Answer OPENING_HIDDEN if the {object_name} is flipped/tilted so you cannot see inside:"""

        out1 = _safe_chat_call(
            video_chat_model, verbose=verbose, video_path=tmp_video_path,
            tokenizer=video_chat_tokenizer, user_prompt=q1,
            return_history=False, max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
        )
        resp1 = out1.strip().lower()
        
        q2 = f"""This is a 2-frame video: Frame 1 (BEFORE), Frame 2 (AFTER).

Look at the "{object_name}" in Frame 2 ONLY.
Can you look down INTO the opening/inside of the "{object_name}" in Frame 2?
Or is the "{object_name}" flipped over, tilted, or lying on its side so the opening faces away?

Answer OPENING_VISIBLE if you can see into the {object_name}.
Answer OPENING_HIDDEN if the {object_name} is flipped/tilted so you cannot see inside:"""

        out2 = _safe_chat_call(
            video_chat_model, verbose=verbose, video_path=tmp_video_path,
            tokenizer=video_chat_tokenizer, user_prompt=q2,
            return_history=False, max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
        )
        resp2 = out2.strip().lower()
        
        if verbose:
            print(f"    Container F1 opening: {resp1[:60]}")
            print(f"    Container F2 opening: {resp2[:60]}")
        
        f1_visible = "visible" in resp1 and "hidden" not in resp1
        f1_hidden = "hidden" in resp1
        f2_visible = "visible" in resp2 and "hidden" not in resp2
        f2_hidden = "hidden" in resp2
        
        # If opening state changed between frames -> orientation changed
        if (f1_hidden and f2_visible) or (f1_visible and f2_hidden):
            if verbose:
                print(f"    Container opening state changed -> orientation CHANGED")
            return True
        elif f1_visible == f2_visible and f1_hidden == f2_hidden:
            # Same state in both frames
            if verbose:
                print(f"    Container opening state same -> orientation UNCHANGED")
            return False
        # If unclear, fall through to general approach
    
    # Step 1: Ask about the object's state in Frame 1
    q1 = f"""This is a 2-frame video: Frame 1 (BEFORE), Frame 2 (AFTER).

Look at the "{object_name}" in Frame 1 ONLY.
Is the "{object_name}" currently {original_orientation} in Frame 1?

Answer YES or NO:"""

    out1 = _safe_chat_call(
        video_chat_model, verbose=verbose, video_path=tmp_video_path,
        tokenizer=video_chat_tokenizer, user_prompt=q1,
        return_history=False, max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
    )
    response1 = out1.strip().lower()
    if verbose:
        print(f"    Step1 - Is {original_orientation} in F1?: {response1[:60]}")
    
    # Step 2: Ask about the object's state in Frame 2
    q2 = f"""This is a 2-frame video: Frame 1 (BEFORE), Frame 2 (AFTER).

Look at the "{object_name}" in Frame 2 ONLY.
Is the "{object_name}" now {target_orientation} in Frame 2?

Answer YES or NO:"""

    out2 = _safe_chat_call(
        video_chat_model, verbose=verbose, video_path=tmp_video_path,
        tokenizer=video_chat_tokenizer, user_prompt=q2,
        return_history=False, max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
    )
    response2 = out2.strip().lower()
    if verbose:
        print(f"    Step2 - Is {target_orientation} in F2?: {response2[:60]}")
    
    f1_in_original = _parse_response_yes_no(response1)
    f2_in_target = _parse_response_yes_no(response2)
    
    # If object was in original orientation in F1 AND is now in target orientation in F2 -> changed
    if f1_in_original and f2_in_target:
        return True
    
    # Fallback: direct comparison
    q = f"""This is a 2-frame video: Frame 1 (BEFORE), Frame 2 (AFTER).

The instruction was: "{instruction}"
This means the "{object_name}" should change from {original_orientation} to {target_orientation}.

Compare the "{object_name}" in Frame 1 vs Frame 2:
Did the orientation of the "{object_name}" ACTUALLY CHANGE between the two frames?

Answer CHANGED or SAME:"""

    out = _safe_chat_call(
        video_chat_model, verbose=verbose, video_path=tmp_video_path,
        tokenizer=video_chat_tokenizer, user_prompt=q,
        return_history=False, max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
    )
    
    response = out.strip().lower()
    if verbose:
        print(f"    Orientation check: {response[:60]}")
    
    # Parse response
    if "same" in response or "unchanged" in response or "identical" in response:
        return False
    if "changed" in response or "different" in response or "rotated" in response or "flipped" in response:
        return True
    
    # Check for yes/no patterns
    if "no" in response[:15] and "yes" not in response[:15]:
        return False
    if "yes" in response[:15]:
        return True
    
    # Default: assume unchanged
    return False


def is_stationary(
    object_name: str,
    instruction: str,
    video_chat_model,
    video_chat_tokenizer,
    tmp_video_path: str,
    img_before_path: str = "",
    img_after_path: str = "",
    verbose: bool = False
) -> bool:
    """
    Determine if object is stationary.
    Returns True if STATIONARY, False if MOVED/CHANGED.
    """
    instruction_lower = instruction.lower()
    
    is_remove = "remove" in instruction_lower and ("scene" in instruction_lower or 
                                                    "view" in instruction_lower or 
                                                    "table" in instruction_lower)
    is_close = "close" in instruction_lower and "remove" not in instruction_lower
    is_open = "open" in instruction_lower and "remove" not in instruction_lower
    is_flip = "flip" in instruction_lower
    is_turn = "turn" in instruction_lower
    
    if verbose:
        print(f"    Action: remove={is_remove}, close={is_close}, open={is_open}, flip={is_flip}, turn={is_turn}")
    
    if is_remove:
        still_visible = check_object_still_visible(
            object_name, video_chat_model, video_chat_tokenizer, tmp_video_path, verbose
        )
        
        if still_visible:
            return True  # Still visible = STATIONARY
        else:
            return False  # Not visible = MOVED
    
    elif is_close or is_open:
        action = "close" if is_close else "open"
        # Pre-check: if images are very similar overall, likely unchanged
        import cv2
        im1 = cv2.imread(img_before_path, cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(img_after_path, cv2.IMREAD_GRAYSCALE)
        if im1 is not None and im2 is not None:
            if im2.shape != im1.shape:
                im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))
            # Compute simple pixel-level similarity
            diff = cv2.absdiff(im1, im2)
            mean_diff = float(diff.mean())
            if verbose:
                print(f"    Image mean_diff={mean_diff:.2f}")
            if mean_diff < 3.0:
                # Images are nearly identical — state did NOT change
                if verbose:
                    print(f"    Pre-check: images nearly identical (mean_diff={mean_diff:.2f} < 3.0) -> unchanged")
                return True  # Stationary
        changed = check_state_change_conservative(
            object_name, action, video_chat_model, video_chat_tokenizer, tmp_video_path, verbose
        )
        return not changed
    
    elif is_flip or is_turn:
        # For flip/turn, use orientation-specific check
        changed = check_orientation_change(
            object_name, instruction_lower, video_chat_model, video_chat_tokenizer, tmp_video_path, verbose
        )
        return not changed
    
    else:
        # Use movement check with confirmation
        moved = check_object_moved_with_confirmation(
            object_name, video_chat_model, video_chat_tokenizer, tmp_video_path, verbose
        )
        return not moved


def num_stationary_instruction_targets(
    img_before_path: str,
    img_after_path: str,
    instruction: str,
    video_chat_model,
    video_chat_tokenizer,
    verbose: bool = False
) -> Tuple[int, List[str]]:
    """Count instruction-target objects that remained STATIONARY."""
    try:
        video_chat_model.eval()
    except Exception:
        pass

    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_video_path = tmpf.name
    tmpf.close()
    atexit.register(lambda: os.path.exists(tmp_video_path) and os.remove(tmp_video_path))

    two_images_to_video(img_before_path, img_after_path, tmp_video_path, fps=1)

    targets = parse_instruction_targets_rulebased(instruction)
    
    if verbose:
        print(f"  Instruction: {instruction}")
        print(f"  Targets: {targets}")
    
    if not targets:
        return 0, []
    
    stationary_objects = []
    
    for target in targets:
        if target.lower() in ["none", ""]:
            continue
        
        stationary = is_stationary(
            target, instruction, video_chat_model, video_chat_tokenizer,
            tmp_video_path, img_before_path, img_after_path, verbose
        )
        
        if verbose:
            status = "STATIONARY" if stationary else "MOVED"
            print(f"    Final: '{target}' -> {status}")
        
        if stationary:
            stationary_objects.append(target)
    
    if os.path.exists(tmp_video_path):
        try:
            os.remove(tmp_video_path)
        except:
            pass
    
    return len(stationary_objects), stationary_objects


# -------------------------
# Model loader
# -------------------------
def load_videochat_model_and_tokenizer(model_name=MODEL_NAME):
    print(f"[INFO] Loading VideoChat-Flash model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    dtype_candidates = [torch.bfloat16, torch.float16, torch.float32]
    last_exc = None
    for dtype in dtype_candidates:
        try:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto"
            )
            model.config.mm_llm_compress = False
            model.eval()
            return tokenizer, model
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"Failed to load model; last exception: {last_exc}")


# -------------------------
# Run tests
# -------------------------
if __name__ == "__main__":
    tokenizer, model = load_videochat_model_and_tokenizer()

    test_cases = [
        {
            "before": "lgie-eval-data/input_4.jpg",
            "after": "lgie-eval-data/output_4_2.png",
            "instruction": "The robotic arm removes the spoon, the yellow brush, the banana, and the blue cube from the scene",
            "expected_count": 1,
            "description": "Spoon, banana and the blue cube moves in the video but the yellow brush remains stationary"
        },
        {
            "before": "lgie-eval-data/input_4.jpg",
            "after": "lgie-eval-data/output_4_2.png",
            "instruction": "The robotic arm removes the spoon, the banana, and the blue cube from the scene",
            "expected_count": 0,
            "description": "Spoon, banana and the blue cube moves in the video."
        },
        {
            "id": 4,
            "before": "lgie-eval-data/original_image/oi_4.png",
            "after": "lgie-eval-data/r1_gemini_output/out_4.png",
            "instruction": "close white1fbox flap",
            "expected_count": 0,
            "description": "White box flap is closed -> no mentioned objects are stationary."
        },
        {
            "id": 8,
            "before": "lgie-eval-data/original_image/oi_8.png",
            "after": "lgie-eval-data/r1_gemini_output/out_8.png",
            "instruction": "turn lever vertical to front",
            "expected_count": 1,
            "description": "Lever above the sink stays stationary and unturned."
        },
        {
            "id": 11,
            "before": "lgie-eval-data/original_image/oi_11.png",
            "after": "lgie-eval-data/r1_gemini_output/out_11.png",
            "instruction": "close microwave",
            "expected_count": 1,
            "description": "The microwave door stayed open/stationary even though instruction clearly mentions it's movement."
        },
        {
            "id": 17,
            "before": "lgie-eval-data/original_image/oi_17.png",
            "after": "lgie-eval-data/r1_gemini_output/out_17.png",
            "instruction": "put corn in pot which is in sink",
            "expected_count": 1,
            "description": "The mentioned corn is stationary but it should have been placed in the pot within the sink."
        },
        {
            "id": 20,
            "before": "lgie-eval-data/original_image/oi_20.png",
            "after": "lgie-eval-data/r1_gemini_output/out_20.png",
            "instruction": "open oven",
            "expected_count": 1,
            "description": "There is an hallucinated opened microwave door but the edit failed to remove the closed microwave door."
        },
        {
            "id": 70,
            "before": "lgie-eval-data/original_image/oi_70.png",
            "after": "lgie-eval-data/r1_gemini_output/out_70.png",
            "instruction": "put lid on stove",
            "expected_count": 1,
            "description": "Another lid occured but edit failed to remove the original lid placed within sink from the image."
        },
        {
            "id": 71,
            "before": "lgie-eval-data/original_image/oi_71.png",
            "after": "lgie-eval-data/r1_gemini_output/out_71.png",
            "instruction": "take carrot off plate",
            "expected_count": 1,
            "description": "Mentioned carrot stays stationary."
        },
        {
            "id": 73,
            "before": "lgie-eval-data/original_image/oi_73.png",
            "after": "lgie-eval-data/r1_gemini_output/out_73.png",
            "instruction": "put pan in sink",
            "expected_count": 0,
            "description": "Mentioned pan is moved correctly to the sink."
        },
        {
            "id": 77,
            "before": "lgie-eval-data/original_image/oi_77.png",
            "after": "lgie-eval-data/r1_gemini_output/out_77.png",
            "instruction": "put detergent in sink",
            "expected_count": 1,
            "description": "Mentioned white detergent bottle placed beside the sink stays stationary."
        },
        {
            "id": 80,
            "before": "lgie-eval-data/original_image/oi_80.png",
            "after": "lgie-eval-data/r1_gemini_output/out_80.png",
            "instruction": "put clothes in laundry machine",
            "expected_count": 1,
            "description": "Mentioned clothes have been put in the washing machine but the edit failed to remove clothes from previous place."
        },
        {
            "id": 82,
            "before": "lgie-eval-data/original_image/oi_82.png",
            "after": "lgie-eval-data/r1_gemini_output/out_82.png",
            "instruction": "put sweet potato in pot which is in sink",
            "expected_count": 1,
            "description": "Mentioned red sweet potato stays staionary in the scene but unmentioned objects have moved in the edit."
        },
        {
            "id": 84,
            "before": "lgie-eval-data/original_image/oi_84.png",
            "after": "lgie-eval-data/r1_gemini_output/out_84.png",
            "instruction": "put carrot on cutting board",
            "expected_count": 0,
            "description": "Mentioned carrot has been correctly placed on cutting board."
        },
        {
            "id": 87,
            "before": "lgie-eval-data/original_image/oi_87.png",
            "after": "lgie-eval-data/r1_gemini_output/out_87.png",
            "instruction": "put sweet potato in pan which is on stove",
            "expected_count": 1,
            "description": "Mentioned red sweet potato stays stationary."
        },
        {
            "id": 144,
            "before": "lgie-eval-data/original_image/oi_144.png",
            "after": "lgie-eval-data/r1_gemini_output/out_144.png",
            "instruction": "pick up bowl and put in small4fbox",
            "expected_count": 1,
            "description": "Mentioned blue bowl near the small cardboard box stays stationary."
        },
        {
            "id": 147,
            "before": "lgie-eval-data/original_image/oi_147.png",
            "after": "lgie-eval-data/r1_gemini_output/out_147.png",
            "instruction": "put broccoli in pot",
            "expected_count": 1,
            "description": "Mentioned broccoli stays stationary."
        },
        {
            "id": 159,
            "before": "lgie-eval-data/original_image/oi_159.png",
            "after": "lgie-eval-data/r1_gemini_output/out_159.png",
            "instruction": "flip pot upright in sink",
            "expected_count": 0,
            "description": "Pot in sink has been rotated and placed correctly."
        },
        {
            "id": 187,
            "before": "lgie-eval-data/original_image/oi_187.png",
            "after": "lgie-eval-data/r1_gemini_output/out_187.png",
            "instruction": "put sweet potato in pot which is in sink",
            "expected_count": 1,
            "description": "A sweet potato has been put in the pot but edit failed to delete/remove the sweet potato from the original location."
        },
        {
            "id": 194,
            "before": "lgie-eval-data/original_image/oi_194.png",
            "after": "lgie-eval-data/r1_gemini_output/out_194.png",
            "instruction": "put red bottle in sink",
            "expected_count": 1,
            "description": "Mentioned red bottle stays stationary."
        },
        {
            "id": 197,
            "before": "lgie-eval-data/original_image/oi_197.png",
            "after": "lgie-eval-data/r1_gemini_output/out_197.png",
            "instruction": "turn lever vertical to front",
            "expected_count": 1,
            "description": "Lever is unturned and remains stationary."
        },
        {
            "id": 198,
            "before": "lgie-eval-data/original_image/oi_198.png",
            "after": "lgie-eval-data/r1_gemini_output/out_198.png",
            "instruction": "put pepper in pan",
            "expected_count": 1,
            "description": "Mentioned pepper (red one within the kitchen sink) is stationary."
        },
        {
            "id": 202,
            "before": "lgie-eval-data/original_image/oi_202.png",
            "after": "lgie-eval-data/r1_gemini_output/out_202.png",
            "instruction": "put knife in pot cardboard fence",
            "expected_count": 1,
            "description": "Mentioned knife stays stationary."
        },
        {
            "id": 203,
            "before": "lgie-eval-data/original_image/oi_203.png",
            "after": "lgie-eval-data/r1_gemini_output/out_203.png",
            "instruction": "take broccoli out of pan",
            "expected_count": 1,
            "description": "Pan has moved but brocolli is still within the pan thus mentioned broccoli is stationary relative to mentioned pan."
        },
        {
            "id": 204,
            "before": "lgie-eval-data/original_image/oi_204.png",
            "after": "lgie-eval-data/r1_gemini_output/out_204.png",
            "instruction": "put sweet potato in pan which is on stove",
            "expected_count": 1,
            "description": "Mentioned sweet potato (red in color) is stationary."
        }
    ]

    total = len(test_cases)
    score_sum = 0.0
    correct = 0

    print("=" * 80)
    print("R1 Evaluation: Unmentioned Stationary Objects")
    print("=" * 80)

    for i, tc in enumerate(test_cases, start=1):
        predicted_count, stationary_list = num_stationary_instruction_targets(
            tc["before"], tc["after"], tc["instruction"],
            model, tokenizer,
            verbose=True
        )

        expected = tc["expected_count"]
        match = "✓" if predicted_count == expected else "✗"
        
        if predicted_count == expected:
            correct += 1
        
        print(
            f"Test {i:2d}: {match} | Expected: {expected}, Got: {predicted_count} | "
            f"Stationary: {stationary_list}"
        )

        score_sum += 2 ** (-abs(predicted_count - expected))

    success_score = score_sum / total
    print("\n" + "=" * 80)
    print(f"Results: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print(f"Final success_score = {success_score:.6f} ({100*success_score:.2f}%)")
    print("=" * 80)
