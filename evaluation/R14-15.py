"""
Minimal end-to-end script: Detect Manipulation Instruction ADHERENCE (Source & Destination)
using VideoChat-Flash.

Logic:
1. Construct a 2-frame video (Original -> Edited).
2. Query (i): Did the robot manipulate the CORRECT OBJECT? (Source Adherence)
3. Query (ii): Did the object end up in the CORRECT LOCATION? (Destination Adherence)

Enhanced approach:
- Parse instruction to extract source object and destination location
- Multi-step verification with visual confirmation
- Handle edge cases: object vanishing, state changes (open/close), no manipulation

Requirements:
%uv pip install transformers torch opencv-python
"""

import os
import re
import atexit
import tempfile
from typing import Tuple, Dict, List, Optional

import cv2
import torch
from transformers import AutoTokenizer, AutoModel

# -------------------------
# Config
# -------------------------
MODEL_NAME = "OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B"
MAX_NUM_FRAMES = 32

# Configs - Low temperature for strict logical evaluation
GEN_CONFIG = dict(do_sample=False, temperature=0.0, max_new_tokens=256, top_p=0.1, num_beams=1)
GEN_CONFIG_SHORT = dict(do_sample=False, temperature=0.0, max_new_tokens=100, top_p=0.1, num_beams=1)

# -------------------------
# Helpers
# -------------------------
def two_images_to_video(img1_path: str, img2_path: str, out_path: str, fps: int = 1, labeled: bool = False) -> str:
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None: raise ValueError(f"Could not read: {img1_path}")
    if img2 is None: raise ValueError(f"Could not read: {img2_path}")

    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w):
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)

    if labeled:
        img1 = img1.copy()
        img2 = img2.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(w, h) / 500.0
        thickness = max(2, int(font_scale * 2))
        cv2.putText(img1, "BEFORE", (10, int(35 * font_scale)), font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(img1, "BEFORE", (10, int(35 * font_scale)), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(img2, "AFTER", (10, int(35 * font_scale)), font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(img2, "AFTER", (10, int(35 * font_scale)), font, font_scale, (255, 255, 255), thickness)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    out.write(img1)
    out.write(img2)
    out.release()
    return out_path

def _safe_chat_call(video_chat_model, verbose: bool = False, **kwargs):
    call_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if "video_path" not in call_kwargs: call_kwargs["video_path"] = ""
    
    for attempt in range(2):
        try:
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            with torch.no_grad():
                return video_chat_model.chat(**call_kwargs)
        except Exception as e:
            if verbose: print(f"DEBUG chat attempt {attempt+1} failed:", repr(e))
            continue
    raise RuntimeError("Model chat failed after retries.")

def _parse_yes_no(text: str) -> str:
    """Helper to extract a strict Yes or No with better parsing."""
    t = text.lower().strip()
    
    # Check structured verdict first
    if "verdict: yes" in t or "verdict:yes" in t:
        return "Yes"
    if "verdict: no" in t or "verdict:no" in t:
        return "No"
    
    # Check for clear yes/no patterns at start
    first_line = t.split('\n')[0] if '\n' in t else t
    first_50 = first_line[:50]
    
    # Strong yes indicators
    if first_50.startswith("yes") or "answer: yes" in t or "result: yes" in t:
        return "Yes"
    if first_50.startswith("no") or "answer: no" in t or "result: no" in t:
        return "No"
    
    # Check for affirmative/negative language
    yes_words = ["yes", "correct", "adhered", "followed", "matched", "successful"]
    no_words = ["no", "incorrect", "not adhered", "wrong", "failed", "did not", "didn't"]
    
    yes_count = sum(1 for w in yes_words if w in t[:200])
    no_count = sum(1 for w in no_words if w in t[:200])
    
    if yes_count > no_count:
        return "Yes"
    if no_count > yes_count:
        return "No"
    
    # Fallback heuristic
    if "yes" in t and "no" not in t:
        return "Yes"
    if "no" in t and "yes" not in t:
        return "No"
    
    return "No"


def parse_instruction(instruction: str) -> Dict[str, str]:
    """Extract source object and destination from instruction."""
    instruction_lower = instruction.lower().strip()
    result = {"source": "", "destination": "", "action": "", "original_location": "", "preposition": "at"}
    
    # Detect action type
    actions = {
        "put": ["put", "place", "set"],
        "take": ["take", "pick", "grab", "remove"],
        "move": ["move", "push", "slide"],
        "open": ["open"],
        "close": ["close"],
        "turn": ["turn", "rotate", "flip"]
    }
    
    for action_type, verbs in actions.items():
        for verb in verbs:
            if verb in instruction_lower:
                result["action"] = action_type
                break
        if result["action"]:
            break
    
    # Pattern for "picks up X and puts it in/on Y"
    pickup_put_pattern = r"picks?\s+up\s+(?:the\s+)?(.+?)\s+and\s+puts?\s+(?:it\s+)?(in|on|into|onto|inside)\s+(?:the\s+)?(.+?)(?:\s*$|\s+which|\s+that)"
    match = re.search(pickup_put_pattern, instruction_lower)
    if match:
        result["source"] = match.group(1).strip()
        prep = match.group(2)
        result["preposition"] = "in" if prep in ("into", "inside") else ("on" if prep == "onto" else prep)
        result["destination"] = match.group(3).strip()
        return result
    
    # Pattern for "picks up X (from Y)" without a put destination
    pickup_from_pattern = r"picks?\s+up\s+(?:the\s+)?(.+?)(?:\s+from\s+(?:the\s+)?(.+?))?(?:\s*$|\s+and)"
    match = re.search(pickup_from_pattern, instruction_lower)
    if match:
        result["source"] = match.group(1).strip()
        if match.group(2):
            result["original_location"] = match.group(2).strip()
        result["destination"] = "surface"
        result["action"] = "pickup"
        return result
    
    # Pattern for "puts X from Y in Z" — must be checked BEFORE generic put pattern
    # to avoid eating "from Y" into the source object name
    put_from_pattern = r"puts?\s+(?:the\s+)?(.+?)\s+from\s+(?:\w+\s+)?(in|on|into)\s+(?:the\s+)?(.+?)(?:\s*$)"
    match = re.search(put_from_pattern, instruction_lower)
    if match:
        result["source"] = match.group(1).strip()
        prep = match.group(2)
        result["preposition"] = "in" if prep == "into" else prep
        result["destination"] = match.group(3).strip()
        return result
    
    # Pattern for "put X in/on Y"
    put_pattern = r"(?:put|place|set)s?\s+(?:the\s+)?(.+?)\s+(in|on|into|onto|inside)\s+(?:the\s+)?(.+?)(?:\s*$|\s+which|\s+that)"
    match = re.search(put_pattern, instruction_lower)
    if match:
        result["source"] = match.group(1).strip()
        prep = match.group(2)
        result["preposition"] = "in" if prep in ("into", "inside") else ("on" if prep == "onto" else prep)
        dest = match.group(3).strip()
        # Post-process: if destination has 3+ words without articles, prepositions,
        # or joining words, it's likely "target_object location_modifier" — take first word
        dest_words = dest.split()
        if len(dest_words) >= 3 and not any(w in ("or", "and", "of", "the", "a", "in", "on") for w in dest_words[1:]):
            dest = dest_words[0]
        result["destination"] = dest
        return result
    
    # Pattern for "take X out of Y"
    take_pattern = r"(?:take|pick|grab|remove)s?\s+(?:the\s+)?(.+?)\s+(?:out of|off|from)\s+(?:the\s+)?(.+?)(?:\s*$|\s+and)"
    match = re.search(take_pattern, instruction_lower)
    if match:
        result["source"] = match.group(1).strip()
        result["original_location"] = match.group(2).strip()
        result["destination"] = "surface"  # Default for take-out actions
        return result
    
    # Pattern for "open/close X"
    state_pattern = r"(?:open|close)s?\s+(?:the\s+)?(.+?)(?:\s*$|'s|\s+flap)"
    match = re.search(state_pattern, instruction_lower)
    if match:
        result["source"] = match.group(1).strip()
        result["destination"] = "state_change"
        return result
    
    # Pattern for "turn X vertical/horizontal"
    turn_pattern = r"(?:turn|rotate|flip)s?\s+(?:the\s+)?(.+?)\s+(?:vertical|horizontal|to)"
    match = re.search(turn_pattern, instruction_lower)
    if match:
        result["source"] = match.group(1).strip()
        result["destination"] = "orientation_change"
        return result
    
    return result


# -------------------------
# Core Logic: Instruction Adherence
# -------------------------
def check_source_adherence(
    instruction: str,
    video_chat_model,
    video_chat_tokenizer,
    tmp_video_path: str,
    verbose: bool = False
) -> str:
    """
    Check if the correct object was manipulated (source adherence).
    Uses a simple two-step approach: detect what changed, then verify match.
    """
    parsed = parse_instruction(instruction)
    source_obj = parsed.get("source", "")
    action = parsed.get("action", "")
    
    if verbose:
        print(f"  Parsed: source='{source_obj}', action='{action}'")
    
    # Step 1: Ask what object changed position or state
    q1 = f"""This is a 2-frame video: Frame 1 (BEFORE), Frame 2 (AFTER).

What SINGLE object changed position or state between Frame 1 and Frame 2?
Name the object that moved, was added, was removed, or changed state:"""

    out1 = _safe_chat_call(
        video_chat_model, verbose=verbose, video_path=tmp_video_path,
        tokenizer=video_chat_tokenizer, user_prompt=q1,
        return_history=False, max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG_SHORT
    )
    
    detected_obj = out1.strip().lower()
    if verbose:
        print(f"  Step 1 - Detected change: {detected_obj[:100]}...")
    
    # Step 2: Ask specifically if source_obj changed
    if action in ("take", "pickup"):
        change_desc = "picked up, grabbed, removed, or moved"
    elif action in ("open", "close"):
        change_desc = "opened or closed"
    elif action in ("turn",):
        change_desc = "turned, rotated, or flipped"
    else:
        change_desc = "moved, grabbed, or changed position"
    
    q2 = f"""This is a 2-frame video: Frame 1 (BEFORE), Frame 2 (AFTER).

Target object: "{source_obj}"

QUESTION: Did the "{source_obj}" get {change_desc} between Frame 1 and Frame 2?
(Not a different object, specifically the "{source_obj}")

Answer YES if the "{source_obj}" changed position, state, or was manipulated.
Answer NO if a different object changed instead, or "{source_obj}" is unchanged.

Verdict: [Yes/No]"""

    out2 = _safe_chat_call(
        video_chat_model, verbose=verbose, video_path=tmp_video_path,
        tokenizer=video_chat_tokenizer, user_prompt=q2,
        return_history=False, max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
    )
    
    if verbose:
        print(f"  Step 2 - Source check: {out2[:200]}...")
    
    return _parse_yes_no(out2)


def check_destination_adherence(
    instruction: str,
    video_chat_model,
    video_chat_tokenizer,
    tmp_video_path: str,
    verbose: bool = False,
    tmp_video_path_labeled: str = None
) -> str:
    """
    Check if the object ended up in the correct location/state.
    Uses simple direct question about final location.
    """
    parsed = parse_instruction(instruction)
    source_obj = parsed.get("source", "")
    destination = parsed.get("destination", "")
    action = parsed.get("action", "")
    original_location = parsed.get("original_location", "")
    preposition = parsed.get("preposition", "at")
    
    if verbose:
        print(f"  Parsed: destination='{destination}', action='{action}', original_location='{original_location}', prep='{preposition}'")
    
    # Build destination-specific prompt
    if destination == "orientation_change":
        q = f"""This is a 2-frame video: Frame 1 (BEFORE) and Frame 2 (AFTER).

Compare Frame 1 and Frame 2 carefully.

QUESTION: Has the "{source_obj}" been turned or rotated in Frame 2 compared to Frame 1?
Answer Yes ONLY if the "{source_obj}" has a CLEARLY different orientation.
Answer No if the "{source_obj}" looks the same in both frames.

Verdict: [Yes/No]"""

        out = _safe_chat_call(
            video_chat_model, verbose=verbose, video_path=tmp_video_path,
            tokenizer=video_chat_tokenizer, user_prompt=q,
            return_history=False, max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
        )
    elif destination == "state_change":
        # State change: check if object was opened/closed - use labeled video
        video_for_state = tmp_video_path_labeled if tmp_video_path_labeled else tmp_video_path
        if action == "close":
            state_verb = "closed"
            opposite_verb = "open"
        else:
            state_verb = "opened"
            opposite_verb = "closed"
        q = f"""This is a 2-frame video: Frame 1 (BEFORE) and Frame 2 (AFTER).

Compare Frame 1 and Frame 2 carefully.

QUESTION: Has the "{source_obj}" been {state_verb} in Frame 2 compared to Frame 1?
Answer Yes if the "{source_obj}" was {opposite_verb} in Frame 1 and is now {state_verb} in Frame 2.
Answer No if it looks the same in both frames.

Verdict: [Yes/No]"""

        out = _safe_chat_call(
            video_chat_model, verbose=verbose, video_path=video_for_state,
            tokenizer=video_chat_tokenizer, user_prompt=q,
            return_history=False, max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
        )
        
        # Arrangement verification: when state changed, check no objects were displaced
        result = _parse_yes_no(out)
        if result == "Yes":
            q_arr = f"""This is a 2-frame video: Frame 1 (BEFORE) and Frame 2 (AFTER).

Look at the "{source_obj}" in Frame 2.
Is there any object ON TOP of or INSIDE the "{source_obj}" in Frame 2 that was NOT there in Frame 1?

Answer YES if a new object appeared on/in the "{source_obj}" after the state change.
Answer NO if nothing new is on/in the "{source_obj}".

Verdict: [Yes/No]"""
            
            out_arr = _safe_chat_call(
                video_chat_model, verbose=verbose, video_path=video_for_state,
                tokenizer=video_chat_tokenizer, user_prompt=q_arr,
                return_history=False, max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
            )
            
            if verbose:
                print(f"  Arrangement check: {out_arr[:200]}...")
            
            if _parse_yes_no(out_arr) == "Yes":
                # Objects displaced → state change result is incorrect
                if verbose:
                    print(f"  Objects displaced during state change → overriding to No")
                result = "No"
        
        if verbose:
            print(f"  Destination check: {out[:200]}...")
        return result
    elif destination == "surface" and original_location:
        if action == "pickup":
            # Pickup action: check if object has been lifted/moved from its location
            q = f"""This is a 2-frame video: Frame 1 (BEFORE) and Frame 2 (AFTER).

Compare how the "{source_obj}" relates to the "{original_location}" in Frame 1 vs Frame 2.

QUESTION: Has the "{source_obj}" been picked up or moved from the "{original_location}"?
Answer Yes if the "{source_obj}" has been lifted, moved away, or repositioned from the "{original_location}".
Answer No if the "{source_obj}" is still in the same position on the "{original_location}" as in Frame 1.

Verdict: [Yes/No]"""
        else:
            # Take-out action: check object is visible OUTSIDE its original container
            q = f"""This is a 2-frame video: Frame 1 (BEFORE) and Frame 2 (AFTER).

In Frame 2, can you see the "{source_obj}" somewhere OUTSIDE of the "{original_location}"?

Answer Yes if "{source_obj}" is visible in Frame 2 and NOT in/on the "{original_location}".
Answer No if "{source_obj}" has vanished, is missing, or is still in the "{original_location}".

Verdict: [Yes/No]"""

        out = _safe_chat_call(
            video_chat_model, verbose=verbose, video_path=tmp_video_path,
            tokenizer=video_chat_tokenizer, user_prompt=q,
            return_history=False, max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
        )
    else:
        # Standard destination check - use labeled video for better frame distinction
        video_for_dest = tmp_video_path_labeled if tmp_video_path_labeled else tmp_video_path
        
        # Handle disjunctive destinations ("pot or pan", "cup or bowl")
        dest_alternatives = [destination]
        if " or " in destination:
            dest_alternatives = [d.strip() for d in destination.split(" or ")]
        
        result = "No"
        for dest_option in dest_alternatives:
            q = f"""This is a 2-frame video: Frame 1 (BEFORE) and Frame 2 (AFTER).

In Frame 2, is the "{source_obj}" located at/on/in the "{dest_option}"?

Answer Yes ONLY if you can clearly see "{source_obj}" at the "{dest_option}".
Answer No if "{source_obj}" is missing, vanished, or NOT at the "{dest_option}".

Verdict: [Yes/No]"""

            out = _safe_chat_call(
                video_chat_model, verbose=verbose, video_path=video_for_dest,
                tokenizer=video_chat_tokenizer, user_prompt=q,
                return_history=False, max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
            )
            
            this_result = _parse_yes_no(out)
            if this_result == "Yes":
                result = "Yes"
                if verbose:
                    print(f"  Destination '{dest_option}' matched.")
                break
            
            # OR-gate: if labeled video says No, double-check with unlabeled video
            if this_result == "No" and tmp_video_path_labeled and video_for_dest == tmp_video_path_labeled:
                if verbose:
                    print(f"  Labeled dest=No for '{dest_option}', double-checking with unlabeled...")
                out2 = _safe_chat_call(
                    video_chat_model, verbose=verbose, video_path=tmp_video_path,
                    tokenizer=video_chat_tokenizer, user_prompt=q,
                    return_history=False, max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
                )
                result2 = _parse_yes_no(out2)
                if result2 == "Yes":
                    if verbose:
                        print(f"  Unlabeled dest=Yes for '{dest_option}', using Yes (OR-gate recovery)")
                    result = "Yes"
                    break
        
        # Source-agnostic fallback: if all specific checks failed,
        # ask if ANY object was newly placed at the destination
        if result == "No":
            primary_dest = dest_alternatives[0]
            q_generic = f"""This is a 2-frame video: Frame 1 (BEFORE) and Frame 2 (AFTER).

In Frame 2 compared to Frame 1, has any object been newly placed at/on/in the "{primary_dest}"?

Answer Yes if you can see a new object at the "{primary_dest}" in Frame 2 that was not there in Frame 1.
Answer No if nothing new is at the "{primary_dest}".

Verdict: [Yes/No]"""
            
            out_generic = _safe_chat_call(
                video_chat_model, verbose=verbose, video_path=tmp_video_path,
                tokenizer=video_chat_tokenizer, user_prompt=q_generic,
                return_history=False, max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
            )
            generic_result = _parse_yes_no(out_generic)
            if generic_result == "Yes":
                if verbose:
                    print(f"  Source-agnostic fallback: Yes → recovering destination to Yes")
                result = "Yes"
        
        if verbose:
            print(f"  Destination check: {out[:200]}...")
        return result
    
    if verbose:
        print(f"  Destination check: {out[:200]}...")
    
    return _parse_yes_no(out)


def check_instruction_adherence(
    img_before_path: str,
    img_after_path: str,
    instruction: str,
    video_chat_model,
    video_chat_tokenizer,
    verbose: bool = False
) -> Dict[str, str]:
    
    # Setup Videos - unlabeled for source, labeled for destination
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_video_path = tmpf.name
    tmpf.close()
    atexit.register(lambda p=tmp_video_path: os.path.exists(p) and os.remove(p))
    two_images_to_video(img_before_path, img_after_path, tmp_video_path)

    tmpf2 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_video_path_labeled = tmpf2.name
    tmpf2.close()
    atexit.register(lambda p=tmp_video_path_labeled: os.path.exists(p) and os.remove(p))
    two_images_to_video(img_before_path, img_after_path, tmp_video_path_labeled, labeled=True)

    results = {}
    parsed = parse_instruction(instruction)
    source_obj = parsed.get("source", "")
    destination = parsed.get("destination", "")
    
    if verbose:
        print(f"\n=== Checking: {instruction} ===")

    # Check Source Adherence (unlabeled video)
    results["Source_Adherence"] = check_source_adherence(
        instruction, video_chat_model, video_chat_tokenizer, 
        tmp_video_path, verbose
    )
    
    # Check Destination Adherence (labeled video for standard/state_change, unlabeled for others)
    results["Destination_Adherence"] = check_destination_adherence(
        instruction, video_chat_model, video_chat_tokenizer,
        tmp_video_path, verbose, tmp_video_path_labeled=tmp_video_path_labeled
    )
    
    return results

# -------------------------
# Loader
# -------------------------
def load_videochat_model_and_tokenizer(model_name=MODEL_NAME):
    print(f"Loading {model_name}...")
    # Ensure CUDA deterministic behavior across containers
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, 
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config.mm_llm_compress = False
    return tokenizer, model

# -------------------------
# Run Tests
# -------------------------
if __name__ == "__main__":
    
    test_cases = [
        {
            "before": "lgie-eval-data/input_3.jpg",
            "after": "lgie-eval-data/output_3_2.png",
            "instruction": "The robotic arm puts the sushi on top of the omelette",
            "expected_source_adherence": "Yes",
            "expected_destination_adherence": "Yes",
            "description": "Sushi is the correct source, and omelette is the correct destination"
        },
        {
            "before": "lgie-eval-data/input_3.jpg",
            "after": "lgie-eval-data/output_3_2.png",
            "instruction": "The robotic arm puts the omelette on top of the cabbage",
            "expected_source_adherence": "No",
            "expected_destination_adherence": "No",
            "description": "Source is sushi instead of omelette and destination is omelette instead of cabbage"
        },
        {
            "before": "lgie-eval-data/input_3.jpg",
            "after": "lgie-eval-data/output_3_2.png",
            "instruction": "The robotic arm puts the sushi on top of the cabbage",
            "expected_source_adherence": "Yes",
            "expected_destination_adherence": "No",
            "description": "Sushi is the correct source, but destination is omelette instead of cabbage"
        },
        {
            "before": "lgie-eval-data/input_3.jpg",
            "after": "lgie-eval-data/output_3_2.png",
            "instruction": "The robotic arm puts the cabbage on top of the omelette",
            "expected_source_adherence": "No",
            "expected_destination_adherence": "Yes",
            "description": "Omelette is the correct destination, but source is sushi instead of cabbage"
        },
        {
            "before": "lgie-eval-data/input_3.jpg",
            "after": "lgie-eval-data/input_3.png",
            "instruction": "The robotic arm puts the sushi on top of the omelette",
            "expected_source_adherence": "No",
            "expected_destination_adherence": "No",
            "description": "Input and output are the same image, no manipulation happened"
        },
        {
            "before": "lgie-eval-data/input_3.jpg",
            "after": "lgie-eval-data/output_3_3.png",
            "instruction": "The robotic arm puts the sushi on top of the omelette",
            "expected_source_adherence": "No",
            "expected_destination_adherence": "No",
            "description": "Extra omlette added, omlette is put on top of cabbage here."
        },
        {
            "before": "lgie-eval-data/input_5.png",
            "after": "lgie-eval-data/gt_5.png",
            "instruction": "The robotic arm picks up glue and puts it into the drawer",
            "expected_source_adherence": "Yes",
            "expected_destination_adherence": "Yes",
            "description": "Ground truth."
        },
        {
            "before": "lgie-eval-data/input_6.png",
            "after": "lgie-eval-data/gt_6.png",
            "instruction": "The robotic arm puts the broccoli in pot or pan",
            "expected_source_adherence": "Yes",
            "expected_destination_adherence": "Yes",
            "description": "Ground truth."
        },
        {
            "before": "lgie-eval-data/input_7.png",
            "after": "lgie-eval-data/gt_7.png",
            "instruction": "The robotic arm puts the banana in colander",
            "expected_source_adherence": "Yes",
            "expected_destination_adherence": "Yes",
            "description": "Ground truth."
        },
        {
            "before": "lgie-eval-data/input_8.png",
            "after": "lgie-eval-data/gt_8.png",
            "instruction": "The robotic arm takes the banana out of the colander",
            "expected_source_adherence": "Yes",
            "expected_destination_adherence": "Yes",
            "description": "Ground truth."
        },
        {
            "before": "lgie-eval-data/input_9.png",
            "after": "lgie-eval-data/gt_9.png",
            "instruction": "The robotic arm takes the broccoli out of the pan",
            "expected_source_adherence": "Yes",
            "expected_destination_adherence": "Yes",
            "description": "Ground truth."
        },
        {
            "before": "lgie-eval-data/input_10.png",
            "after": "lgie-eval-data/gt_10.png",
            "instruction": "The robotic arm puts the cucumber in the cup",
            "expected_source_adherence": "Yes",
            "expected_destination_adherence": "Yes",
            "description": "Ground truth."
        },
        {
            "before": "lgie-eval-data/input_11.png",
            "after": "lgie-eval-data/gt_11.png",
            "instruction": "The robotic arm puts the small spoon from basked in the tray",
            "expected_source_adherence": "Yes",
            "expected_destination_adherence": "Yes",
            "description": "Ground truth."
        },
        {
            "before": "lgie-eval-data/input_12.png",
            "after": "lgie-eval-data/gt_12.png",
            "instruction": "The robotic arm closes the brown box's flap",
            "expected_source_adherence": "Yes",
            "expected_destination_adherence": "Yes",
            "description": "Ground truth."
        },
        {
            "before": "lgie-eval-data/input_13.png",
            "after": "lgie-eval-data/gt_13.png",
            "instruction": "The robotic arm puts the corn in the pan which is on the stove",
            "expected_source_adherence": "Yes",
            "expected_destination_adherence": "Yes",
            "description": "Ground truth."
        },
        {
            "before": "lgie-eval-data/input_14.png",
            "after": "lgie-eval-data/gt_14.png",
            "instruction": "The robotic arm picks up pan from the stove",
            "expected_source_adherence": "Yes",
            "expected_destination_adherence": "Yes",
            "description": "Ground truth."
        },
        
        {
          "id": 184,
          "before": "lgie-eval-data/original_image/oi_184.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_184.png",
          "instruction": "take clothes out of laundry machine",
          "expected_source_adherence": "Yes",
          "expected_destination_adherence": "No",
          "description": "Clothes are the correct source, and they have vanished from the image, but they are to be placed on surface or basket."
        },
        {
          "id": 185,
          "before": "lgie-eval-data/original_image/oi_185.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_185.png",
          "instruction": "take sushi out of pan",
          "expected_source_adherence": "Yes",
          "expected_destination_adherence": "No",
          "description": "Sushi is the correct source, and kitchen surface is the desired destination but sushi has vanished from the scene"
        },
        {
          "id": 186,
          "before": "lgie-eval-data/original_image/oi_186.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_186.png",
          "instruction": "put spatula on cutting board",
          "expected_source_adherence": "Yes",
          "expected_destination_adherence": "No",
          "description": "Spatuala is the correct source, but it isn't put on cutting board, only moved on the surface."
        },
        {
          "id": 187,
          "before": "lgie-eval-data/original_image/oi_187.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_187.png",
          "instruction": "put sweet potato in pot which is in sink",
          "expected_source_adherence": "No",
          "expected_destination_adherence": "Yes",
          "description": "Red sweet potato is the desired source, and pot in kitchen sink is the correct destination but a new sweet potato has been hallucinated and placed in the pot."
        },
        {
          "id": 188,
          "before": "lgie-eval-data/original_image/oi_188.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_188.png",
          "instruction": "take carrot off plate",
          "expected_source_adherence": "Yes",
          "expected_destination_adherence": "Yes",
          "description": "Carrot is the correct source, and surface is the correct destination when no explicit destination is mentioned."
        },
        {
          "id": 189,
          "before": "lgie-eval-data/original_image/oi_189.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_189.png",
          "instruction": "put broccoli in pot",
          "expected_source_adherence": "Yes",
          "expected_destination_adherence": "Yes",
          "description": "Brocolli is the correct source, and Metal pot is the correct destination"
        },
        {
          "id": 190,
          "before": "lgie-eval-data/original_image/oi_190.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_190.png",
          "instruction": "take broccoli out of pan",
          "expected_source_adherence": "Yes",
          "expected_destination_adherence": "No",
          "description": "Broccoli is the correct source, but it has vanished from the scene, it should be placed somewhere in the scene if no explicit destination is mentioned."
        },
        {
          "id": 191,
          "before": "lgie-eval-data/original_image/oi_191.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_191.png",
          "instruction": "open oven",
          "expected_source_adherence": "Yes",
          "expected_destination_adherence": "Yes",
          "description": "Oven is the correct source, and is in open state -> model correctly followed the instruction."
        },
        {
          "id": 192,
          "before": "lgie-eval-data/original_image/oi_192.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_192.png",
          "instruction": "take carrot off plate",
          "expected_source_adherence": "Yes",
          "expected_destination_adherence": "No",
          "description": "Carrot is the correct source, but it has moved out of scene."
        },
        {
          "id": 193,
          "before": "lgie-eval-data/original_image/oi_193.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_193.png",
          "instruction": "open book",
          "expected_source_adherence": "Yes",
          "expected_destination_adherence": "No",
          "description": "Book is opened but omelette which should be below the book is on top of it."
        },
        {
          "id": 194,
          "before": "lgie-eval-data/original_image/oi_194.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_194.png",
          "instruction": "put red bottle in sink",
          "expected_source_adherence": "No",
          "expected_destination_adherence": "No",
          "description": "Red chilly has been moved and removed from the scene but it is not the correct source neither the destination is kitchen sink."
        },
        {
          "id": 195,
          "before": "lgie-eval-data/original_image/oi_195.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_195.png",
          "instruction": "put knife in pot cardboard fence",
          "expected_source_adherence": "Yes",
          "expected_destination_adherence": "No",
          "description": "Knife is the correct source, but pot inside the cardboard fence is the correct destination not the cardboard fence itself."
        },
        {
          "id": 196,
          "before": "lgie-eval-data/original_image/oi_196.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_196.png",
          "instruction": "put corn in pot which is in sink",
          "expected_source_adherence": "Yes",
          "expected_destination_adherence": "Yes",
          "description": "Corn is the correct source, and pot inside the sink is the correct destination"
        },
        {
          "id": 197,
          "before": "lgie-eval-data/original_image/oi_197.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_197.png",
          "instruction": "turn lever vertical to front",
          "expected_source_adherence": "No",
          "expected_destination_adherence": "No",
          "description": "Lever is unturned."
        },
        {
          "id": 198,
          "before": "lgie-eval-data/original_image/oi_198.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_198.png",
          "instruction": "put pepper in pan",
          "expected_source_adherence": "No",
          "expected_destination_adherence": "Yes",
          "description": "Yellow bell pepper is not the correct source instead red chilly pepper is, though pan is the correct destination."
        },
        {
          "id": 199,
          "before": "lgie-eval-data/original_image/oi_199.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_199.png",
          "instruction": "put corn in pot which is in sink",
          "expected_source_adherence": "Yes",
          "expected_destination_adherence": "No",
          "description": "Corn is the correct source, but pan is not the correct destination"
        },
        {
          "id": 200,
          "before": "lgie-eval-data/original_image/oi_200.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_200.png",
          "instruction": "take sushi out of pan",
          "expected_source_adherence": "Yes",
          "expected_destination_adherence": "Yes",
          "description": "Sushi is the correct source, and is placed outside the pan which is correct destination."
        },
        {
          "id": 201,
          "before": "lgie-eval-data/original_image/oi_201.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_201.png",
          "instruction": "put spatula on cutting board",
          "expected_source_adherence": "Yes",
          "expected_destination_adherence": "No",
          "description": "Spatula is the correct source, but is vanished from the scene."
        },
        {
          "id": 202,
          "before": "lgie-eval-data/original_image/oi_202.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_202.png",
          "instruction": "put knife in pot cardboard fence",
          "expected_source_adherence": "No",
          "expected_destination_adherence": "No",
          "description": "Pot is placed on top of knife whereas knife should have been placed inside of pot."
        },
        {
          "id": 203,
          "before": "lgie-eval-data/original_image/oi_203.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_203.png",
          "instruction": "take broccoli out of pan",
          "expected_source_adherence": "No",
          "expected_destination_adherence": "No",
          "description": "Pan has moved but brocolli is still within the pan."
        },
        {
          "id": 204,
          "before": "lgie-eval-data/original_image/oi_204.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_204.png",
          "instruction": "put sweet potato in pan which is on stove",
          "expected_source_adherence": "No",
          "expected_destination_adherence": "Yes",
          "description": "Pan on the stove is the correct destination but carrot is not the correct source."
        },
        {
          "id": 205,
          "before": "lgie-eval-data/original_image/oi_205.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_205.png",
          "instruction": "put broccoli in pot or pan",
          "expected_source_adherence": "Yes",
          "expected_destination_adherence": "No",
          "description": "Brocolli is the correct source, and it has moved but not placed in the pot or pan."
        },
        {
          "id": 206,
          "before": "lgie-eval-data/original_image/oi_206.png",
          "after": "lgie-eval-data/r14-15_gemini_output/out_206.png",
          "instruction": "put knife on cutting board",
          "expected_source_adherence": "Yes",
          "expected_destination_adherence": "Yes",
          "description": "Knife is the correct source, and cutting board is the correct destination"
        }

    ]

    # Warn about missing files and only run cases that exist
    for tc in test_cases:
        if not (os.path.exists(tc["before"]) and os.path.exists(tc["after"])):
            print(f"WARNING: Test files for case '{tc['description']}' not found: "
                  f"{tc['before']} or {tc['after']}. This case will be skipped.")

    # If none of the test image pairs exist, skip loading the heavy model
    any_pairs_exist = any((os.path.exists(tc["before"]) and os.path.exists(tc["after"])) for tc in test_cases)
    if not any_pairs_exist:
        print("No valid test image pairs were found. Skipping inference run.")
    else:
        # Load model & tokenizer once (heavy)
        tokenizer, model = load_videochat_model_and_tokenizer()

        executed = 0
        score_sum = 0.0

        for i, tc in enumerate(test_cases, start=1):
            if not (os.path.exists(tc["before"]) and os.path.exists(tc["after"])):
                continue  # skip missing case

            executed += 1
            preds = check_instruction_adherence(
                tc["before"], tc["after"], tc["instruction"],
                model, tokenizer, verbose=False
            )

            # Normalize predictions and expected values
            p_source = preds.get("Source_Adherence", "No").strip().title()
            p_dest = preds.get("Destination_Adherence", "No").strip().title()

            e_source = tc["expected_source_adherence"].strip().title()
            e_dest = tc["expected_destination_adherence"].strip().title()

            # compute per-test correctness (each question worth 1/2)
            correct = 0
            correct += 1 if p_source == e_source else 0
            correct += 1 if p_dest == e_dest else 0

            per_case_score = correct / 2.0
            score_sum += per_case_score

            # Print using the requested phrases
            print(
                f"Test {i}: {tc['description']} -> "
                f"Predictions: source_adherence={p_source}, destination_adherence={p_dest} | "
                f"Expected: expected_source_adherence={e_source}, expected_destination_adherence={e_dest} | "
                f"score={per_case_score:.3f}"
            )

        if executed > 0:
            success_score = score_sum / executed
            print(f"\nFinal success_score = {success_score:.6f} ({success_score * 100:.2f}%) over {executed} executed tests")
        else:
            print("No test cases were executed (all missing or skipped).")