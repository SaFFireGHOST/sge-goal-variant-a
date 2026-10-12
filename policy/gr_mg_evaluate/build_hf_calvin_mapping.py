# Copyright (c) 2026
"""Build a 1-to-1 mapping between rows of the Hugging Face dataset
``SuyashAjitChavan2002/Calvin-Dataset-Final`` (test split) and language-annotated
segments of the official CALVIN-D ``task_D_D/validation`` split.

Why this is needed
------------------
The HF parquet ships only ``original_image`` / ``edited_image`` /
``edit_instruction`` / masks per row, identified by an opaque
``episode_<E>_atomic_<A>`` string. To actually *evaluate* GR-MG using those
images as the final goal, we need the matching CALVIN frame indices so we can:

* reset the PyBullet env to the same initial ``robot_obs`` / ``scene_obs``,
* know the canonical task name (e.g. ``"open_drawer"``) for the success oracle,
* know the official template instruction (for sanity-check / fallback).

Strategy: pixel hash match
--------------------------
For each language-annotated segment ``(start, end, task, ann)`` in
``validation/lang_annotations/auto_lang_ann.npy`` we read
``episode_{start:07d}.npz`` and ``episode_{end:07d}.npz`` and hash their
``rgb_static`` arrays. We then hash each HF row's ``(original_image,
edited_image)`` pair and look up the segment with the same pair of hashes.

If exact hashing fails for a row (e.g. JPEG re-encoding inside the parquet),
we fall back to a nearest-neighbour search by L2 distance against all
candidate segments whose canonical task annotation contains tokens shared
with the HF ``edit_instruction``. The fallback is logged.

Output
------
A JSON file::

    {
      "split": "validation",
      "num_hf_rows":   1087,
      "num_segments":  <S>,
      "matches": [
        {
          "hf_row":    0,
          "hf_id":     "episode_0_atomic_0001",
          "start_frame": 359663,
          "end_frame":   359789,
          "task":      "open_drawer",
          "calvin_ann":"open the drawer",
          "hf_instruction": "pull the drawer",
          "match_method": "exact" | "nn_l2"
        },
        ...
      ],
      "unmatched": [ ... rows with no plausible candidate ... ]
    }

Run
---
    python evaluate/build_hf_calvin_mapping.py \\
        --calvin_val_dir /data/calvin/task_D_D/validation \\
        --hf_parquet     /tmp/calvin_test.parquet \\
        --output         /runs/hf_calvin_mapping.json

If ``--hf_parquet`` is omitted we stream the file from the Hugging Face hub.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# CALVIN side
# ---------------------------------------------------------------------------

def _load_lang_annotations(val_dir: Path) -> Dict:
    p = val_dir / "lang_annotations" / "auto_lang_ann.npy"
    if not p.exists():
        raise FileNotFoundError(
            f"Could not find {p}. This file ships with task_D_D/validation."
        )
    data = np.load(p, allow_pickle=True).item()
    # Expected schema:
    #   data['info']['indx']      : list of (start, end) frame tuples
    #   data['info']['episodes']  : (optional)
    #   data['language']['ann']   : list of free-form annotations (str)
    #   data['language']['task']  : list of canonical task names (str)
    indx = list(data["info"]["indx"])
    ann = list(data["language"]["ann"])
    task = list(data["language"]["task"])
    assert len(indx) == len(ann) == len(task), (
        f"Inconsistent annotation lengths: indx={len(indx)} ann={len(ann)} task={len(task)}"
    )
    return {"indx": indx, "ann": ann, "task": task}


def _read_rgb_static(val_dir: Path, frame_idx: int) -> np.ndarray:
    npz_path = val_dir / f"episode_{frame_idx:07d}.npz"
    with np.load(npz_path, allow_pickle=True) as z:
        rgb = z["rgb_static"]  # uint8, (200, 200, 3)
    return np.asarray(rgb, dtype=np.uint8)


def _array_md5(a: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# HF side
# ---------------------------------------------------------------------------

_HF_TEST_URL = (
    "https://huggingface.co/datasets/SuyashAjitChavan2002/Calvin-Dataset-Final/"
    "resolve/main/data/test-00000-of-00001-f87345d1f3d9dd3f.parquet"
)


def _download_hf_parquet(dest: Path) -> Path:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mapping] downloading HF test parquet -> {dest}")
    urllib.request.urlretrieve(_HF_TEST_URL, dest)
    return dest


def _decode_image_bytes(blob: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(blob)).convert("RGB")
    if img.size != (200, 200):
        img = img.resize((200, 200), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def _load_hf_test_rows(parquet_path: Path) -> List[Dict]:
    import pyarrow.parquet as pq

    print(f"[mapping] reading {parquet_path}")
    t = pq.read_table(parquet_path)
    cols = t.column_names
    needed = {"id", "original_image", "edited_image", "edit_instruction"}
    missing = needed - set(cols)
    if missing:
        raise RuntimeError(f"HF parquet missing columns: {missing}")

    n = t.num_rows
    rows: List[Dict] = []
    ids = t["id"].to_pylist()
    instr = t["edit_instruction"].to_pylist()
    orig_col = t["original_image"]
    edit_col = t["edited_image"]
    # HuggingFace `image` dtype stores struct {bytes, path}. Handle both that
    # and the case where it's already raw bytes.
    for i in range(n):
        if i % 100 == 0:
            print(f"[mapping]   decoded {i}/{n} HF rows", end="\r", flush=True)
        rows.append({
            "row": i,
            "id": ids[i],
            "instruction": instr[i],
            "original": _decode_hf_image(orig_col[i].as_py()),
            "edited":   _decode_hf_image(edit_col[i].as_py()),
        })
    print(f"[mapping]   decoded {n}/{n} HF rows")
    return rows


def _decode_hf_image(value) -> np.ndarray:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return _decode_image_bytes(value["bytes"])
        if value.get("path"):
            return np.asarray(
                Image.open(value["path"]).convert("RGB").resize((200, 200), Image.BICUBIC),
                dtype=np.uint8,
            )
    if isinstance(value, (bytes, bytearray)):
        return _decode_image_bytes(bytes(value))
    raise TypeError(f"Unsupported HF image cell type: {type(value)} -> {value!r}")


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _build_segment_hash_index(val_dir: Path, ann: Dict, verbose: bool = True):
    """For every segment, compute md5 of rgb_static at its start and end
    frames. Return ``(hash_index, segments_meta, start_arrs, end_arrs)``."""
    n = len(ann["indx"])
    index: Dict[Tuple[str, str], int] = {}
    seg_meta: List[Dict] = []
    start_arrs: List[np.ndarray] = []
    end_arrs: List[np.ndarray] = []
    t0 = time.time()
    for i, (s, e) in enumerate(ann["indx"]):
        s_int, e_int = int(s), int(e)
        a_start = _read_rgb_static(val_dir, s_int)
        a_end = _read_rgb_static(val_dir, e_int)
        h_s = _array_md5(a_start)
        h_e = _array_md5(a_end)
        index[(h_s, h_e)] = i
        seg_meta.append({
            "seg_idx": i,
            "start_frame": s_int,
            "end_frame": e_int,
            "task": ann["task"][i],
            "calvin_ann": ann["ann"][i],
            "hash_start": h_s,
            "hash_end": h_e,
        })
        start_arrs.append(a_start)
        end_arrs.append(a_end)
        if verbose and (i + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"[mapping]   hashed {i+1}/{n} segments  ({dt:.1f}s)",
                  end="\r", flush=True)
    if verbose:
        print(f"[mapping]   hashed {n}/{n} segments")
    return index, seg_meta, np.stack(start_arrs), np.stack(end_arrs)


def _nn_match(orig: np.ndarray, edit: np.ndarray,
              start_arrs: np.ndarray, end_arrs: np.ndarray,
              candidate_idxs: np.ndarray) -> Tuple[int, float, float]:
    """Pick the candidate segment that minimises L2 dist on (orig, edit)."""
    diffs_s = start_arrs[candidate_idxs].astype(np.int32) - orig.astype(np.int32)
    diffs_e = end_arrs[candidate_idxs].astype(np.int32) - edit.astype(np.int32)
    d_s = np.sqrt((diffs_s ** 2).reshape(len(candidate_idxs), -1).mean(axis=1))
    d_e = np.sqrt((diffs_e ** 2).reshape(len(candidate_idxs), -1).mean(axis=1))
    score = d_s + d_e
    j = int(np.argmin(score))
    return int(candidate_idxs[j]), float(d_s[j]), float(d_e[j])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--calvin_val_dir", required=True,
                   help="Path to task_D_D/validation (must contain "
                        "lang_annotations/auto_lang_ann.npy and episode_*.npz).")
    p.add_argument("--hf_parquet", default="",
                   help="Local path to the HF test parquet. If omitted, "
                        "downloads from the Hugging Face hub to the cache.")
    p.add_argument("--cache_dir", default=os.path.expanduser("~/.cache/calvin_hf"),
                   help="Where to cache the downloaded HF parquet.")
    p.add_argument("--output", required=True,
                   help="Path to write the mapping JSON.")
    p.add_argument("--nn_fallback", action="store_true", default=True,
                   help="If exact hash match fails, do nearest-L2 search "
                        "(default on).")
    p.add_argument("--no_nn_fallback", dest="nn_fallback", action="store_false")
    p.add_argument("--nn_max_dist", type=float, default=20.0,
                   help="Reject NN matches whose mean per-pixel L2 dist on "
                        "the start frame exceeds this. JPEG re-encoding is "
                        "typically <5; >20 means it's not actually the same "
                        "scene.")
    args = p.parse_args()

    val_dir = Path(args.calvin_val_dir).expanduser().resolve()
    if not val_dir.exists():
        sys.exit(f"CALVIN val dir does not exist: {val_dir}")

    if args.hf_parquet:
        parquet_path = Path(args.hf_parquet).expanduser().resolve()
    else:
        parquet_path = Path(args.cache_dir).expanduser() / "test-00000-of-00001.parquet"
        if not parquet_path.exists():
            _download_hf_parquet(parquet_path)

    print("[mapping] loading CALVIN language annotations...")
    ann = _load_lang_annotations(val_dir)
    print(f"[mapping]   {len(ann['indx'])} segments in val split")

    print("[mapping] hashing rgb_static at every segment's start & end frame...")
    seg_index, seg_meta, start_arrs, end_arrs = _build_segment_hash_index(val_dir, ann)

    print("[mapping] decoding HF test rows...")
    hf_rows = _load_hf_test_rows(parquet_path)

    matches: List[Dict] = []
    unmatched: List[Dict] = []
    seg_used: Dict[int, int] = defaultdict(int)
    method_counts = defaultdict(int)

    # Pre-index segments by token overlap with their CALVIN annotation, for
    # cheap NN candidate filtering on the HF instruction.
    def _tokens(s: str) -> set:
        return {t.lower() for t in s.split() if len(t) >= 3}
    seg_tokens = [_tokens(s["calvin_ann"]) for s in seg_meta]

    n = len(hf_rows)
    for k, row in enumerate(hf_rows):
        if k % 50 == 0:
            print(f"[mapping]   matching row {k}/{n}", end="\r", flush=True)
        h_o = _array_md5(row["original"])
        h_e = _array_md5(row["edited"])
        seg_idx = seg_index.get((h_o, h_e))
        method = "exact"
        d_s = d_e = 0.0
        if seg_idx is None and args.nn_fallback:
            # Filter candidate segments by token overlap with HF instruction.
            hf_tok = _tokens(row["instruction"])
            cand = np.asarray([
                i for i, st in enumerate(seg_tokens) if hf_tok & st
            ], dtype=np.int64)
            if len(cand) == 0:
                cand = np.arange(len(seg_meta), dtype=np.int64)
            seg_idx, d_s, d_e = _nn_match(row["original"], row["edited"],
                                          start_arrs, end_arrs, cand)
            method = "nn_l2"
            if d_s > args.nn_max_dist:
                unmatched.append({
                    "hf_row": row["row"],
                    "hf_id": row["id"],
                    "hf_instruction": row["instruction"],
                    "best_seg_idx": int(seg_idx),
                    "best_task": seg_meta[seg_idx]["task"],
                    "best_d_start": d_s,
                    "best_d_end": d_e,
                    "reason": f"d_start={d_s:.2f} > nn_max_dist={args.nn_max_dist}",
                })
                continue

        if seg_idx is None:
            unmatched.append({
                "hf_row": row["row"],
                "hf_id": row["id"],
                "hf_instruction": row["instruction"],
                "reason": "no exact hash match and nn_fallback disabled",
            })
            continue

        meta = seg_meta[seg_idx]
        seg_used[seg_idx] += 1
        method_counts[method] += 1
        matches.append({
            "hf_row": row["row"],
            "hf_id": row["id"],
            "hf_instruction": row["instruction"],
            "calvin_seg_idx": int(seg_idx),
            "start_frame": meta["start_frame"],
            "end_frame": meta["end_frame"],
            "task": meta["task"],
            "calvin_ann": meta["calvin_ann"],
            "match_method": method,
            "d_start": d_s,
            "d_end": d_e,
        })
    print(f"[mapping]   matching row {n}/{n}")

    out = {
        "split": "validation",
        "calvin_val_dir": str(val_dir),
        "hf_parquet": str(parquet_path),
        "num_hf_rows": n,
        "num_segments": len(seg_meta),
        "method_counts": dict(method_counts),
        "num_matched": len(matches),
        "num_unmatched": len(unmatched),
        "num_unique_segments_used": len(seg_used),
        "max_segment_reuse": max(seg_used.values()) if seg_used else 0,
        "matches": matches,
        "unmatched": unmatched,
    }
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[mapping] wrote {out_path}")
    print(f"[mapping] matched={len(matches)}/{n}  exact={method_counts.get('exact',0)} "
          f"nn_l2={method_counts.get('nn_l2',0)}  unmatched={len(unmatched)}  "
          f"unique_segments_used={len(seg_used)}  max_reuse={out['max_segment_reuse']}")


if __name__ == "__main__":
    main()
