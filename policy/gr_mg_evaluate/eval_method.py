# Copyright (c) 2026
"""Partition-aware GR-MG evaluator over the 1087 HF test rows.

Same model/env/oracle wiring as ``eval_hf.py``, but:

1. Goal images are loaded from one of two sources:

   * ``--method_dir <CALVIN-EVALS/<method>/calvin_abcd_*_test_data>``:
     Each ``i+1`` (1-indexed) subdirectory contains ``edited_image.{jpg,png}``
     and ``edit_instruction.txt`` for HF row ``i``. Used to evaluate a
     specific goal-image generation method.
   * ``--hf_parquet_goal <calvin_test.parquet>``:
     Goals come from the HF dataset's own ``edited_image`` column
     (ground-truth case).

2. Supports ``--row_start`` / ``--row_end`` (HALF-OPEN, 0-indexed) so the
   1087 rows can be split across N parallel GPU workers. Each worker writes
   its own ``per_sample.jsonl`` shard under ``--output_dir`` and an aggregator
   later merges them.

The mapping JSON is still required: it provides the CALVIN ``start_frame``
(env init), canonical ``task`` (oracle key) and HF↔CALVIN bookkeeping.

Each shard writes:

* ``per_sample_<row_start>_<row_end>.jsonl`` — one record per HF row.
* ``shard_done_<row_start>_<row_end>.json`` — sentinel with summary stats.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from pytorch_lightning import seed_everything
from termcolor import colored

# ``wrapper`` lives at evaluate/wrapper/.
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from wrapper.model_wrapper import CustomModel  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Image / instruction loaders
# ---------------------------------------------------------------------------

def _load_image_to_200(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.size != (200, 200):
        img = img.resize((200, 200), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def _decode_hf_image_value(value) -> np.ndarray:
    if isinstance(value, dict) and value.get("bytes") is not None:
        img = Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    elif isinstance(value, dict) and value.get("path"):
        img = Image.open(value["path"]).convert("RGB")
    elif isinstance(value, (bytes, bytearray)):
        img = Image.open(io.BytesIO(bytes(value))).convert("RGB")
    else:
        raise TypeError(f"Unsupported HF image cell: {type(value)}")
    if img.size != (200, 200):
        img = img.resize((200, 200), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def _find_method_subdir(method_dir: Path) -> Path:
    """``CALVIN-EVALS/<method>/`` always has exactly one sub-folder
    (``calvin_abcd_test_data`` or ``calvin_abcd_ier_fer_gt_test_data``)."""
    sub = [p for p in method_dir.iterdir() if p.is_dir()]
    if len(sub) != 1:
        raise ValueError(
            f"Expected exactly one sub-folder in {method_dir}, got {len(sub)}"
        )
    return sub[0]


def _resolve_method_files(sample_dir: Path) -> Dict[str, Path]:
    """Return paths to original image, edited image and instruction text."""
    out: Dict[str, Path] = {}
    for stem, key in (
        ("original_image", "original"),
        ("edited_image", "edited"),
    ):
        for ext in ("jpg", "jpeg", "png"):
            cand = sample_dir / f"{stem}.{ext}"
            if cand.exists():
                out[key] = cand
                break
        if key not in out:
            raise FileNotFoundError(f"Missing {stem}.* in {sample_dir}")
    instr = sample_dir / "edit_instruction.txt"
    if not instr.exists():
        raise FileNotFoundError(f"Missing edit_instruction.txt in {sample_dir}")
    out["instruction_txt"] = instr
    return out


def _load_method_row(method_root: Path, hf_row: int) -> Dict:
    """Load goal image + instruction for HF row ``hf_row`` (0-indexed)
    from a CALVIN-EVALS method directory."""
    sample_dir = method_root / str(hf_row + 1)   # 1-indexed naming
    files = _resolve_method_files(sample_dir)
    return {
        "instruction": files["instruction_txt"].read_text().strip(),
        "edited": _load_image_to_200(files["edited"]),
    }


def _load_hf_parquet_rows(
    parquet_path: Path, row_start: int, row_end: int
) -> Dict[int, Dict]:
    import pyarrow.parquet as pq

    print(f"[eval_method] reading {parquet_path}")
    t = pq.read_table(parquet_path)
    ids = t["id"].to_pylist()
    instr = t["edit_instruction"].to_pylist()
    edit_col = t["edited_image"]
    out: Dict[int, Dict] = {}
    for i in range(row_start, row_end):
        out[i] = {
            "id": ids[i],
            "instruction": instr[i],
            "edited": _decode_hf_image_value(edit_col[i].as_py()),
        }
    print(f"[eval_method]   loaded HF rows [{row_start}, {row_end})")
    return out


# ---------------------------------------------------------------------------
# Env / policy / oracle (same as eval_hf.py)
# ---------------------------------------------------------------------------

def _make_env(dataset_path: str, observation_space: dict, device_id: int):
    val_folder = Path(dataset_path) / "validation"
    from wrapper.calvin_env_wrapper_raw import CalvinEnvWrapperRaw  # type: ignore

    device = torch.device("cuda", device_id)
    return CalvinEnvWrapperRaw(val_folder, observation_space, device)


def _read_npz_init(val_dir: Path, frame_idx: int):
    p = val_dir / f"episode_{frame_idx:07d}.npz"
    with np.load(p, allow_pickle=True) as z:
        return np.asarray(z["robot_obs"]), np.asarray(z["scene_obs"])


def _load_policy(ckpt_dir: str, epoch: int, config_path: str, device_id: int):
    with open(config_path, "r") as f:
        configs = json.load(f)
    ckpt_path: Optional[str] = None
    for ckpt_file in os.listdir(ckpt_dir):
        m = re.search(r"epoch=(\d+)", ckpt_file)
        if m and int(m.group(1)) == epoch:
            ckpt_path = os.path.join(ckpt_dir, ckpt_file)
            break
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint matching epoch={epoch} in {ckpt_dir}"
        )
    device = torch.device("cuda", device_id)
    return CustomModel(ckpt_path=ckpt_path, configs=configs, device=device)


def _load_task_oracle():
    import hydra
    from omegaconf import OmegaConf

    conf_dir = Path("./calvin/calvin_models/conf")
    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    return hydra.utils.instantiate(task_cfg)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    seed_everything(0, workers=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])

    # Policy / env
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--epoch", type=int, default=47)
    p.add_argument("--config_path", required=True)
    p.add_argument("--device_id", type=int, default=0)

    # Mapping (always needed for env reset + canonical task)
    p.add_argument("--mapping_json", required=True)

    # Goal source: exactly one of these
    p.add_argument("--method_dir", type=Path, default=None,
                   help="Path to a CALVIN-EVALS/<method>/ directory")
    p.add_argument("--hf_parquet_goal", type=Path, default=None,
                   help="HF parquet to use ground-truth ``edited_image`` as goal")

    # Sharding
    p.add_argument("--row_start", type=int, default=0,
                   help="Inclusive 0-indexed HF row start.")
    p.add_argument("--row_end", type=int, default=1087,
                   help="Exclusive 0-indexed HF row end.")

    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--ep_len", type=int, default=360)
    p.add_argument("--method_name", default="ground_truth",
                   help="Used in shard sentinel/output filenames.")
    p.add_argument("--conditioning", default="both",
                   choices=["both", "text_only", "goal_only"],
                   help="Conditioning mode: both (default), text_only, goal_only.")

    args = p.parse_args()

    if (args.method_dir is None) == (args.hf_parquet_goal is None):
        raise SystemExit(
            "Specify exactly one of --method_dir or --hf_parquet_goal"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_tag = f"{args.row_start}_{args.row_end}"
    per_sample_path = args.output_dir / f"per_sample_{shard_tag}.jsonl"
    sentinel_path = args.output_dir / f"shard_done_{shard_tag}.json"

    # -- Load mapping -------------------------------------------------------
    print(f"[eval_method] loading mapping {args.mapping_json}")
    with open(args.mapping_json, "r") as f:
        mapping = json.load(f)
    matches_by_row = {m["hf_row"]: m for m in mapping["matches"]}
    print(f"[eval_method]   {len(matches_by_row)}/{mapping.get('num_hf_rows', 1087)} "
          f"HF rows have a CALVIN match")

    # -- Load goal sources --------------------------------------------------
    method_root: Optional[Path] = None
    parquet_rows: Dict[int, Dict] = {}
    if args.method_dir is not None:
        method_root = _find_method_subdir(args.method_dir)
        print(f"[eval_method] goal source: method_dir={method_root}")
    else:
        parquet_rows = _load_hf_parquet_rows(
            args.hf_parquet_goal, args.row_start, args.row_end
        )
        print(f"[eval_method] goal source: HF parquet ground truth")

    # -- Load policy / env / oracle ----------------------------------------
    print(f"[eval_method] loading GR-MG (epoch {args.epoch}) on cuda:{args.device_id}")
    model = _load_policy(args.ckpt_dir, args.epoch, args.config_path, args.device_id)

    observation_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }
    env = _make_env(args.dataset_path, observation_space, args.device_id)
    val_dir = Path(args.dataset_path) / "validation"
    task_oracle = _load_task_oracle()

    # -- Roll out shard -----------------------------------------------------
    open(per_sample_path, "w").close()
    success_count = 0
    evaluated = 0
    skipped = 0

    print(colored(
        f"[eval_method] shard rows [{args.row_start}, {args.row_end})  "
        f"out={per_sample_path}",
        "cyan",
    ))

    for hf_row in range(args.row_start, args.row_end):
        if hf_row not in matches_by_row:
            rec = {
                "hf_row": hf_row,
                "success": False,
                "skipped": True,
                "reason": "no_calvin_match",
            }
            with open(per_sample_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            skipped += 1
            continue

        m = matches_by_row[hf_row]
        task = m["task"]
        start_frame = int(m["start_frame"])

        # Goal + instruction
        if method_root is not None:
            try:
                grow = _load_method_row(method_root, hf_row)
            except FileNotFoundError as e:
                print(f"[eval_method]   WARN row {hf_row}: {e}; skipping")
                rec = {"hf_row": hf_row, "success": False,
                       "skipped": True, "reason": f"method_files_missing"}
                with open(per_sample_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                skipped += 1
                continue
        else:
            grow = parquet_rows[hf_row]

        try:
            robot_obs, scene_obs = _read_npz_init(val_dir, start_frame)
        except FileNotFoundError as e:
            print(f"[eval_method]   WARN row {hf_row}: missing {e}; skipping")
            rec = {"hf_row": hf_row, "success": False,
                   "skipped": True, "reason": "npz_missing"}
            with open(per_sample_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            skipped += 1
            continue

        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        model.reset()
        start_info = env.get_info()
        goal_rgb = grow["edited"]
        lang = grow["instruction"]

        # Conditioning ablation: mask out goal image or language
        if args.conditioning == "text_only":
            # Black image (no goal information)
            ablation_goal = np.zeros_like(goal_rgb)
        elif args.conditioning == "goal_only":
            ablation_goal = deepcopy(goal_rgb)
            lang = ""
        else:  # "both"
            ablation_goal = deepcopy(goal_rgb)

        success = False
        steps_used = args.ep_len
        obs = env.get_obs()
        for i in range(args.ep_len):
            action, _ = model.step(obs, [deepcopy(ablation_goal)], [lang])
            obs, _, _, current_info = env.step(action)
            if len(task_oracle.get_task_info_for_set(
                start_info, current_info, {task}
            )) > 0:
                success = True
                steps_used = i + 1
                break

        success_count += int(success)
        evaluated += 1

        rec = {
            "hf_row": hf_row,
            "hf_id": m["hf_id"],
            "task": task,
            "instruction_used": lang,
            "calvin_ann": m["calvin_ann"],
            "start_frame": start_frame,
            "end_frame": int(m["end_frame"]),
            "success": bool(success),
            "skipped": False,
            "steps_used": steps_used,
        }
        with open(per_sample_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        running_sr = success_count / max(1, evaluated)
        print(f"[eval_method] row {hf_row:4d}  SR={running_sr:.3f}  "
              f"task={task:30s}  success={int(success)}", flush=True)

    summary = {
        "method_name": args.method_name,
        "row_start": args.row_start,
        "row_end": args.row_end,
        "evaluated": evaluated,
        "skipped": skipped,
        "success_count": success_count,
        "shard_success_rate": success_count / max(1, evaluated),
    }
    with open(sentinel_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(colored(
        f"[eval_method] shard done  "
        f"SR={summary['shard_success_rate']:.4f}  "
        f"({success_count}/{evaluated})  skipped={skipped}",
        "green",
    ))


if __name__ == "__main__":
    main()
