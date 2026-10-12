# Copyright (c) 2026
"""Evaluate GR-MG on the 1087 test rows of
``SuyashAjitChavan2002/Calvin-Dataset-Final``, using each row's ``edited_image``
as the *fixed final goal image* and its ``edit_instruction`` as the language
input.

For each HF row this driver:

1. Looks up the matching CALVIN-D val segment (start frame, end frame, task)
   from the mapping JSON produced by ``build_hf_calvin_mapping.py``.
2. Resets the PyBullet env to the ``robot_obs`` / ``scene_obs`` stored at
   that start frame's ``episode_<start>.npz``.
3. Runs the GR-MG policy for up to ``--ep_len`` steps, conditioning every
   step on the *same* HF goal image (no sub-goals).
4. Calls the official CALVIN task oracle with the recovered canonical task
   name to decide success.

Outputs in ``--output_dir``:

* ``per_sample.jsonl`` — one line per HF row with success flag, num steps,
  task, instruction, etc.
* ``summary.json``     — aggregate success rate, plus per-task success rate.
* ``success_rate.txt`` — running-average lines, in the same shape as the
  upstream GR-MG ``success_rate.txt``.

Run::

    python evaluate/eval_hf.py \\
        --dataset_path  /data/calvin/task_D_D \\
        --ckpt_dir      /ckpts/gr_mg \\
        --epoch         47 \\
        --config_path   policy/config/train.json \\
        --mapping_json  /runs/hf_calvin_mapping.json \\
        --hf_parquet    /tmp/calvin_test.parquet \\
        --output_dir    /runs/eval_hf_gt_final
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from termcolor import colored

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from wrapper.model_wrapper import CustomModel


def _make_env(dataset_path: str, observation_space: dict, device_id: int):
    val_folder = Path(dataset_path) / "validation"
    from wrapper.calvin_env_wrapper_raw import CalvinEnvWrapperRaw

    device = torch.device("cuda", device_id)
    return CalvinEnvWrapperRaw(val_folder, observation_space, device)


def _decode_hf_image_value(value) -> np.ndarray:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            img = Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        elif value.get("path"):
            img = Image.open(value["path"]).convert("RGB")
        else:
            raise ValueError(f"Unrecognised HF image cell: {value!r}")
    elif isinstance(value, (bytes, bytearray)):
        img = Image.open(io.BytesIO(bytes(value))).convert("RGB")
    else:
        raise TypeError(f"Unsupported HF image cell type: {type(value)}")
    if img.size != (200, 200):
        img = img.resize((200, 200), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def _load_hf_test_rows(parquet_path: Path) -> Dict[int, dict]:
    """Return {row_idx: {'id', 'instruction', 'edited'}} keyed by parquet row index."""
    import pyarrow.parquet as pq

    print(f"[eval_hf] reading {parquet_path}")
    t = pq.read_table(parquet_path)
    ids = t["id"].to_pylist()
    instr = t["edit_instruction"].to_pylist()
    edit_col = t["edited_image"]
    out: Dict[int, dict] = {}
    n = t.num_rows
    for i in range(n):
        out[i] = {
            "id": ids[i],
            "instruction": instr[i],
            "edited": _decode_hf_image_value(edit_col[i].as_py()),
        }
        if (i + 1) % 100 == 0:
            print(f"[eval_hf]   decoded {i+1}/{n} HF rows", end="\r", flush=True)
    print(f"[eval_hf]   decoded {n}/{n} HF rows")
    return out


def _read_npz_init(val_dir: Path, frame_idx: int):
    p = val_dir / f"episode_{frame_idx:07d}.npz"
    with np.load(p, allow_pickle=True) as z:
        robot_obs = np.asarray(z["robot_obs"])
        scene_obs = np.asarray(z["scene_obs"])
    return robot_obs, scene_obs


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
    model = CustomModel(ckpt_path=ckpt_path, configs=configs, device=device)
    return model


def _load_task_oracle():
    conf_dir = Path("./calvin/calvin_models/conf")
    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    return hydra.utils.instantiate(task_cfg)


def main():
    seed_everything(0, workers=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset_path", required=True,
                   help="CALVIN split root, e.g. /.../task_D_D")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--epoch", type=int, default=47)
    p.add_argument("--config_path", required=True)
    p.add_argument("--device_id", type=int, default=0)

    p.add_argument("--mapping_json", required=True,
                   help="Path to the JSON produced by build_hf_calvin_mapping.py")
    p.add_argument("--hf_parquet", required=True,
                   help="Path to the HF test parquet (must match the one used "
                        "when building the mapping).")

    p.add_argument("--output_dir", required=True)
    p.add_argument("--ep_len", type=int, default=360,
                   help="Max env steps per HF row.")
    p.add_argument("--limit", type=int, default=-1,
                   help="If >0, only evaluate the first N matched HF rows. "
                        "Useful for quick smoke tests.")
    p.add_argument(
        "--instruction_source",
        choices=["hf", "calvin"],
        default="hf",
        help="Which instruction string to feed the policy. 'hf' (default) "
             "uses the HF edit_instruction. 'calvin' uses the canonical CALVIN "
             "annotation recovered via the mapping.",
    )
    args = p.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sr_path = out_dir / "success_rate.txt"
    per_sample_path = out_dir / "per_sample.jsonl"
    summary_path = out_dir / "summary.json"

    print(f"[eval_hf] loading mapping {args.mapping_json}")
    with open(args.mapping_json, "r") as f:
        mapping = json.load(f)
    matches: List[dict] = mapping["matches"]
    if args.limit > 0:
        matches = matches[: args.limit]
    print(f"[eval_hf]   {len(matches)} matched HF rows  "
          f"(unmatched in mapping: {mapping.get('num_unmatched', 0)})")

    hf_rows = _load_hf_test_rows(Path(args.hf_parquet))

    print(f"[eval_hf] loading GR-MG policy (epoch {args.epoch}) on cuda:{args.device_id}")
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

    success_count = 0
    per_task_total: Dict[str, int] = defaultdict(int)
    per_task_success: Dict[str, int] = defaultdict(int)

    # Truncate output files.
    open(sr_path, "w").close()
    open(per_sample_path, "w").close()

    print(colored(
        f"[eval_hf] evaluating {len(matches)} HF samples  "
        f"instruction_source={args.instruction_source}  out={out_dir}",
        "cyan",
    ))

    for k, m in enumerate(matches):
        row = hf_rows[m["hf_row"]]
        task = m["task"]
        start_frame = int(m["start_frame"])
        try:
            robot_obs, scene_obs = _read_npz_init(val_dir, start_frame)
        except FileNotFoundError as e:
            print(f"[eval_hf]   WARN row {m['hf_row']}: missing {e}; skipping")
            continue

        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        model.reset()
        start_info = env.get_info()
        goal_rgb = row["edited"]
        if args.instruction_source == "hf":
            lang = row["instruction"]
        else:
            lang = m["calvin_ann"]

        success = False
        steps_used = args.ep_len
        obs = env.get_obs()
        for i in range(args.ep_len):
            action, _progress = model.step(obs, [deepcopy(goal_rgb)], [lang])
            obs, _, _, current_info = env.step(action)
            current_task_info = task_oracle.get_task_info_for_set(
                start_info, current_info, {task}
            )
            if len(current_task_info) > 0:
                success = True
                steps_used = i + 1
                break

        success_count += int(success)
        per_task_total[task] += 1
        per_task_success[task] += int(success)

        rec = {
            "k": k,
            "hf_row": m["hf_row"],
            "hf_id": m["hf_id"],
            "task": task,
            "instruction_used": lang,
            "hf_instruction": row["instruction"],
            "calvin_ann": m["calvin_ann"],
            "start_frame": start_frame,
            "end_frame": int(m["end_frame"]),
            "match_method": m.get("match_method", ""),
            "success": bool(success),
            "steps_used": steps_used,
        }
        with open(per_sample_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        running_sr = success_count / (k + 1)
        with open(sr_path, "a") as f:
            f.write(f"{k+1}/{len(matches)}: SR={running_sr:.4f} "
                    f"({success_count}/{k+1})  task={task}  "
                    f"success={int(success)}\n")
        print(f"[eval_hf] {k+1:4d}/{len(matches)}  SR={running_sr:.3f}  "
              f"task={task:30s}  success={int(success)}",
              flush=True)

    summary = {
        "num_evaluated": len(matches),
        "num_success": success_count,
        "success_rate": success_count / max(1, len(matches)),
        "per_task": {
            t: {
                "total": per_task_total[t],
                "success": per_task_success[t],
                "sr": per_task_success[t] / per_task_total[t]
                if per_task_total[t] else 0.0,
            }
            for t in sorted(per_task_total)
        },
        "args": vars(args),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(colored(
        f"[eval_hf] done  SR={summary['success_rate']:.4f}  "
        f"({success_count}/{len(matches)})  out={out_dir}",
        "green",
    ))


if __name__ == "__main__":
    main()
