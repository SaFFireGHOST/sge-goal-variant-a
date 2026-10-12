"""Chain-aware GR-MG visualization.

For a given (method, chain_length, chain_instance), this script:

* Resets the env to the FIRST chain step's CALVIN start_frame.
* Sequentially rolls out N policy episodes — one per chain step — using each
  step's goal image (from CALVIN-EVALS/<method>/.../<hf_row+1>/edited_image.*)
  and instruction (edit_instruction.txt).
* Between steps the env state CARRIES OVER (no reset), mimicking the
  original consecutive-task pattern in CALVIN data collection.
* Saves an mp4 at ``--output``.

A "chain instance" is identified by ``--chain_skeletons``,
``--skeleton_idx`` and ``--start_in_chain`` (which sub-chain of the skeleton
to start at), plus ``--chain_length`` (how many steps to roll out from there).
For each step's HF row we pick ``hf_rows[0]`` of the corresponding cluster
(deterministic; comparable across methods).

A goal source is required:
  * ``--method_dir`` (CALVIN-EVALS/<method>/), OR
  * ``--hf_parquet_goal`` (HF parquet — uses dataset's own ``edited_image``).
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
from typing import List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning import seed_everything
from termcolor import colored

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from wrapper.model_wrapper import CustomModel  # type: ignore  # noqa: E402

EP_LEN_PER_STEP = 360


# ---- IO helpers (mostly shared with eval_method.py) ------------------------

def _load_image_to_200(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.size != (200, 200):
        img = img.resize((200, 200), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def _decode_hf_image(value) -> np.ndarray:
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
    sub = [p for p in method_dir.iterdir() if p.is_dir()]
    if len(sub) != 1:
        raise ValueError(f"Expected exactly one sub-folder in {method_dir}")
    return sub[0]


def _load_goal_for_row(
    hf_row: int, method_root: Optional[Path], parquet_t=None
) -> dict:
    if method_root is not None:
        sample_dir = method_root / str(hf_row + 1)
        for stem in ("edited_image",):
            for ext in ("jpg", "jpeg", "png"):
                p = sample_dir / f"{stem}.{ext}"
                if p.exists():
                    edited = _load_image_to_200(p)
                    break
            else:
                raise FileNotFoundError(f"{stem} not found in {sample_dir}")
        instr = (sample_dir / "edit_instruction.txt").read_text().strip()
        return {"edited": edited, "instruction": instr}
    else:
        return {
            "edited": _decode_hf_image(parquet_t["edited_image"][hf_row].as_py()),
            "instruction": parquet_t["edit_instruction"][hf_row].as_py(),
        }


# ---- Env / policy / oracle -------------------------------------------------

def _make_env(dataset_path, observation_space, device_id):
    val_folder = Path(dataset_path) / "validation"
    from wrapper.calvin_env_wrapper_raw import CalvinEnvWrapperRaw  # type: ignore

    return CalvinEnvWrapperRaw(val_folder, observation_space,
                               torch.device("cuda", device_id))


def _read_npz_init(val_dir: Path, frame_idx: int):
    p = val_dir / f"episode_{frame_idx:07d}.npz"
    with np.load(p, allow_pickle=True) as z:
        return np.asarray(z["robot_obs"]), np.asarray(z["scene_obs"])


def _load_policy(ckpt_dir: str, epoch: int, config_path: str, device_id: int):
    with open(config_path) as f:
        configs = json.load(f)
    ckpt_path = None
    for ckpt_file in os.listdir(ckpt_dir):
        m = re.search(r"epoch=(\d+)", ckpt_file)
        if m and int(m.group(1)) == epoch:
            ckpt_path = os.path.join(ckpt_dir, ckpt_file)
            break
    if ckpt_path is None:
        raise FileNotFoundError(f"No ckpt epoch={epoch} in {ckpt_dir}")
    return CustomModel(
        ckpt_path=ckpt_path, configs=configs,
        device=torch.device("cuda", device_id),
    )


def _load_task_oracle():
    import hydra
    from omegaconf import OmegaConf

    conf_dir = Path("./calvin/calvin_models/conf")
    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    return hydra.utils.instantiate(task_cfg)


# ---- Frame composition -----------------------------------------------------

def _annotate(img: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(img).convert("RGB")
    drw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12
        )
    except OSError:
        font = ImageFont.load_default()
    drw.rectangle([(0, 0), (pil.size[0], 16)], fill=(0, 0, 0))
    drw.text((4, 1), text, fill=(255, 255, 255), font=font)
    return np.asarray(pil, dtype=np.uint8)


def _compose_frame(
    static_rgb: np.ndarray, gripper_rgb: np.ndarray, goal_rgb: np.ndarray,
    step_idx: int, n_steps: int, instruction: str, success_so_far: int,
) -> np.ndarray:
    panels = [
        _annotate(static_rgb,   f"static  step {step_idx+1}/{n_steps}"),
        _annotate(gripper_rgb,  "gripper"),
        _annotate(goal_rgb,     f"goal  succ={success_so_far}/{step_idx+1}"),
    ]
    panels = [
        np.asarray(
            Image.fromarray(p).convert("RGB").resize((200, 200), Image.BICUBIC),
            dtype=np.uint8,
        )
        for p in panels
    ]
    row = np.concatenate(panels, axis=1)

    # Bottom strip with the active instruction.
    strip = np.zeros((26, row.shape[1], 3), dtype=np.uint8)
    pil = Image.fromarray(strip)
    drw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
        )
    except OSError:
        font = ImageFont.load_default()
    drw.text((4, 6), instruction[:90], fill=(255, 255, 255), font=font)
    strip = np.asarray(pil, dtype=np.uint8)
    return np.concatenate([row, strip], axis=0)


def _write_video(frames, out_path: str, fps: int) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio  # type: ignore
        imageio.mimsave(out_path, frames, fps=fps, codec="libx264",
                        quality=8, macro_block_size=1)
        return
    except Exception as e_io:
        print(f"[viz_chain] imageio failed ({e_io}); using opencv...")
    import cv2  # type: ignore
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


# ---- Chain instance loader -------------------------------------------------

def _build_chain_instance(skeleton: dict, start: int, length: int) -> List[dict]:
    """Take ``length`` consecutive clusters from ``skeleton['clusters']`` starting
    at ``start``. Returns a list of ``{hf_row, task, start_frame, end_frame}``."""
    clusters = skeleton["clusters"]
    if start + length > len(clusters):
        raise ValueError(
            f"skeleton (id={skeleton['chain_id']}) has only "
            f"{len(clusters)} clusters; cannot start={start} length={length}"
        )
    out = []
    for cl in clusters[start : start + length]:
        out.append(
            {
                "hf_row": int(cl["hf_rows"][0]),
                "task": cl["task"],
                "start_frame": int(cl["start_frame"]),
                "end_frame": int(cl["end_frame"]),
            }
        )
    return out


# ---- Driver ----------------------------------------------------------------

def main():
    seed_everything(0, workers=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])

    p.add_argument("--dataset_path", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--epoch", type=int, default=47)
    p.add_argument("--config_path", required=True)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--mapping_json", required=True)

    p.add_argument("--chain_skeletons", required=True, type=Path)
    p.add_argument("--skeleton_idx", type=int, required=True,
                   help="Index into chain_skeletons[*].chain_skeletons")
    p.add_argument("--start_in_chain", type=int, default=0,
                   help="Sub-chain start position within the skeleton.")
    p.add_argument("--chain_length", type=int, required=True)

    p.add_argument("--method_dir", type=Path, default=None)
    p.add_argument("--hf_parquet_goal", type=Path, default=None)

    p.add_argument("--output", required=True)
    p.add_argument("--ep_len_per_step", type=int, default=EP_LEN_PER_STEP)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--continue_on_failure", action="store_true",
                   help="If a step fails, still proceed to the next step.")
    args = p.parse_args()

    if (args.method_dir is None) == (args.hf_parquet_goal is None):
        raise SystemExit("Specify exactly one of --method_dir / --hf_parquet_goal")

    # -- Load chain instance ---
    with args.chain_skeletons.open() as fh:
        sk_doc = json.load(fh)
    skeleton = sk_doc["chain_skeletons"][args.skeleton_idx]
    instance = _build_chain_instance(skeleton, args.start_in_chain,
                                     args.chain_length)
    print(colored(
        f"[viz_chain] skeleton={args.skeleton_idx} start={args.start_in_chain}"
        f" length={args.chain_length}  steps:",
        "cyan",
    ))
    for k, st in enumerate(instance):
        print(f"  step {k+1}: hf_row={st['hf_row']:4d}  task={st['task']:30s}"
              f"  frames=[{st['start_frame']}-{st['end_frame']}]")

    # -- Goal source ---
    method_root = None
    parquet_t = None
    if args.method_dir is not None:
        method_root = _find_method_subdir(args.method_dir)
        print(f"[viz_chain] goal source: method_dir={method_root}")
    else:
        import pyarrow.parquet as pq
        parquet_t = pq.read_table(args.hf_parquet_goal)
        print(f"[viz_chain] goal source: HF parquet")

    # -- Policy / env / oracle ---
    print(f"[viz_chain] loading policy on cuda:{args.device_id}")
    model = _load_policy(args.ckpt_dir, args.epoch, args.config_path,
                         args.device_id)
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

    # -- Reset to FIRST step's start_frame ---
    first = instance[0]
    robot_obs, scene_obs = _read_npz_init(val_dir, first["start_frame"])
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    frames: List[np.ndarray] = []
    success_so_far = 0
    overall_success = True

    for step_idx, st in enumerate(instance):
        hf_row = st["hf_row"]
        task = st["task"]
        grow = _load_goal_for_row(hf_row, method_root, parquet_t)
        goal_rgb = grow["edited"]
        instruction = grow["instruction"]

        print(colored(
            f"[viz_chain] step {step_idx+1}/{len(instance)}: hf_row={hf_row} "
            f"task={task} instr='{instruction}'", "yellow",
        ))

        model.reset()  # reset policy hidden state (env state preserved!)
        start_info = env.get_info()
        step_success = False
        steps_used = args.ep_len_per_step
        obs = env.get_obs()
        for j in range(args.ep_len_per_step):
            static_rgb = obs["rgb_obs"]["rgb_static"]
            hand_rgb = obs["rgb_obs"]["rgb_gripper"]
            frames.append(_compose_frame(
                static_rgb, hand_rgb, goal_rgb,
                step_idx, len(instance),
                instruction, success_so_far,
            ))
            action, _ = model.step(obs, [deepcopy(goal_rgb)], [instruction])
            obs, _, _, current_info = env.step(action)
            if len(task_oracle.get_task_info_for_set(
                start_info, current_info, {task}
            )) > 0:
                step_success = True
                steps_used = j + 1
                # Capture the success frame.
                frames.append(_compose_frame(
                    obs["rgb_obs"]["rgb_static"], obs["rgb_obs"]["rgb_gripper"],
                    goal_rgb, step_idx, len(instance),
                    instruction, success_so_far + 1,
                ))
                break

        success_so_far += int(step_success)
        if step_success:
            print(colored(
                f"[viz_chain]   step {step_idx+1} SUCCESS in {steps_used} steps",
                "green",
            ))
        else:
            print(colored(
                f"[viz_chain]   step {step_idx+1} FAILED",
                "red",
            ))
            overall_success = False
            if not args.continue_on_failure:
                break

    print(colored(
        f"[viz_chain] DONE. chain_success={overall_success}  "
        f"steps_succeeded={success_so_far}/{len(instance)}  "
        f"writing {args.output}",
        "green" if overall_success else "yellow",
    ))
    _write_video(frames, args.output, fps=args.fps)


if __name__ == "__main__":
    main()
