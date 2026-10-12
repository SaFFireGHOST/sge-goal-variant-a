# Copyright (c) 2026
# Visualisation utility: render a CALVIN simulation video of a chosen test
# sample using GR-MG with a user-supplied final goal image.
#
# The test sample is identified by ``--sequence_idx`` and ``--subtask_idx``
# referring to the same enumeration as ``evaluate/eval.py`` (i.e. the official
# CALVIN long-horizon test sequences returned by ``get_sequences``).
#
# Output: an mp4 video of the static-camera frames produced as the policy
# unrolls in the CALVIN PyBullet environment, plus a side panel showing the
# fixed final goal image you supplied.
#
# Example::
#
#     python evaluate/visualize.py \
#         --dataset_path  /data/calvin/task_D_D \
#         --ckpt_dir      /ckpts/gr_mg \
#         --epoch         47 \
#         --config_path   policy/config/train.json \
#         --sequence_idx  3 \
#         --subtask_idx   0 \
#         --goal_image    /goals/seq3_sub0.png \
#         --output        /tmp/seq3_sub0.mp4

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from termcolor import colored

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition

from goal_image_provider import StaticGoalImageProvider, _load_image_as_static_rgb
from wrapper.model_wrapper import CustomModel

EP_LEN_DEFAULT = 360


def _make_env(dataset_path, observation_space, device_id):
    val_folder = Path(dataset_path) / "validation"
    from wrapper.calvin_env_wrapper_raw import CalvinEnvWrapperRaw

    device = torch.device("cuda", device_id)
    return CalvinEnvWrapperRaw(val_folder, observation_space, device)


def _compose_frame(static_rgb: np.ndarray, gripper_rgb: np.ndarray,
                   goal_rgb: np.ndarray) -> np.ndarray:
    """Side-by-side: [ static-cam | gripper-cam | goal ]. All resized to 200x200."""
    panels = []
    for img in (static_rgb, gripper_rgb, goal_rgb):
        pil = Image.fromarray(img).convert("RGB").resize((200, 200), Image.BICUBIC)
        panels.append(np.asarray(pil, dtype=np.uint8))
    return np.concatenate(panels, axis=1)


def _write_video(frames, out_path: str, fps: int) -> None:
    """Try imageio first, fall back to opencv. Save as mp4."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio  # type: ignore

        imageio.mimsave(out_path, frames, fps=fps, codec="libx264",
                        quality=8, macro_block_size=1)
        return
    except Exception as e_imageio:
        print(f"[visualize] imageio failed ({e_imageio}); trying opencv...")
    try:
        import cv2  # type: ignore

        h, w, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
    except Exception as e_cv2:
        # Last-resort: dump frames as PNGs in a folder next to out_path.
        out_dir = out_path + "_frames"
        os.makedirs(out_dir, exist_ok=True)
        for i, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(out_dir, f"{i:05d}.png"))
        raise RuntimeError(
            "Neither imageio nor opencv could write the mp4. "
            f"Frames dumped to {out_dir}. Last error: {e_cv2}"
        )


def _decode_hf_image(value) -> np.ndarray:
    """Decode an HF parquet image cell to a (200,200,3) uint8 array."""
    import io as _io
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            img = Image.open(_io.BytesIO(value["bytes"])).convert("RGB")
        elif value.get("path"):
            img = Image.open(value["path"]).convert("RGB")
        else:
            raise ValueError(f"Unrecognised HF image cell: {value!r}")
    elif isinstance(value, (bytes, bytearray)):
        img = Image.open(_io.BytesIO(bytes(value))).convert("RGB")
    else:
        raise TypeError(f"Unsupported HF image cell type: {type(value)}")
    if img.size != (200, 200):
        img = img.resize((200, 200), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def main():
    seed_everything(0, workers=True)
    p = argparse.ArgumentParser(
        description="Render a CALVIN simulation video for a chosen test sample, "
                    "conditioning GR-MG on a user-supplied final goal image."
    )
    p.add_argument("--dataset_path", required=True,
                   help="CALVIN split root, e.g. /.../task_D_D")
    p.add_argument("--ckpt_dir", required=True,
                   help="Directory holding the GR-MG policy ckpt files.")
    p.add_argument("--epoch", type=int, default=47, help="Policy ckpt epoch.")
    p.add_argument("--config_path", required=True,
                   help="Path to the policy config JSON.")
    p.add_argument("--device_id", type=int, default=0)

    # ---- HF-row mode (preferred) ----
    p.add_argument("--hf_row", type=int, default=-1,
                   help="HF parquet row index to visualise. When set, uses "
                        "--mapping_json and --hf_parquet to determine the "
                        "correct env reset, goal image, and instruction.")
    p.add_argument("--mapping_json", default="",
                   help="Path to hf_calvin_mapping.json (required with --hf_row).")
    p.add_argument("--hf_parquet", default="",
                   help="Path to the HF test parquet (required with --hf_row).")

    # ---- Legacy sequence mode ----
    p.add_argument("--sequence_idx", type=int, default=-1,
                   help="(Legacy) Which long-horizon test sequence to visualise.")
    p.add_argument("--subtask_idx", type=int, default=0)
    p.add_argument("--num_sequences", type=int, default=1000)
    p.add_argument("--goal_image", default="",
                   help="(Legacy) Path to a user-supplied goal image file.")

    # ---- Common ----
    p.add_argument("--output", required=True,
                   help="Path to the output .mp4 file.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--ep_len", type=int, default=EP_LEN_DEFAULT)
    p.add_argument("--stop_on_success", action="store_true", default=True)
    p.add_argument("--no_stop_on_success", dest="stop_on_success",
                   action="store_false")

    args = p.parse_args()

    use_hf_mode = args.hf_row >= 0

    # Load policy.
    with open(args.config_path, "r") as f:
        configs = json.load(f)

    ckpt_path = None
    for ckpt_file in os.listdir(args.ckpt_dir):
        m = re.search(r"epoch=(\d+)", ckpt_file)
        if m and int(m.group(1)) == args.epoch:
            ckpt_path = os.path.join(args.ckpt_dir, ckpt_file)
            break
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint matching epoch={args.epoch} in {args.ckpt_dir}"
        )

    device = torch.device("cuda", args.device_id)
    model = CustomModel(ckpt_path=ckpt_path, configs=configs, device=device)

    observation_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }
    env = _make_env(args.dataset_path, observation_space, args.device_id)

    val_dir = Path(args.dataset_path) / "validation"
    conf_dir = Path("./calvin/calvin_models/conf")
    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)

    if use_hf_mode:
        # ----- HF-row mode: correct env reset + real goal + real instruction -----
        if not args.mapping_json or not args.hf_parquet:
            raise ValueError("--mapping_json and --hf_parquet are required "
                             "when using --hf_row")
        with open(args.mapping_json, "r") as f:
            mapping = json.load(f)
        # Find the mapping entry for this HF row
        entry = None
        for m_entry in mapping["matches"]:
            if m_entry["hf_row"] == args.hf_row:
                entry = m_entry
                break
        if entry is None:
            raise ValueError(f"HF row {args.hf_row} not found in mapping "
                             f"(only {len(mapping['matches'])} matched rows)")

        subtask = entry["task"]
        start_frame = int(entry["start_frame"])
        lang_annotation = entry["calvin_ann"]

        # Read HF parquet for goal image and instruction
        import pyarrow.parquet as pq
        t = pq.read_table(args.hf_parquet)
        hf_instruction = t["edit_instruction"][args.hf_row].as_py()
        goal_rgb = _decode_hf_image(t["edited_image"][args.hf_row].as_py())
        original_rgb = _decode_hf_image(t["original_image"][args.hf_row].as_py())

        # Reset env from the correct start frame
        npz_path = val_dir / f"episode_{start_frame:07d}.npz"
        with np.load(npz_path, allow_pickle=True) as z:
            robot_obs = np.asarray(z["robot_obs"])
            scene_obs = np.asarray(z["scene_obs"])
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        print(colored(
            f"[visualize] HF-row mode\n"
            f"  hf_row        = {args.hf_row}\n"
            f"  hf_id         = {entry['hf_id']}\n"
            f"  task          = {subtask}\n"
            f"  instruction   = {hf_instruction}\n"
            f"  calvin_ann    = {lang_annotation}\n"
            f"  start_frame   = {start_frame}\n"
            f"  end_frame     = {entry['end_frame']}",
            "cyan",
        ))
        # Use the HF instruction (what the user cares about)
        lang_annotation = hf_instruction
    else:
        # ----- Legacy sequence mode -----
        if args.sequence_idx < 0:
            raise ValueError("Must specify either --hf_row or --sequence_idx")
        if not args.goal_image:
            raise ValueError("--goal_image required in legacy sequence mode")

        eval_sequences = get_sequences(args.num_sequences)
        if args.sequence_idx >= len(eval_sequences):
            raise IndexError(
                f"--sequence_idx {args.sequence_idx} out of range "
                f"[0, {len(eval_sequences)})"
            )
        initial_state, eval_sequence = eval_sequences[args.sequence_idx]
        if args.subtask_idx >= len(eval_sequence):
            raise IndexError(
                f"--subtask_idx {args.subtask_idx} out of range for sequence "
                f"{args.sequence_idx} (len={len(eval_sequence)})"
            )
        subtask = eval_sequence[args.subtask_idx]

        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        val_annotations = OmegaConf.load(
            conf_dir / "annotations/new_playtable_validation.yaml"
        )
        lang_annotation = val_annotations[subtask][0]
        goal_rgb = _load_image_as_static_rgb(args.goal_image)
        original_rgb = None

        print(colored(
            f"[visualize] sequence={args.sequence_idx}  "
            f"subtask_idx={args.subtask_idx}  subtask='{subtask}'\n"
            f"[visualize] instruction: {lang_annotation}",
            "cyan",
        ))

    model.reset()
    start_info = env.get_info()

    frames = []
    success = False
    obs = env.get_obs()
    for i in range(args.ep_len):
        static_rgb = obs["rgb_obs"]["rgb_static"]
        hand_rgb = obs["rgb_obs"]["rgb_gripper"]
        frames.append(_compose_frame(static_rgb, hand_rgb, goal_rgb))

        action, _progress = model.step(obs, [deepcopy(goal_rgb)],
                                       [lang_annotation])
        obs, _, _, current_info = env.step(action)

        current_task_info = task_oracle.get_task_info_for_set(
            start_info, current_info, {subtask}
        )
        if len(current_task_info) > 0:
            success = True
            frames.append(_compose_frame(
                obs["rgb_obs"]["rgb_static"], obs["rgb_obs"]["rgb_gripper"],
                goal_rgb))
            if args.stop_on_success:
                break

    print(colored(
        f"[visualize] finished. success={success}  frames={len(frames)}  "
        f"writing {args.output}",
        "green" if success else "yellow",
    ))
    _write_video(frames, args.output, fps=args.fps)


if __name__ == "__main__":
    main()
