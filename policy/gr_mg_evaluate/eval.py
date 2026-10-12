# MIT License
#
# Copyright (c) 2021 Oier Mees
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# Additions (c) 2026 — local extensions for user-supplied goal-image
#                      evaluation on the CALVIN-D test set.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
import argparse
import json
import logging
import os
import re
import sys
from copy import deepcopy
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
)

from goal_image_provider import (
    IP2PGoalImageProvider,
    ManifestGoalImageProvider,
)
from utils.utils import print_and_save
from wrapper.model_wrapper import CustomModel

logger = logging.getLogger(__name__)
EP_LEN = 360
NUM_SEQUENCES = 1000
SAVE_DIR = None
FAIL_COUNTER = 0


def make_env(dataset_path, observation_space, device_id):
    val_folder = Path(dataset_path) / "validation"
    from wrapper.calvin_env_wrapper_raw import CalvinEnvWrapperRaw

    device = torch.device("cuda", device_id)
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    return env


def evaluate_policy(model, env, eval_sr_path, eval_result_path, goal_provider,
                    num_sequences):
    """Run this function to evaluate a model on the CALVIN challenge."""
    conf_dir = Path("./calvin/calvin_models/conf")
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    eval_sequences = get_sequences(num_sequences)
    results = []
    sequence_i = 0
    for index, (initial_state, eval_sequence) in enumerate(eval_sequences):
        result = evaluate_sequence(
            env, model, task_oracle, initial_state, eval_sequence,
            val_annotations, sequence_i, goal_provider,
        )
        results.append(result)
        success_list = count_success(results)
        with open(eval_sr_path, "a") as f:
            line = f"{sequence_i}/{num_sequences}: "
            for sr in success_list:
                line += f"{sr:.3f} | "
            sequence_i += 1
            line += "\n"
            f.write(line)

        if index % 100 == 0 and index != 0:
            print_and_save(results, eval_sequences[: index + 1],
                           eval_result_path[:-5] + f"_{index+1}" + ".json", None)
    print_and_save(results, eval_sequences, eval_result_path, None)
    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence,
                      val_annotations, sequence_i, goal_provider):
    """Evaluates a sequence of language instructions."""
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout(env, model, task_checker, subtask, val_annotations,
                          subtask_i, sequence_i, goal_provider)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, subtask_i,
            sequence_i, goal_provider):
    """Run the actual rollout on one subtask."""
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    goal_provider.reset()
    start_info = env.get_info()
    debug_image = []
    progress = 0
    goal_image = None
    for i in range(EP_LEN):
        # Refresh the goal image. With the user-supplied / static providers
        # this returns the same final goal image every step (no sub-goals).
        # With the IP2P provider this re-plans every K steps internally.
        goal_image = goal_provider.get_goal(
            sequence_idx=sequence_i,
            subtask_idx=subtask_i,
            subtask_name=subtask,
            step_idx=i,
            lang_annotation=lang_annotation,
            progress=progress,
            obs=obs,
        )

        if i % 20 == 0:
            static_rgb = obs["rgb_obs"]["rgb_static"]
            hand_rgb = obs["rgb_obs"]["rgb_gripper"]
            debug_image.append([static_rgb, goal_image, hand_rgb])

        # The policy expects a list-like goal container (it indexes [0]).
        action, progress = model.step(obs, [deepcopy(goal_image)],
                                      [lang_annotation])
        obs, _, _, current_info = env.step(action)

        current_task_info = task_oracle.get_task_info_for_set(
            start_info, current_info, {subtask}
        )
        if len(current_task_info) > 0:
            print("success!")
            return True
    print("fail!")

    global FAIL_COUNTER
    FAIL_COUNTER += 1
    if FAIL_COUNTER % 30 == 0 and SAVE_DIR is not None:
        length = len(debug_image)
        fig, ax = plt.subplots(length, 2, figsize=(5.5, 46.58))
        for ax_ in ax.flat:
            ax_.axis("off")
        for j in range(length):
            ax[j][0].imshow(debug_image[j][0])
            ax[j][1].imshow(debug_image[j][1])
        plt.tight_layout()
        plt.axis("off")
        plt.savefig(os.path.join(SAVE_DIR, f"{sequence_i}-{subtask_i}-{subtask}.png"),
                    dpi=100)
        plt.close()
    return False


def build_goal_provider(args):
    if args.goal_image_source == "ip2p":
        if not args.ip2p_ckpt_path:
            raise ValueError("--goal_image_source ip2p requires --ip2p_ckpt_path")
        from goal_gen.evaluate import IP2PEvaluation  # heavy; lazy import

        ip2p_model = IP2PEvaluation(args.ip2p_ckpt_path)
        return IP2PGoalImageProvider(ip2p_model)
    if args.goal_image_source == "manifest":
        if not args.goal_images_manifest:
            raise ValueError(
                "--goal_image_source manifest requires --goal_images_manifest"
            )
        return ManifestGoalImageProvider(args.goal_images_manifest)
    raise ValueError(f"Unknown --goal_image_source: {args.goal_image_source}")


def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(
        description="Evaluate GR-MG on CALVIN with either IP2P-generated or "
                    "user-provided final goal images."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the CALVIN dataset root, e.g. /.../task_D_D for the "
             "CALVIN-D test set, or /.../task_ABC_D for the ABC->D split.",
    )
    parser.add_argument("--config_path", type=str, default="",
                        help="Path to the policy config JSON.")
    parser.add_argument("--ckpt_dir", type=str, default="",
                        help="Directory containing the policy ckpt files.")
    parser.add_argument("--epoch", type=int, default=47,
                        help="Epoch index of the policy ckpt to load.")
    parser.add_argument("--device_id", default=0, type=int, help="CUDA device id.")

    # ---- goal-image source -------------------------------------------------
    parser.add_argument(
        "--goal_image_source",
        type=str,
        default="ip2p",
        choices=["ip2p", "manifest"],
        help="Where each subtask's goal image comes from. 'ip2p' = the original "
             "GR-MG behaviour (re-planned every 20 steps by InstructPix2Pix). "
             "'manifest' = user-supplied final goal image per test sample, held "
             "fixed for the whole subtask rollout (no sub-goal re-planning).",
    )
    parser.add_argument("--ip2p_ckpt_path", default="", type=str,
                        help="IP2P checkpoint path (only for --goal_image_source ip2p).")
    parser.add_argument(
        "--goal_images_manifest",
        type=str,
        default="",
        help="Path to a JSON manifest of user-supplied goal images "
             "(only for --goal_image_source manifest). See "
             "evaluate/goal_image_provider.py for supported formats.",
    )

    # ---- evaluation scope --------------------------------------------------
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=NUM_SEQUENCES,
        help="Number of long-horizon test sequences to evaluate "
             "(default 1000, the official CALVIN LH-MTLC count).",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional tag appended to the output directory name so different "
             "goal-image sources don't overwrite each other's results.",
    )

    args = parser.parse_args()
    assert args.config_path, "--config_path is required"

    with open(args.config_path, "r") as f:
        configs = json.load(f)

    # Locate ckpt
    ckpt_path = None
    for ckpt_file in os.listdir(args.ckpt_dir):
        m = re.search(r"epoch=(\d+)", ckpt_file)
        if m and int(m.group(1)) == args.epoch:
            ckpt_path = os.path.join(args.ckpt_dir, ckpt_file)
            break
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint matching epoch={args.epoch} found in {args.ckpt_dir}"
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
    env = make_env(args.dataset_path, observation_space, args.device_id)

    # Output directory: encode dataset split + goal-image source + epoch + tag.
    split_tag = Path(args.dataset_path).name  # e.g. "task_D_D"
    sub_dir = f"{split_tag}_{args.goal_image_source}_{args.epoch}_epoch"
    if args.run_tag:
        sub_dir += f"_{args.run_tag}"
    global SAVE_DIR
    SAVE_DIR = os.path.join(args.ckpt_dir, sub_dir)
    os.makedirs(SAVE_DIR, exist_ok=True)
    sr_path = os.path.join(SAVE_DIR, "success_rate.txt")
    result_path = os.path.join(SAVE_DIR, "results.json")

    goal_provider = build_goal_provider(args)
    print(colored(
        f"[eval] split={split_tag}  goal_image_source={args.goal_image_source}  "
        f"num_sequences={args.num_sequences}  out={SAVE_DIR}",
        "cyan",
    ))

    evaluate_policy(
        model,
        env,
        eval_sr_path=sr_path,
        eval_result_path=result_path,
        goal_provider=goal_provider,
        num_sequences=args.num_sequences,
    )


if __name__ == "__main__":
    main()
