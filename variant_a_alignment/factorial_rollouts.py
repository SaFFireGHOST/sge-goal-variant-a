#!/usr/bin/env python3
"""Stage A2 Factorial Rollout & Advantage Computation for SGE-Goal Variant A.

Implements the nested factorial sampling scheme from Section 3.2 of the Research Proposal:
For each evaluation task / scene initial condition:
  1. Sample K_m = 3 candidate masks from Step 2 LoRA: {m_1, m_2, m_3}
  2. For each mask m_j, sample K_a = 2 appearance completions from Step 3 LoRA: {a_{j,1}, a_{j,2}}
  3. Execute policy rollouts in CALVIN across K_s = 2 environment seeds.
  4. Record binary / sparse reward R(m, a, s) in {0, 1}.
  5. Compute advantage estimates:
     - Appearance Advantage: A_3(m, a) = R_bar(m, a) - R_bar(m)
     - Mask Advantage:       A_2(m)    = R_bar(m)    - (1/K_m) * sum_{m'} R_bar(m')
  6. Export preference pairs (winner, loser) for Flow-DPO training.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FactorialRollouts")


def parse_args():
    parser = argparse.ArgumentParser(description="Factorial Rollout Collection for Flow-DPO")
    parser.add_argument("--km", type=int, default=3, help="Number of mask samples per prompt")
    parser.add_argument("--ka", type=int, default=2, help="Number of appearance samples per mask")
    parser.add_argument("--ks", type=int, default=2, help="Number of environment seeds per goal image")
    parser.add_argument("--num_tasks", type=int, default=200, help="Number of CALVIN validation tasks to sample")
    parser.add_argument("--output_dir", type=str, default="./outputs/factorial_rollouts")
    parser.add_argument("--step2_lora_path", type=str, default="/ckpts/ckpts/calvin-fer-mask-prediction.safetensors")
    parser.add_argument("--step3_lora_path", type=str, default="/ckpts/ckpts/calvin-image-editing.safetensors")
    parser.add_argument("--policy_ckpt_path", type=str, default="/ckpts/gr_mg/epoch=47-step=83712.ckpt")
    return parser.parse_args()


def simulate_rollout_reward(mask_idx: int, appearance_idx: int, env_seed: int, task_name: str) -> float:
    """Simulates or calls the CALVIN environment rollout.
    In real deployment, this invokes wrapper.calvin_env_wrapper_raw and model.step.
    Returns 1.0 if subtask succeeds, 0.0 otherwise.
    """
    # Deterministic simulation hash for testing/offline dry-run
    h = hash((task_name, mask_idx, appearance_idx, env_seed))
    prob = 0.3 + 0.3 * (appearance_idx % 2) + 0.2 * (mask_idx == 0)
    return 1.0 if (abs(h) % 100) < (prob * 100) else 0.0


def main():
    if "--allow_mock" not in sys.argv:
        raise SystemExit(
            "MOCK IMPLEMENTATION: surrogate models / simulated rewards, results are "
            "meaningless. See plan.md (Gate 0 first, real implementation in Phase 2A). "
            "Pass --allow_mock to run anyway."
        )
    sys.argv = [a for a in sys.argv if a != "--allow_mock"]
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Robust multi-environment path resolution
    repo_root = Path(__file__).resolve().parent.parent
    step2_candidates = [
        args.step2_lora_path,
        "/ckpts/ckpts/calvin-fer-mask-prediction.safetensors",
        "/ckpts/calvin-fer-mask-prediction.safetensors",
        str(repo_root / "checkpoints/calvin-fer-mask-prediction/pytorch_lora_weights.safetensors"),
        "../checkpoints/calvin-fer-mask-prediction/pytorch_lora_weights.safetensors",
    ]
    for p in step2_candidates:
        if p and os.path.exists(p):
            args.step2_lora_path = p
            break

    step3_candidates = [
        args.step3_lora_path,
        "/ckpts/ckpts/calvin-image-editing.safetensors",
        "/ckpts/calvin-image-editing.safetensors",
        str(repo_root / "checkpoints/calvin-image-editing/pytorch_lora_weights.safetensors"),
        "../checkpoints/calvin-image-editing/pytorch_lora_weights.safetensors",
    ]
    for p in step3_candidates:
        if p and os.path.exists(p):
            args.step3_lora_path = p
            break

    policy_candidates = [
        args.policy_ckpt_path,
        "/ckpts/gr_mg/epoch=47-step=83712.ckpt",
        "/ckpts/calvin_gr_mg_policy.pth",
    ]
    for p in policy_candidates:
        if p and os.path.exists(p):
            args.policy_ckpt_path = p
            break
    logger.info(f"Starting Factorial Rollout Sampling: K_m={args.km}, K_a={args.ka}, K_s={args.ks}")
    logger.info(f"Step 2 LoRA: {args.step2_lora_path}")
    logger.info(f"Step 3 LoRA: {args.step3_lora_path}")

    tasks = [f"calvin_task_{i:04d}" for i in range(args.num_tasks)]
    
    appearance_preference_pairs = []
    mask_preference_pairs = []

    total_rollouts = 0
    total_successes = 0

    for task in tasks:
        # 1. Mask evaluations: R_mask[m_idx] = list of Ka average rewards
        mask_rewards = {}
        appearance_rewards = {}

        for m_idx in range(args.km):
            appearance_rewards[m_idx] = {}
            for a_idx in range(args.ka):
                seed_rewards = []
                for s_idx in range(args.ks):
                    r = simulate_rollout_reward(m_idx, a_idx, s_idx, task)
                    seed_rewards.append(r)
                    total_rollouts += 1
                    if r > 0.5:
                        total_successes += 1

                # R_bar(m, a) = mean over seeds
                appearance_rewards[m_idx][a_idx] = float(np.mean(seed_rewards))

            # R_bar(m) = mean over appearance samples
            mask_rewards[m_idx] = float(np.mean(list(appearance_rewards[m_idx].values())))

        # 2. Compute Appearance Advantage A_3(m, a) = R_bar(m, a) - R_bar(m)
        # Form preference pairs if there is a gap
        for m_idx in range(args.km):
            for a1 in range(args.ka):
                for a2 in range(a1 + 1, args.ka):
                    r1 = appearance_rewards[m_idx][a1]
                    r2 = appearance_rewards[m_idx][a2]
                    if abs(r1 - r2) > 1e-4:
                        winner = a1 if r1 > r2 else a2
                        loser = a2 if r1 > r2 else a1
                        appearance_preference_pairs.append({
                            "task": task,
                            "mask_idx": m_idx,
                            "winner_appearance_idx": winner,
                            "loser_appearance_idx": loser,
                            "advantage_diff": abs(r1 - r2),
                        })

        # 3. Compute Mask Advantage A_2(m) = R_bar(m) - (1/K_m) * sum R_bar(m')
        # Form mask preference pairs
        for m1 in range(args.km):
            for m2 in range(m1 + 1, args.km):
                rm1 = mask_rewards[m1]
                rm2 = mask_rewards[m2]
                if abs(rm1 - rm2) > 1e-4:
                    winner_m = m1 if rm1 > rm2 else m2
                    loser_m = m2 if rm1 > rm2 else m1
                    mask_preference_pairs.append({
                        "task": task,
                        "winner_mask_idx": winner_m,
                        "loser_mask_idx": loser_m,
                        "advantage_diff": abs(rm1 - rm2),
                    })

    sr = (total_successes / max(1, total_rollouts)) * 100.0
    logger.info(f"Completed {total_rollouts} rollouts across {args.num_tasks} tasks.")
    logger.info(f"Aggregate Rollout Success Rate: {sr:.2f}%")
    logger.info(f"Collected {len(appearance_preference_pairs)} appearance preference pairs (Step 3).")
    logger.info(f"Collected {len(mask_preference_pairs)} mask preference pairs (Step 2).")

    # Save dataset manifests
    app_out = os.path.join(args.output_dir, "step3_appearance_preferences.json")
    mask_out = os.path.join(args.output_dir, "step2_mask_preferences.json")

    with open(app_out, "w") as f:
        json.dump(appearance_preference_pairs, f, indent=2)

    with open(mask_out, "w") as f:
        json.dump(mask_preference_pairs, f, indent=2)

    logger.info(f"Saved Flow-DPO preference datasets to:\n  - {app_out}\n  - {mask_out}")


if __name__ == "__main__":
    main()
