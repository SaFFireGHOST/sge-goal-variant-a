#!/usr/bin/env python3
"""Stage A2: Continuous Flow-DPO (Direct Preference Optimization) for SGE-Goal LoRAs.

Aligns Step 2 (Mask Generation) and Step 3 (Appearance Completion) LoRAs using
advantage-weighted preference pairs collected from closed-loop CALVIN rollouts.

Formulation (Flow-DPO):
  L_Flow-DPO = -E [ log sigma( beta * ( ||v_ref(x_t^w) - v_t^w||^2 - ||v_theta(x_t^w) - v_t^w||^2
                                      - (||v_ref(x_t^l) - v_t^l||^2 - ||v_theta(x_t^l) - v_t^l||^2) ) ) ]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage A2: Continuous Flow-DPO Training")
    parser.add_argument("--preference_data_path", type=str, required=True,
                        help="Path to JSON preference pairs file (from factorial_rollouts.py)")
    parser.add_argument("--target_stage", type=str, default="step3", choices=["step2", "step3"],
                        help="Whether optimizing Step 2 (FER mask) or Step 3 (appearance)")
    parser.add_argument("--base_lora_weights", type=str, default=None,
                        help="Path to pre-trained LoRA weights to fine-tune")
    parser.add_argument("--output_dir", type=str, default="./outputs/flow_dpo")
    parser.add_argument("--beta", type=float, default=2500.0,
                        help="DPO temperature scaling hyperparameter for continuous flow matching")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class FlowDPOPreferenceDataset(Dataset):
    def __init__(self, json_path: str, latent_dim: int = 16, spatial_size: int = 32):
        self.latent_dim = latent_dim
        self.spatial_size = spatial_size
        with open(json_path, "r") as f:
            self.pairs = json.load(f)
        logger.info(f"Loaded {len(self.pairs)} preference pairs from {json_path}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        diff = float(pair.get("advantage_diff", 1.0))
        # Winning and losing latents / trajectories (synthetic representations for standalone training)
        x_w = torch.randn(self.latent_dim, self.spatial_size, self.spatial_size)
        x_l = torch.randn(self.latent_dim, self.spatial_size, self.spatial_size)
        x_0 = torch.randn(self.latent_dim, self.spatial_size, self.spatial_size)

        return {
            "x_w": x_w,
            "x_l": x_l,
            "x_0": x_0,
            "weight": torch.tensor(diff, dtype=torch.float32),
        }


def main():
    if "--allow_mock" not in sys.argv:
        raise SystemExit(
            "MOCK IMPLEMENTATION: surrogate models / simulated rewards, results are "
            "meaningless. See plan.md (Gate 0 first, real implementation in Phase 2A). "
            "Pass --allow_mock to run anyway."
        )
    sys.argv = [a for a in sys.argv if a != "--allow_mock"]
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_dir=args.output_dir,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Initialized Flow-DPO Alignment for {args.target_stage.upper()}.")
    logger.info(f"Beta: {args.beta} | LR: {args.learning_rate}")

    # 1. Dataset
    dataset = FlowDPOPreferenceDataset(args.preference_data_path)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    # 2. Models: Trainable Model and Frozen Reference Model
    class VelocityPredictor(nn.Module):
        def __init__(self, in_channels=32, out_channels=16):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(64, out_channels, 3, padding=1),
            )
        def forward(self, x_t, t_emb):
            return self.conv(x_t)

    model = VelocityPredictor().to(accelerator.device)
    ref_model = VelocityPredictor().to(accelerator.device)

    # Copy weights to reference model and freeze it
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # 3. Optimization Loop
    logger.info(f"Starting Flow-DPO training for {args.num_train_epochs} epochs...")
    for epoch in range(args.num_train_epochs):
        model.train()
        total_epoch_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Flow-DPO Epoch {epoch+1}"):
            with accelerator.accumulate(model):
                x_w = batch["x_w"]
                x_l = batch["x_l"]
                x_0 = batch["x_0"]
                weight = batch["weight"]

                # Continuous flow time step t ~ U(0, 1)
                t = torch.rand(x_w.shape[0], 1, 1, 1, device=x_w.device)

                # Linear flow matching trajectory: x_t = (1 - t) * x_0 + t * x_1
                # Target velocity: v_t = x_1 - x_0
                x_t_w = (1.0 - t) * x_0 + t * x_w
                v_t_w = x_w - x_0

                x_t_l = (1.0 - t) * x_0 + t * x_l
                v_t_l = x_l - x_0

                # Input to velocity predictor: concat x_t with conditioning x_0
                inp_w = torch.cat([x_t_w, x_0], dim=1)
                inp_l = torch.cat([x_t_l, x_0], dim=1)

                v_pred_w = model(inp_w, t)
                v_pred_l = model(inp_l, t)

                with torch.no_grad():
                    v_ref_w = ref_model(inp_w, t)
                    v_ref_l = ref_model(inp_l, t)

                # Flow MSE errors
                mse_w = torch.mean((v_pred_w - v_t_w) ** 2, dim=[1, 2, 3])
                mse_l = torch.mean((v_pred_l - v_t_l) ** 2, dim=[1, 2, 3])

                ref_mse_w = torch.mean((v_ref_w - v_t_w) ** 2, dim=[1, 2, 3])
                ref_mse_l = torch.mean((v_ref_l - v_t_l) ** 2, dim=[1, 2, 3])

                # Flow-DPO implicit reward difference
                # Note: lower MSE error -> higher implicit reward
                # diff = (mse_l - mse_w) - (ref_mse_l - ref_mse_w)
                implicit_reward_diff = (ref_mse_w - mse_w) - (ref_mse_l - mse_l)

                # DPO loss
                dpo_logits = args.beta * implicit_reward_diff
                loss = -F.logsigmoid(dpo_logits).mean()

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                total_epoch_loss += loss.item()

        avg_loss = total_epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} finished | Flow-DPO Loss: {avg_loss:.5f}")

        # Save checkpoint
        save_path = os.path.join(args.output_dir, f"{args.target_stage}_flow_dpo_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), save_path)
        logger.info(f"Saved Flow-DPO weights to {save_path}")

    logger.info("Flow-DPO training finished successfully.")


if __name__ == "__main__":
    main()
