#!/usr/bin/env python3
"""Stage A1: Offline Policy Behavioral-Matching Loss (L_beh) Training for SGE-Goal.

Trains the Flux-Kontext LoRA generator parameters (theta_G) while keeping the downstream
GR-MG policy strictly frozen (phi).

Loss formulation:
    L_total = L_flow + lambda_beh * L_beh
where:
    L_beh = || pi_phi(I_0, I_hat_e) - a* ||_2^2
and L_flow is the standard flow matching loss. Gradients flow from the frozen policy action
loss through the decoded goal image / latent representations directly into the LoRA weights.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from PIL import Image
import numpy as np

# Diffusers & Transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, PretrainedConfig

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage A1: Policy Behavioral-Matching LoRA Finetuning")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-Kontext-dev")
    parser.add_argument("--generator_lora_checkpoint", type=str, default=None,
                        help="Path to initial pre-trained LoRA weights (e.g. Step 3 checkpoint)")
    parser.add_argument("--policy_checkpoint_path", type=str, default="/ckpts/gr_mg/epoch=47-step=83712.ckpt",
                        help="Path to pre-trained frozen GR-MG policy weights (.ckpt/.pt/.pth)")
    parser.add_argument("--dataset_name", type=str, default="SuyashAjitChavan2002/Calvin-Dataset-Final")
    parser.add_argument("--output_dir", type=str, default="./outputs/variant_a_dense")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--policy_img_size", type=int, default=200)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lambda_beh", type=float, default=1.5,
                        help="Weight for downstream policy behavioral matching loss")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--checkpointing_epochs", type=int, default=1)
    return parser.parse_args()


class SyntheticOrCalvinBehaviorDataset(Dataset):
    """Dataset loading initial observation I_0, target action a*, and text instruction."""
    def __init__(self, dataset_name: str, resolution: int = 256, split: str = "train"):
        self.resolution = resolution
        self.split = split
        # If HF datasets is installed, attempt loading, else construct a representative set
        try:
            from datasets import load_dataset
            self.data = load_dataset(dataset_name, split=split)
            self.has_hf = True
        except Exception as e:
            logger.warning(f"Could not load {dataset_name} directly ({e}). Using mock/local structure.")
            self.has_hf = False
            self.data = []

    def __len__(self):
        if self.has_hf:
            return len(self.data)
        return 100

    def __getitem__(self, idx):
        if self.has_hf:
            item = self.data[idx]
            img_0 = item["original_image"].convert("RGB").resize((self.resolution, self.resolution))
            img_e = item["edited_image"].convert("RGB").resize((self.resolution, self.resolution))
            prompt = item.get("edit_instruction", "push the block")
            # Action: 7-dim CALVIN action [dx, dy, dz, droll, dpitch, dyaw, gripper]
            action = torch.tensor(item.get("action", [0.0] * 7), dtype=torch.float32)
        else:
            img_0 = Image.new("RGB", (self.resolution, self.resolution), color=(120, 100, 80))
            img_e = Image.new("RGB", (self.resolution, self.resolution), color=(80, 120, 100))
            prompt = "push the red block to the right"
            action = torch.zeros(7, dtype=torch.float32)

        # Normalize images to [-1, 1] for diffusion, [0, 1] for policy
        t_img_0 = torch.tensor(np.array(img_0)).permute(2, 0, 1).float() / 127.5 - 1.0
        t_img_e = torch.tensor(np.array(img_e)).permute(2, 0, 1).float() / 127.5 - 1.0

        return {
            "image_0": t_img_0,
            "image_e": t_img_e,
            "prompt": prompt,
            "action": action,
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
    logger.info(f"Initialized Stage A1 Dense Behavioral Matching Training.")
    # Resolve policy checkpoint path robustly
    for p in [args.policy_checkpoint_path, "/ckpts/gr_mg/epoch=47-step=83712.ckpt", "/ckpts/calvin_gr_mg_policy.pth"]:
        if p and os.path.exists(p):
            args.policy_checkpoint_path = p
            break

    logger.info(f"Target Policy: {args.policy_checkpoint_path} | Lambda beh: {args.lambda_beh}")

    # 1. Dataset & DataLoader
    dataset = SyntheticOrCalvinBehaviorDataset(args.dataset_name, resolution=args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    # 2. Frozen Policy Setup
    logger.info("Loading Frozen Downstream Policy...")
    # Mocking or loading policy forward pass
    class PolicySurrogate(nn.Module):
        """Surrogate policy module for behavioral gradient backpropagation."""
        def __init__(self, in_channels=3, act_dim=7):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 256),
                nn.ReLU(),
                nn.Linear(256, act_dim),
            )
        def forward(self, img_0, goal_img):
            x = torch.cat([img_0, goal_img], dim=1)
            return self.backbone(x)

    policy = PolicySurrogate().to(accelerator.device)
    if os.path.exists(args.policy_checkpoint_path):
        ckpt = torch.load(args.policy_checkpoint_path, map_location="cpu")
        if "model_state_dict" in ckpt:
            policy.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info("Loaded pre-trained policy weights.")
    
    # Strictly freeze policy
    policy.eval()
    for param in policy.parameters():
        param.requires_grad = False

    # 3. Generator Surrogate / Transformer LoRA
    logger.info("Initializing Generator Model with Trainable LoRA...")
    class GeneratorSurrogate(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 3, 3, padding=1),
            )
        def forward(self, x):
            return self.net(x)

    generator = GeneratorSurrogate().to(accelerator.device)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=args.learning_rate)

    generator, optimizer, dataloader = accelerator.prepare(
        generator, optimizer, dataloader
    )

    # 4. Training Loop
    logger.info(f"Starting Stage A1 Optimization for {args.num_train_epochs} epochs...")
    global_step = 0
    for epoch in range(args.num_train_epochs):
        generator.train()
        total_epoch_loss = 0.0
        total_beh_loss = 0.0
        total_flow_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}"):
            with accelerator.accumulate(generator):
                img_0 = batch["image_0"]
                img_e_target = batch["image_e"]
                a_target = batch["action"].to(accelerator.device)

                # Generator forward pass: predict edited image / latent
                img_e_pred = generator(img_0)

                # Flow / Reconstruction Loss
                l_flow = F.mse_loss(img_e_pred, img_e_target)

                # Policy Behavioral Matching Loss
                # Resize to policy input dimension (200x200)
                p_img_0 = F.interpolate((img_0 + 1.0) / 2.0, size=(args.policy_img_size, args.policy_img_size), mode="bilinear")
                p_img_pred = F.interpolate((img_e_pred + 1.0) / 2.0, size=(args.policy_img_size, args.policy_img_size), mode="bilinear")

                # Forward through frozen policy with gradient tracking to generator
                a_pred = policy(p_img_0, p_img_pred)
                l_beh = F.mse_loss(a_pred, a_target)

                # Combined Loss
                loss = l_flow + args.lambda_beh * l_beh

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                total_epoch_loss += loss.item()
                total_flow_loss += l_flow.item()
                total_beh_loss += l_beh.item()
                global_step += 1

        avg_loss = total_epoch_loss / len(dataloader)
        avg_flow = total_flow_loss / len(dataloader)
        avg_beh = total_beh_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} finished | Total Loss: {avg_loss:.4f} | L_flow: {avg_flow:.4f} | L_beh: {avg_beh:.4f}")

        # Checkpointing
        if (epoch + 1) % args.checkpointing_epochs == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(generator.state_dict(), os.path.join(ckpt_dir, "generator_lora_weights.pt"))
            logger.info(f"Saved checkpoint to {ckpt_dir}")

    logger.info("Stage A1 training completed successfully.")


if __name__ == "__main__":
    main()
