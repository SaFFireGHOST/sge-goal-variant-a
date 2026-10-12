# SGE-Goal Follow-up: Variant A (Policy-Aligned Generator)

Self-contained workspace for running and deploying **Variant A** of the SGE-Goal follow-up research agenda in the cloud.

---

## 1. Variant A Overview

In **Variant A**, the downstream multi-task policy $\pi_\phi$ (GR-MG on CALVIN) remains **strictly frozen**, and the goal generator $G_\theta$ (Flux-Kontext LoRA) is trained to produce goal images that maximize policy task execution success.

### Key Algorithmic Components:
1. **Stage A1: Offline Policy Behavioral-Matching ($\mathcal{L}_{\text{beh}}$)**
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{flow}} + \lambda_{\text{beh}} \|\pi_\phi(\cdot \mid I_0, \hat{I}_e) - a^*\|_2^2$$
   Backpropagates policy action gradients through the decoded goal images / latents directly into the generator LoRA weights $\theta_G$.
2. **Stage A2: Advantage Direct Preference Optimization (Flow-DPO)**
   - Collects candidate rollouts using nested factorial sampling: $K_m=3$ masks $\times$ $K_a=2$ appearance completions $\times$ $K_s=2$ environment seeds.
   - Computes isolated appearance advantage $A_3$ and mask advantage $A_2$.
   - Optimizes generator LoRAs with continuous Flow-DPO without requiring online policy updates.

---

## 2. Directory Structure

```text
sge_followup_variant_a/
├── README.md                           # This execution and setup guide
├── checkpoints/                        # Base pre-trained LoRA weights (CALVIN)
│   ├── calvin-fer-mask-prediction/     # Step 2: Epoch 22 LoRA safetensors (104 MB)
│   └── calvin-image-editing/           # Step 3: Epoch 18 LoRA safetensors (104 MB)
├── generator/                          # SGE Generator training & inference scripts
│   ├── flux_kontext_lora_finetuning.py # Local multi-GPU training
│   ├── pipeline_flux_kontext_multi_imgs.py # Custom multi-image Flux pipeline
│   ├── flux_kontext_eval_utils.py      # Goal image generation & visualizer utils
│   ├── modal_flux_kontext_train.py     # Modal cloud training runner (B200/H100)
│   ├── flux_2_klein_lora_finetuning.py # Flux-2 Klein lightweight variant
│   └── requirements.txt                # Generator dependencies
├── policy/                             # Downstream Policy (GR-MG) & CALVIN Benchmark
│   ├── gr_mg_policy/                   # Policy model architecture & GPT-2 backbone
│   ├── gr_mg_evaluate/                 # CALVIN challenge evaluation & rollout loops
│   ├── modal_calvin_eval.py            # Modal cloud runner for CALVIN LH-MTLC rollouts
│   └── USAGE.md                        # Policy evaluation guide & volume mapping
├── evaluation/                         # SGE Diagnostic Evaluators
│   ├── R14-15.py                       # Source & Destination adherence (DINOv2/LoFTR)
│   ├── R1.py                           # Stationary object preservation
│   └── sge_goal_eval_aggregator.py     # Aggregates diagnostic metrics
└── variant_a_alignment/                # Variant A Alignment Pipeline
    ├── train_a_dense.py                # Stage A1: Behavioral matching training
    ├── modal_train_a_dense.py          # Stage A1: Modal cloud launcher
    ├── factorial_rollouts.py           # Stage A2: Km x Ka x Ks advantage collector
    ├── train_a_flow_dpo.py             # Stage A2: Continuous Flow-DPO training
    └── modal_train_a_flow_dpo.py       # Stage A2: Modal cloud launcher
```

---

## 3. Cloud Execution (Modal)

The codebase is configured to run serverlessly on [Modal](https://modal.com) across persistent storage volumes:
- `rm-policy-calvin-data`: CALVIN benchmark environment and dataset.
- `rm-policy-ckpts`: GR-MG policy checkpoints and MAE vision backbones.
- `rm-policy-runs`: Training checkpoints, rollout results, and evaluation logs.

### Step 3.1: Run Stage A1 Dense Behavioral Matching
```bash
cd variant_a_alignment
modal run modal_train_a_dense.py
```
This launches distributed training on a cloud GPU (B200/H100) updating the Step 3 LoRA with combined Flow Matching + Behavioral action matching loss.

### Step 3.2: Collect Factorial Rollout Preferences
```bash
modal run modal_train_a_flow_dpo.py --action rollouts
```
Runs $K_m \times K_a \times K_s$ rollouts inside the remote CALVIN environment and generates `step3_appearance_preferences.json` and `step2_mask_preferences.json`.

### Step 3.3: Train Continuous Flow-DPO
```bash
# Optimize Step 3 (Appearance Completion LoRA)
modal run modal_train_a_flow_dpo.py --action train --stage step3

# Optimize Step 2 (Mask Generation LoRA)
modal run modal_train_a_flow_dpo.py --action train --stage step2
```

### Step 3.4: Evaluate Downstream Policy Success on CALVIN
```bash
cd ../policy
modal run modal_calvin_eval.py::eval_calvin --manifest-path /runs/variant_a_goals.json
```

---

## 4. Compute & Resource Profile

| Component | Target GPU | VRAM | Batch Size | Expected Time |
|---|---|---|---|---|
| **Stage A1 ($\mathcal{L}_{\text{beh}}$)** | 1x B200 or 1x H100 | $\ge 40$ GB | 8 (grad accum 2) | ~3.5 hours (15 epochs) |
| **Stage A2 (Factorial Rollouts)** | 1x A100-80GB / H100 | $\ge 24$ GB | Sequential rollouts | ~5–6 hours (200 tasks) |
| **Stage A2 (Flow-DPO)** | 1x A100 / H100 | $\ge 40$ GB | 8 | ~1.5 hours (5 epochs) |
| **Final Benchmark Eval** | 1x A100 | 24 GB | 1 | ~1.5 hours (1000 sequences) |

---

## 5. Datasets & Mirrors

- **CALVIN Goal Dataset:** `SuyashAjitChavan2002/Calvin-Dataset-Final` (Mirror: `RiddhiCh/Calvin-Dataset-Final`)
- **Base Diffusion Model:** `black-forest-labs/FLUX.1-Kontext-dev`
