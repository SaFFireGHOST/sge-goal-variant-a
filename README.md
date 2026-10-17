# SGE-Goal Follow-up: Variant A (Policy-Aligned Generator)

Self-contained workspace for running, training, and deploying **Variant A** of the SGE-Goal follow-up research agenda in the cloud (Modal, RunPod, Lambda Labs, or local multi-GPU clusters).

---

## 1. Variant A Overview

In **Variant A**, the downstream multi-task policy $\pi_\phi$ (GR-MG on CALVIN) remains **strictly frozen**, while the goal generator $G_\theta$ (Flux-Kontext LoRA) is trained to produce goal images that maximize downstream policy task execution success.

### Key Algorithmic Components:
1. **Stage A1: Offline Policy Behavioral-Matching ($\mathcal{L}_{\text{beh}}$)**
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{flow}} + \lambda_{\text{beh}} \|\pi_\phi(\cdot \mid I_0, \hat{I}_e) - a^*\|_2^2$$
   Backpropagates policy action gradients through the decoded goal images / latents directly into the generator LoRA weights $\theta_G$.
2. **Stage A2: Advantage Direct Preference Optimization (Flow-DPO)**
   - Collects candidate rollouts using nested factorial sampling: $K_m=3$ masks $\times$ $K_a=2$ appearance completions $\times$ $K_s=2$ environment seeds.
   - Isolates appearance advantage $A_3$ and mask advantage $A_2$.
   - Optimizes generator LoRAs with continuous Flow-DPO without requiring online policy updates.

---

## 2. Decoupled Cloud Architecture

To avoid Git LFS bandwidth quotas and huge git bloat, **code and weights are strictly decoupled**:
- **Git Repository:** 100% lightweight code, configs, runners, and documentation (**~400 KB total size**).
- **Model Checkpoints:** Hosted on **Hugging Face Model Hub** and **Modal Cloud Persistent Volumes**.

### Checkpoint Registry:
- **Hugging Face Hub:** [**https://huggingface.co/SaFFire/sge-goal**](https://huggingface.co/SaFFire/sge-goal)
  - `calvin-fer-mask-prediction/pytorch_lora_weights.safetensors` (Step 2 LoRA, 103.4 MB)
  - `calvin-image-editing/pytorch_lora_weights.safetensors` (Step 3 LoRA, 103.4 MB)
- **Modal Volume:** `rm-policy-ckpts` (Mounted at `/ckpts` inside cloud containers)

---

## 3. Environment Setup (`.env`)

A local `.env` file stores configuration secrets (automatically ignored by Git):

```bash
# Hugging Face Hub
HF_TOKEN=hf_...
HF_REPO_ID=SaFFire/sge-goal
HF_DATASET=SuyashAjitChavan2002/Calvin-Dataset-Final
HF_HOME=/data/datasets/.cache

# Modal Cloud Infrastructure
MODAL_PROFILE=mupparapukoushik
MODAL_VOLUME_CKPTS=rm-policy-ckpts
MODAL_VOLUME_DATA=rm-policy-calvin-data
MODAL_VOLUME_RUNS=rm-policy-runs
```

*(See `.env.example` for a clean template).*

---

## 4. Directory Structure

```text
sge_followup_variant_a/
├── README.md                           # This execution and setup guide
├── .env                                # Local secrets and credentials (Git-ignored)
├── .env.example                        # Template for environment variables
├── .gitignore                          # Excludes weights (*.safetensors), runs, and .env
│
├── docs/                               # Research Papers & Theoretical Proposals
│   ├── sge_goal_research_paper.md      # Full Markdown conversion of original SGE-Goal paper
│   ├── SGE-Goal_Follow-up_Research_Proposal_V2.md # Follow-up proposal & Variant A spec
│   └── documentation.md                # Repository codebase documentation
│
├── checkpoints/                        # Model Checkpoint Management
│   ├── README.md                       # Checkpoint documentation
│   ├── sync_checkpoints.sh             # 1-command sync (Hugging Face <-> Modal Volume)
│   ├── upload_to_hf.py                 # Automated Hugging Face uploader
│   ├── calvin-fer-mask-prediction/     # Step 2: Mask LoRA weights & metadata
│   └── calvin-image-editing/           # Step 3: Appearance LoRA weights & metadata
│
├── generator/                          # SGE Generator Training & Inference
│   ├── flux_kontext_lora_finetuning.py # Local multi-GPU training
│   ├── pipeline_flux_kontext_multi_imgs.py # Custom multi-image Flux pipeline
│   ├── flux_kontext_eval_utils.py      # Goal image generation & visualizer utils
│   ├── modal_flux_kontext_train.py     # Modal cloud training runner (B200/H100)
│   ├── flux_2_klein_lora_finetuning.py # Flux-2 Klein lightweight variant
│   └── requirements.txt                # Generator dependencies
│
├── policy/                             # Downstream Policy (GR-MG) & Benchmark
│   ├── gr_mg_policy/                   # Policy model architecture & GPT-2 backbone
│   ├── gr_mg_evaluate/                 # CALVIN challenge evaluation & rollout loops
│   ├── modal_calvin_eval.py            # Modal cloud runner for CALVIN LH-MTLC rollouts
│   └── USAGE.md                        # Policy evaluation guide & volume mapping
│
├── evaluation/                         # SGE Diagnostic Evaluators
│   ├── R14-15.py                       # Source & Destination adherence (DINOv2/LoFTR)
│   ├── R1.py                           # Stationary object preservation
│   └── sge_goal_eval_aggregator.py     # Aggregates diagnostic metrics
│
└── variant_a_alignment/                # Variant A Implementation Pipeline
    ├── train_a_dense.py                # Stage A1: Behavioral matching training
    ├── modal_train_a_dense.py          # Stage A1: Modal cloud runner (Nvidia B200)
    ├── factorial_rollouts.py           # Stage A2: Km x Ka x Ks advantage collector
    ├── train_a_flow_dpo.py             # Stage A2: Continuous Flow-DPO training
    └── modal_train_a_flow_dpo.py       # Stage A2: Modal cloud runner for Flow-DPO
```

---

## 5. Cloud Execution Workflow (Modal)

### Step 5.1: Synchronize Checkpoints
If launching on a new machine or Modal account:
```bash
cd checkpoints

# Download weights from Hugging Face into local folder:
bash sync_checkpoints.sh --pull-hf

# Push weights into your Modal persistent volume:
bash sync_checkpoints.sh --push-modal
```

### Gate 0 — Best-of-N headroom (run this first)
Decides whether Variant A has any signal before any training. Full plan, costs and decision rule: [plan.md](plan.md).
```bash
python variant_a_alignment/gate0_select_rows.py --n 150          # local, free
cd generator
modal run modal_gate0_generate.py::prefetch                       # CPU: cache FLUX weights
modal run modal_gate0_generate.py --limit 5                       # L40S pilot
modal run --detach modal_gate0_generate.py --num-shards 2         # L40S full
cd ../policy
modal run modal_calvin_eval.py::run_gate0 --num-shards 1 --limit 5   # T4 pilot
modal run --detach modal_calvin_eval.py::run_gate0 --num-shards 8    # T4 full
modal volume get rm-policy-runs eval_method/gate0 ../results/
cd .. && python variant_a_alignment/gate0_analyze.py --results results/gate0
```

> **Steps 5.2–5.4 are NOT implemented.** `train_a_dense.py`, `factorial_rollouts.py` and
> `train_a_flow_dpo.py` are mock scripts (surrogate CNNs, `hash()` rewards, random latents)
> and refuse to run without `--allow_mock`. The checkpoint in
> `rm-policy-runs:/variant_a_dense_checkpoints/` is a toy CNN, not a FLUX LoRA.

### Step 5.2: Stage A1 — Offline Dense Behavioral Matching Training (mock)
Run in detached mode so remote execution survives local disconnects:
```bash
cd ../variant_a_alignment

# Detached background execution (Recommended):
modal run --detach modal_train_a_dense.py

# Or interactive foreground execution:
modal run modal_train_a_dense.py
```
- Provisions a serverless **Nvidia B200** GPU on Modal.
- Mounts code into `/workspace` and loads weights from `/ckpts`.
- Saves output LoRAs and logs directly to `/runs/variant_a_dense_checkpoints`.

### Step 5.3: Stage A2 — Advantage Preference Rollouts (mock)
```bash
# Detached background execution:
modal run --detach modal_train_a_flow_dpo.py --action rollouts
```
Runs $K_m \times K_a \times K_s$ rollouts inside the remote CALVIN environment and exports preference datasets to `/runs/variant_a_rollouts/`.

### Step 5.4: Stage A2 — Continuous Flow-DPO Optimization (mock)
```bash
# Fine-tune Step 3 (Appearance Completion LoRA)
modal run --detach modal_train_a_flow_dpo.py --action train --stage step3

# Fine-tune Step 2 (Mask Generation LoRA)
modal run --detach modal_train_a_flow_dpo.py --action train --stage step2
```

### Step 5.5: Benchmark Policy Evaluation on CALVIN
```bash
cd ../policy

# 1. Evaluate on Hugging Face Test Rows (using generated mapping):
modal run --detach modal_calvin_eval.py::eval_hf

# 2. Standard Long-Horizon Evaluation (1,000 Chains with goal manifest):
modal run --detach modal_calvin_eval.py::eval_calvin --manifest-path /runs/variant_a_goals.json
```

### Step 5.6: Managing Detached Cloud Jobs
When running with `--detach`, jobs execute independently on Modal infrastructure. Use these commands to monitor and manage them:

```bash
# List all active/detached cloud apps
modal app list

# Stream live terminal logs from a running app
modal app logs <app-id>

# Stop/terminate a running cloud app
modal app stop <app-id>
```

---

## 6. Compute & Resource Profile

| Component | Target GPU | VRAM | Batch Size | Expected Time | Est. Modal Cost |
|---|---|---|---|---|---|
| **Stage A1 ($\mathcal{L}_{\text{beh}}$)** | 1x Nvidia B200 | $\ge 40$ GB | 8 (accum 2) | ~2.5 – 3.5 hrs | **~$13.20** |
| **Stage A2 (Factorial Rollouts)** | 1x Nvidia A100-80GB | $\ge 24$ GB | Sequential | ~3.5 – 4.5 hrs | **~$10.30** |
| **Stage A2 (Flow-DPO)** | 1x Nvidia H100 / B200 | $\ge 40$ GB | 8 | ~1.5 hrs | **~$6.60** |
| **CALVIN 1000 Eval** | 1x Nvidia A100-80GB | 24 GB | 1 | ~1.5 hrs | **~$4.40** |

*(Modal provides **$30 free credits/month**, which fully covers Stage A1 training and initial evaluation).*

---

## 7. Datasets & Resources

- **Pre-trained Checkpoints:** [Hugging Face: SaFFire/sge-goal](https://huggingface.co/SaFFire/sge-goal)
- **CALVIN Goal Dataset:** [SuyashAjitChavan2002/Calvin-Dataset-Final](https://huggingface.co/datasets/SuyashAjitChavan2002/Calvin-Dataset-Final) (Mirror: `RiddhiCh/Calvin-Dataset-Final`)
- **Base Generator Model:** `black-forest-labs/FLUX.1-Kontext-dev`
- **Modal Documentation:** [Modal Model Weights Guide](https://modal.com/docs/guide/model-weights)
