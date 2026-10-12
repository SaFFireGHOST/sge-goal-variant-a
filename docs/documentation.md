# SGE-Goal: Language-Guided Visual Goal Generation for Robotic Manipulation

## 1. Overview & Project Architecture

**SGE-Goal** (Structured Goal Editing) is a framework for language-guided image editing (LGIE) designed for visual goal generation in robotic manipulation. It allows robot policies to generate photorealistic, goal-conditioned target visual states given an initial camera observation and natural language instructions.

The system incorporates:
1. **Diffusion-Based Goal Generation Backbones**: Fine-tuned versions of **FLUX.1-Kontext** and **FLUX.2-Klein** using LoRA with Static Scene Consistency Loss (SSCL) and Edit Region Mask (ERM) modeling.
2. **Robotic Dataset Preprocessing Pipeline**: Automated segmentation and tracking pipeline extracting initial observation, final goal state, robotic arm masks, and static region masks from robot demonstration datasets across **BridgeData V2**, **CALVIN**, and **LIBERO**.
3. **Evaluation Benchmarks**:
   - **SGE-Goal-Evaluator**: 15 distinct diagnostic evaluation dimensions (R1–R15) evaluating stationary object preservation, distractor invariance, attribute editing accuracy, and physical plausibility.
   - **I2I-Bench**: Automated multi-metric evaluator combining specialist models (Q-Insight, ArtiMuse, Sa2VA, DINOv3, GOT-OCR2.0) and Large Multimodal Models (Qwen2.5-VL / Qwen-VL-Max / Gemini).
4. **Downstream Policy Evaluation**: Integration with **SIMPLER** (SAPIEN-based real-to-sim benchmark) and **CALVIN** goal-conditioned policies (GCBC, GC-DDPM, GR-MG, GHIL-Glue).

---

## 2. Hugging Face Datasets & Metadata

The models are trained and evaluated across three primary robotic manipulation benchmarks. The underlying training data (Parquet shards and Git LFS objects) is **100% byte-for-byte identical** between the active training repositories (`SuyashAjitChavan2002/`) and the officially documented dataset mirrors (`RiddhiCh/`).

### 2.1 Dataset Repositories Comparison

| Benchmark | Active Code Repo (Used in Training Scripts) | Official Mirror (With SGE-Goal Metadata Card) | Size | License |
| :--- | :--- | :--- | :--- | :--- |
| **BridgeData V2** | [SuyashAjitChavan2002/LGRM_BridgeDataV2_Teleoperated](https://huggingface.co/datasets/SuyashAjitChavan2002/LGRM_BridgeDataV2_Teleoperated) | [RiddhiCh/LGRM_BridgeDataV2_Teleoperated](https://huggingface.co/datasets/RiddhiCh/LGRM_BridgeDataV2_Teleoperated) | ~8.29 GB | `cc-by-4.0` |
| **CALVIN** | [SuyashAjitChavan2002/Calvin-Dataset-Final](https://huggingface.co/datasets/SuyashAjitChavan2002/Calvin-Dataset-Final) | [RiddhiCh/Calvin-Dataset-Final](https://huggingface.co/datasets/RiddhiCh/Calvin-Dataset-Final) | ~2.63 GB | `mit` |
| **LIBERO** | [SuyashAjitChavan2002/LGRM_Libero_10_90_Combined](https://huggingface.co/datasets/SuyashAjitChavan2002/LGRM_Libero_10_90_Combined) | [RiddhiCh/LGRM_Libero_10_90_Combined](https://huggingface.co/datasets/RiddhiCh/LGRM_Libero_10_90_Combined) | ~0.29 GB | `mit` |
| **Debug / Dummy** | *N/A* | [RiddhiCh/dummy-image-editing-dataset](https://huggingface.co/datasets/RiddhiCh/dummy-image-editing-dataset) | < 50 MB | *Testing* |

> **Note on Parity**: Both repository URLs point to identical Git LFS object IDs (e.g., `test.parquet` sha `fd3b442d...` for BridgeData V2). The `RiddhiCh` repositories contain official YAML frontmatter metadata, dataset tags, task categories, and formal dataset documentation cards.

---

### 2.2 Dataset Splits & Example Counts

| Dataset | Train Split | Validation Split | Test Split | Total Samples |
| :--- | :--- | :--- | :--- | :--- |
| **`LGRM_BridgeDataV2_Teleoperated`** | 40,851 | 1,000 | 2,000 | 43,851 |
| **`Calvin-Dataset-Final`** | 22,000 | 966 | 1,087 | 24,053 |
| **`LGRM_Libero_10_90_Combined`** | 6,464 | 358 | 376 | 7,198 |

*Note on LIBERO sources*: `LGRM_Libero_10_90_Combined` was created by combining `SuyashAjitChavan2002/LGRM_Libero_10` and `SuyashAjitChavan2002/LGRM_Libero_90` using [`LGIE-ER-RFL-SSCL/combine_libero_datasets.py`](file:///home/sgeextn/SGE-Goal-Cleaned/LGIE-ER-RFL-SSCL/combine_libero_datasets.py).

---

### 2.3 Feature Schema & Metadata

Each dataset sample adheres to the following unified schema:

```json
{
  "id": "string (unique sample identifier, e.g. Libero10_0001 or episode hash)",
  "original_image": "image (PIL.Image - starting scene observation)",
  "edited_image": "image (PIL.Image - visual target / goal state)",
  "initial_edit_region_mask": "image (PIL.Image - binary mask of manipulated object at t=0)",
  "final_edit_region_mask": "image (PIL.Image - binary mask of manipulated object at t=goal)",
  "edit_instruction": "string (natural language prompt describing the robotic task)"
}
```

#### Dataset Metadata Headers (YAML Frontmatter)
```yaml
task_categories:
  - image-to-image
  - robotics
language:
  - en
tags:
  - robotics
  - manipulation
  - goal-image-generation
  - image-editing
  - bridgedata-v2 / calvin / libero
```

#### Loading Data in Python
```python
from datasets import load_dataset

# Load any dataset using either the active training ID or the RiddhiCh mirror:
bridge_data = load_dataset("RiddhiCh/LGRM_BridgeDataV2_Teleoperated")
calvin_data = load_dataset("RiddhiCh/Calvin-Dataset-Final")
libero_data = load_dataset("RiddhiCh/LGRM_Libero_10_90_Combined")
```

---

## 3. Directory Structure & File Catalog

```
SGE-Goal-Cleaned/
├── Data-Processing-Optimisations/       # Trajectory preprocessing and mask generation
├── LGIE-ER-RFL-SSCL/                    # Local training & FLUX-Kontext fine-tuning
├── LGIE-ER-RFL-SSCL-Modal/              # Modal serverless training for FLUX & Klein
├── erm-prediction-erm-prompts-erm-loss/ # Local ERM training & prediction
├── erm-prediction-erm-prompts-erm-loss-modal/ # Modal ERM training pipelines
├── Inferencers/                         # Goal image generation inference models
├── SGE-Goal-Evaluator/                  # 15-dimension diagnostic evaluation suite
├── I2I-Bench/                           # Multi-metric automated evaluation benchmark
├── RM-Policy/                           # Reward modeling & policy evaluation
└── simpler-eval/                        # SIMPLER simulation rollout evaluation
```

---

### 3.1 `Data-Processing-Optimisations/`
Contains the end-to-end data processing pipeline that converts raw teleoperated robot demonstration videos into annotated image-editing datasets with spatial masks and text instructions.

* **Core Scripts**:
  * [`data_processing_v3.py`](file:///home/sgeextn/SGE-Goal-Cleaned/Data-Processing-Optimisations/data_processing_v3.py): Primary multi-threaded production pipeline. Uses SAM and EVF-SAM for robotic arm masking, XMem for cross-frame video tracking, LoFTR for keypoint correspondence extraction, and BLIP-2 for caption verification, before pushing shards to Hugging Face.
  * [`data_processing_v2.py`](file:///home/sgeextn/SGE-Goal-Cleaned/Data-Processing-Optimisations/data_processing_v2.py): Predecessor version of the data extraction pipeline.
  * [`backup_data_processing_v3.py`](file:///home/sgeextn/SGE-Goal-Cleaned/Data-Processing-Optimisations/backup_data_processing_v3.py) & [`old_code.py`](file:///home/sgeextn/SGE-Goal-Cleaned/Data-Processing-Optimisations/old_code.py): Reference and backup implementations.
  * [`benchmark_pipeline.py`](file:///home/sgeextn/SGE-Goal-Cleaned/Data-Processing-Optimisations/benchmark_pipeline.py): Profiles latency and memory usage of individual pipeline stages.
  * [`movement_segregation.py`](file:///home/sgeextn/SGE-Goal-Cleaned/Data-Processing-Optimisations/movement_segregation.py): Filters trajectory segments by robot gripper movement magnitude.
* **Helper Utilities (`scripts/`)**:
  * `copy_first_50.py`: Utility to sample initial subsets of episodes for quick verification.
  * `corrupt.py`: Checks for corrupted images in video frames.
  * `dedup_corrupt_images.py`: Deduplicates corrupt image logs.
* **Submodules & Models**:
  * `evf_sam_model/`: Explicit Vision-Language Grounding / Referring SAM model.
  * `segment-anything/`: Meta's Segment Anything Model (SAM) source.
  * `sam_vit_h_4b8939.pth`: ViT-H SAM model checkpoint (2.56 GB).
  * `XMem/`: Semi-supervised video object segmentation tracker.
  * `loftr_src/` & `loftr_weights/`: Detector-free local feature matching with transformers.
  * `Test_Data/`: Sample trajectory data for testing processing scripts locally.
* **Metadata & Verification Logs**:
  * `all_dirs.sorted`, `dirs_nonzero.sorted`: Directory indexes of robot episodes.
  * `corrupt_episodes.txt`, `corrupt_images.txt`: Trajectory auditing and error logs.
  * `episode_image_counts.csv`, `top20_by_image_count.csv`: Statistics on frames per episode.
  * `bridge_data_generation.json`: Hyperparameter config for BridgeData extraction.

---

### 3.2 `LGIE-ER-RFL-SSCL/`
Core implementation of Language-Guided Image Editing with Static Scene Consistency Loss (SSCL) and Region-Focused Loss (RFL) for FLUX.1-Kontext.

* [`flux_kontext_lora_finetuning.py`](file:///home/sgeextn/SGE-Goal-Cleaned/LGIE-ER-RFL-SSCL/flux_kontext_lora_finetuning.py): Main Hugging Face Diffusers training script for FLUX.1-Kontext. Integrates LoRA adapters, gradient checkpointing, multi-image conditioning, and auxiliary SSCL loss to prevent hallucinations in unedited scene regions.
* [`pipeline_flux_kontext_multi_imgs.py`](file:///home/sgeextn/SGE-Goal-Cleaned/LGIE-ER-RFL-SSCL/pipeline_flux_kontext_multi_imgs.py): Custom Diffusers inference and validation pipeline supporting multi-image inputs and spatial mask guidance.
* [`flux_kontext_eval_utils.py`](file:///home/sgeextn/SGE-Goal-Cleaned/LGIE-ER-RFL-SSCL/flux_kontext_eval_utils.py): Evaluation metrics (PSNR, SSIM, LPIPS) and qualitative visual grid generator during training.
* [`combine_libero_datasets.py`](file:///home/sgeextn/SGE-Goal-Cleaned/LGIE-ER-RFL-SSCL/combine_libero_datasets.py): Combines `LGRM_Libero_10` and `LGRM_Libero_90` into the consolidated Hugging Face dataset.
* [`benchmark_batch_sizes.py`](file:///home/sgeextn/SGE-Goal-Cleaned/LGIE-ER-RFL-SSCL/benchmark_batch_sizes.py): Scripts to test VRAM footprint across different batch sizes on target GPUs.

---

### 3.3 `LGIE-ER-RFL-SSCL-Modal/`
Modal serverless GPU wrappers and deployment pipelines for fine-tuning models in the cloud across distributed H100 / B200 GPUs.

* **Benchmark Subdirectories**:
  * `flux2_klein_bridge_finetuning/`: FLUX.2-Klein LoRA training on `LGRM_BridgeDataV2_Teleoperated`.
    * `modal_flux_2_klein_train.py`: Modal entrypoint configuring remote container, GPU specs (B200), volume mounts, and checkpoint sync.
    * `flux_2_klein_lora_finetuning.py`: PyTorch training script for FLUX.2-Klein.
  * `flux2_klein_calvin_finetuning/`: FLUX.2-Klein LoRA training on `Calvin-Dataset-Final`.
  * `flux2_klein_libero_finetuning/`: FLUX.2-Klein LoRA training on `LGRM_Libero_10_90_Combined`.
  * `flux_kontext_bridge_finetuning/`: FLUX.1-Kontext training on BridgeData V2 with watchdog monitoring (`watchdog_training.py`), parallel visualization (`modal_parallel_viz.py`), and volume synchronizers (`sync_modal_volume.py`).
  * `flux_kontext_calvin_finetuning/`: FLUX.1-Kontext training on CALVIN.

---

### 3.4 `erm-prediction-erm-prompts-erm-loss/` & `erm-prediction-erm-prompts-erm-loss-modal/`
Experiments exploring explicit **Edit Region Mask (ERM)** modeling, where the diffusion model is conditioned on or predicts the region of change directly.

* **Local Directory (`erm-prediction-erm-prompts-erm-loss/`)**:
  * `flux_kontext_lora_finetuning.py`: Fine-tuning script with ERM prompt injection and region-specific loss weighting.
  * `pipeline_flux_kontext_multi_imgs.py`: Multi-image conditioning pipeline adapted for ERM.
  * `flux_kontext_eval_utils.py`: Metric computation focused on edited versus preserved regions.
  * `monitor_training.sh` / `monitor_training.log`: Background training process watchdog.
* **Modal Directory (`erm-prediction-erm-prompts-erm-loss-modal/`)**:
  * `flux2_klein_bridge_finetuning/`, `flux2_klein_calvin_finetuning/`, `flux2_klein_libero_finetuning/`: Cloud implementations with `modal_flux_2_klein_erm_train.py` and `flux_2_klein_erm_finetuning.py`.
  * `flux_kontext_bridge_finetuning/`, `flux_kontext_calvin_finetuning/`: Cloud implementations for FLUX Kontext.

---

### 3.5 `Inferencers/`
Contains standalone inference pipelines for all tested baseline models and the proposed goal generation architectures. Each subfolder includes test datasets (`bridge_data_v2_tel_test_data`, `calvin_abcd_test_data`, `libero_10_90_combi_test_data`) and inference generation scripts.

* **Model Subdirectories**:
  * `flux2-klein/`: Zero-shot pretrained FLUX.2-Klein (`modal_flux2_klein_pretrained_inference.py`).
  * `flux2-klein-ft/` & `flux2-klein-bft-i1/` / `flux2-klein-bft-i2/`: Fine-tuned FLUX.2-Klein checkpoints (iterations 1 and 2).
  * `flux-kontext-dev/`: Zero-shot pretrained FLUX.1-Kontext-dev.
  * `flux-kontext-dev-ft/` & `flux-kontext-dev-bft-i1/` / `flux-kontext-dev-bft-i2/`: Fine-tuned FLUX.1-Kontext checkpoints.
  * `qwen-image-edit/`: Qwen-VL based image-to-image editing baseline (`modal_qwen_ie_inference.py`, `modal_qwen_ie_parallel_inference.py`).
  * `qwen-kontext-hybrid-ft and qwen-klein-hybrid-ft/`: Hybrid models combining Qwen semantic planning with FLUX visual rendering.
  * `Clover/`: CLOVER visual planning baseline (`clover_visual_planner.py`, `modal_clover_inference.py`).
  * `SuSIE/`: Subgoal Synthesis via Image Editing (diffusion-based visual goal baseline).
  * `gemini-3-pro-image/` & `gpt-image-1.5/`: Proprietary API baselines for comparison.

---

### 3.6 `SGE-Goal-Evaluator/`
A comprehensive, automated 15-dimension diagnostic evaluation framework designed specifically to assess the quality, realism, and precision of language-guided robotic goal image editing.

* **Dimension Evaluator Scripts**:
  * [`R1.py`](file:///home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-Evaluator/R1.py): R1 - Unmentioned Stationary Objects (preservation of non-target items).
  * [`R2.py`](file:///home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-Evaluator/R2.py): R2 - Object Attribute Preservation (colors, shapes, textures of unchanged objects).
  * [`R3.py`](file:///home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-Evaluator/R3.py): R3 - Background and Scene Consistency (lighting, table surface, workspace boundaries).
  * [`R4.py`](file:///home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-Evaluator/R4.py): R4 - Distractor Invariance (immunity to visual clutter).
  * [`R5-6-7.py`](file:///home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-Evaluator/R5-6-7.py): Multi-dimension evaluator for instruction alignment and spatial displacement.
  * [`R8-9.py`](file:///home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-Evaluator/R8-9.py): Object manipulation accuracy and state change validity.
  * [`R10-11-12.py`](file:///home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-Evaluator/R10-11-12.py): Directional, relational, and spatial accuracy checks.
  * [`R13.py`](file:///home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-Evaluator/R13.py): Physical plausibility and collision checking.
  * [`R14-15.py`](file:///home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-Evaluator/R14-15.py): Compositional multi-step reasoning.
* **Evaluation Infrastructure**:
  * [`lgie_evaluator.py`](file:///home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-Evaluator/lgie_evaluator.py): Base evaluation engine integrating SAM segmentation, LoFTR optical correspondence, and LMM scoring.
  * [`modal_eval_runner.py`](file:///home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-Evaluator/modal_eval_runner.py): Modal harness to run all R-scripts in parallel across cloud GPUs.
  * [`create_data_folder.py`](file:///home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-Evaluator/create_data_folder.py): Auto-generates structured evaluation test folders containing before/after pairs and prompts.
  * [`sge_goal_eval_aggregator.py`](file:///home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-Evaluator/sge_goal_eval_aggregator.py): Summarizes scores across all dimensions and models into formatted tables.
  * `lgie-eval-data/`: Curated benchmark images and ground truth goal states (~59 MB).
  * `FINAL-EVAL-RUNS/` & `aggregated-eval-results/`: Aggregated benchmark metric outputs.

---

### 3.7 `I2I-Bench/`
Implementation of the standardized single-reference evaluation benchmark from *"I2I-Bench: A Comprehensive Benchmark Suite for Image-to-Image Editing Models"* (Wang et al., 2025).

* **Evaluator Modules**:
  * [`i2i_bench_evaluator.py`](file:///home/sgeextn/SGE-Goal-Cleaned/I2I-Bench/i2i_bench_evaluator.py): Core evaluator calculating metrics across 15 dimensions using specialist models (Q-Insight for image quality, ArtiMuse for aesthetic score, Sa2VA + DINOv3 for region fidelity, GOT-OCR2.0 for text).
  * [`phase_runner.py`](file:///home/sgeextn/SGE-Goal-Cleaned/I2I-Bench/phase_runner.py): Staged execution runner for multi-phase evaluation jobs.
  * [`modal_i2ibench_eval.py`](file:///home/sgeextn/SGE-Goal-Cleaned/I2I-Bench/modal_i2ibench_eval.py): Serverless Modal runner for parallel benchmark execution.
* **Orchestration & Launching**:
  * `_launch_all_26.sh`, `_launch_extra_8.sh`: Batch scripts launching evaluation runs on Modal.
  * `_sync_all_26.py`, `_final_sync_and_status.py`: Downloads results and logs from Modal volumes.
* **Benchmark Data & Results**:
  * `data/`: Standardized numbered test cases.
  * `i2ibench-human-alignment-test/`: Human evaluation alignment study data.
  * `FINAL-EVAL-RUNS-DICEnI2IBench/`: Processed benchmark run outputs.
  * `i2ibench_evaluation_results.txt`: Comprehensive result tables.

---

### 3.8 `RM-Policy/`
Reward modeling and goal-conditioned policy evaluation using generated visual goals.

* **`GHIL-Glue/`**: Goal-conditioned Hierarchical Imitation Learning and GCBC (Goal-Conditioned Behavioral Cloning).
  * [`diffusion_gc_policy.py`](file:///home/sgeextn/SGE-Goal-Cleaned/RM-Policy/GHIL-Glue/diffusion_gc_policy.py): Diffusion-based goal-conditioned action policy.
  * [`gcbc_train_config.py`](file:///home/sgeextn/SGE-Goal-Cleaned/RM-Policy/GHIL-Glue/gcbc_train_config.py): Training configurations supporting goal-relabeling (delta goals, UniPi, generated encode-decode goals).
  * `mini_dataset/`: Local TFRecord dataset shards (`0.tfrecord`, `1.tfrecord`).
* **`GR-MG/`**: Multi-modal Goal-Conditioned Policy code for CALVIN and simulated manipulation.
  * `goal_gen/`: Goal image generation conditioning modules.
  * `policy/`: Transformer / ResNet policy backbones conditioned on visual subgoals.
* **`CALVIN-EVALS/` & `CALVIN-ABLATION-EVALS/`**: Evaluation logs and rollout benchmarks on CALVIN ABC->D tasks.

---

### 3.9 `simpler-eval/`
Evaluation environment connecting SGE-Goal with **SIMPLER** (Simulation for Policy Evaluation in Real-to-sim environments) for testing manipulation policies on simulated WidowX and Google Robot setups.

* [`modal_rollout_gc.py`](file:///home/sgeextn/SGE-Goal-Cleaned/simpler-eval/modal_rollout_gc.py): Main Modal runner for executing goal-conditioned closed-loop policy rollouts.
* [`modal_phaseA_dump_obs.py`](file:///home/sgeextn/SGE-Goal-Cleaned/simpler-eval/modal_phaseA_dump_obs.py): Dumps initial observations from simulation environments to feed into goal generators.
* [`orchestrate_goals.py`](file:///home/sgeextn/SGE-Goal-Cleaned/simpler-eval/orchestrate_goals.py): Coordinates goal generation and sends goals to the policy agent.
* [`pinocchio_ik_backend.py`](file:///home/sgeextn/SGE-Goal-Cleaned/simpler-eval/pinocchio_ik_backend.py): Inverse kinematics solver using Pinocchio for WidowX robotic arms.
* [`modal_sapien_2204.py`](file:///home/sgeextn/SGE-Goal-Cleaned/simpler-eval/modal_sapien_2204.py) & [`modal_sapien_kisak.py`](file:///home/sgeextn/SGE-Goal-Cleaned/simpler-eval/modal_sapien_kisak.py): Container setups with Vulkan rendering and SAPIEN simulation engines.
* [`seed_table_driver.py`](file:///home/sgeextn/SGE-Goal-Cleaned/simpler-eval/seed_table_driver.py) & [`overnight_run.py`](file:///home/sgeextn/SGE-Goal-Cleaned/simpler-eval/overnight_run.py): Automated test harnesses running multiple seeds and evaluating success rates.

---

## 4. Workflows & Usage Quickstart

### 4.1 Running Fine-Tuning on Modal
To run FLUX.2-Klein or FLUX.1-Kontext training on Modal:
```bash
# Example: Train FLUX.2-Klein on BridgeData V2
cd LGIE-ER-RFL-SSCL-Modal/flux2_klein_bridge_finetuning
modal run modal_flux_2_klein_train.py
```

### 4.2 Running Benchmark Inference
To generate goal images using fine-tuned models:
```bash
# Example: Generate goals using FLUX.2-Klein
cd Inferencers/flux2-klein-bft-i2
modal run modal_flux2_klein_ie_inference.py
```

### 4.3 Running SGE-Goal-Evaluator
To evaluate generated images on the 15 diagnostic dimensions:
```bash
cd SGE-Goal-Evaluator
# Run all dimensions on Modal
modal run modal_eval_runner.py --script R1.py
modal run modal_eval_runner.py --script R2.py
# Aggregate results
python3 sge_goal_eval_aggregator.py
```
