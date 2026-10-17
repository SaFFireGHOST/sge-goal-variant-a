# Variant A: Next Steps Plan

**Budget:** ~$25 of Modal credit (free plan, $30/month refresh)
**Goal:** Decide cheaply whether Variant A (training the generator against a frozen policy) has any signal before spending money on training.

---

## 0. Where things stand (read first)

| Fact | Consequence |
|---|---|
| `variant_a_alignment/*.py` are **mock code**: a toy CNN generator and toy CNN policy (`train_a_dense.py`), `hash()` rewards (`factorial_rollouts.py`), and DPO on `torch.randn` latents (`train_a_flow_dpo.py`). | `rm-policy-runs:/variant_a_dense_checkpoints/` holds toy CNN weights, **not a FLUX LoRA**. Don't run inference or eval on it, and don't re-run `modal_train_a_dense.py` (B200 at about $6.70/hr for nothing). |
| `eval_method.py` runs each HF row **once, deterministically** (`seed_everything(0)`, fixed reset from `episode_<start>.npz`). | Re-running with "K_s seeds" gives identical results. Start-state perturbations are needed to get variance. |
| CL2–5 come from `aggregate_chain_success.py`, which multiplies **independent single-step** per-row successes over clusters. These aren't closed-loop chained rollouts. | All of Variant A's headroom has to come from **per-row single-step success**. "Long horizons hold the headroom" doesn't apply to this metric. |
| GR-MG: GT goal 73.6% vs SGE-Goal 73.1% (n=1087, SE ≈ 1.35 pts). | The aggregate gap is noise. The real question is **per-row best-of-N headroom**: GT isn't a ceiling, because a generated goal can work better for the policy than the real frame. |

**The plan:** run three cheap phases and stop at each gate if the answer is "no".

```
Phase 0  Sanity + GT baseline            ~$0–2    (1 day)
Phase 1  Gate 0: best-of-N headroom       ~$9–15   (2–3 days)   ← the go/no-go decision
Phase 2  Real Variant A (only if Gate 0 passes)  next month's credits or lab GPUs
```

---

## 1. GPU choices (cost per hour including 4 CPU + 32 GiB RAM ≈ $0.45/hr)

| Job | GPU | ~$/hr | Why |
|---|---|---|---|
| CALVIN rollouts (GR-MG inference + PyBullet) | **T4** | ~$1.05 (about $0.80 if you drop to 2 CPU / 16 GiB) | The policy is small and the simulator is CPU-bound, so faster GPUs add nothing. Already configured in `modal_calvin_eval.py`. |
| FLUX Kontext goal generation (Step 2 + Step 3 inference) | **L40S (48 GB)** | ~$2.40 | FLUX-12B in bf16 + T5 + VAE ≈ 35 GB fits with no CPU offload. L4/A10G (24 GB) need offload and end up slower and more expensive per image. H100 is about 2× faster at about 1.8× the price, roughly a wash, so only use it if L40S is unavailable. |
| Mapping / aggregation / analysis | **CPU only** (`gpu=None`) or local | ~$0.05 | No GPU work. |
| Flow-DPO / L_beh training (Phase 2) | H100 or lab RTX Pro 6000 | $4.40 | Not in this month's budget. |

**Rules for saving money**
- Always run a **5-row pilot** first and multiply its wall-clock time out before launching a full job.
- Use `--detach` for long jobs, and check `modal app list` afterwards for anything left running.
- Write generator outputs **directly into the `rm-policy-eval-data` volume** so nothing round-trips through your laptop.
- Your lab has 4× A6000 + an RTX Pro 6000 (see proposal §8). If you can get on them, run FLUX generation there for free and use Modal only for CALVIN rollouts.

---

## Phase 0: Cleanup and GT baseline (~$0–2)

### 0.1 Fence off the mock code (no cost)

**Edit:** `variant_a_alignment/train_a_dense.py`, `factorial_rollouts.py`, `train_a_flow_dpo.py`

Add at the top of `main()` in each:

```python
if "--allow_mock" not in sys.argv:
    raise SystemExit(
        "MOCK IMPLEMENTATION: surrogate models / simulated rewards. "
        "Results are meaningless. See plan.md Phase 2."
    )
```

**Edit:** `README.md` §5.2–5.5. Mark these steps *"not implemented — mock code"* so nobody (including future you) mistakes them for real results.

**Optional:** delete the toy checkpoint so it can't be picked up by accident. This can't be undone, so check its contents first:
```bash
modal volume ls rm-policy-runs variant_a_dense_checkpoints/checkpoint-epoch-15
modal volume rm -r rm-policy-runs variant_a_dense_checkpoints   # only once you're sure
```

### 0.2 Two integrity checks on the paper's numbers (no cost)

1. **Which masks did the SGE-Goal row in Table 2 use?** The eval folder is named `calvin_abcd_ier_fer_gt_test_data`. If "ier_fer_gt" means GT initial and final masks were fed to Step 3, then 73.1% is an oracle-mask number, not the full pipeline. Find the script that produced that folder.
2. **How were CL2–5 computed in the paper?** If they came from `aggregate_chain_success.py`, then the paper's "generated goal becomes the next step's I_o" description and the comparison with GR-MG's native average length of 4.04 (closed-loop protocol) don't match. Fix the wording before the camera-ready.

### 0.3 GT baseline: check whether it already exists before paying

```bash
modal volume ls rm-policy-runs
modal volume ls rm-policy-runs eval_hf_task_D_D_gt_final_epoch47     # earlier GT run?
modal volume ls rm-policy-runs eval_method
```

If a full 1087-row GT `per_sample*.jsonl` already exists, **skip the run** and go straight to aggregation.

Otherwise (~$1.5, 5 T4s × ~18 min):

```bash
cd policy
modal run --detach modal_calvin_eval.py::run_method \
  --method-name ground_truth --use-hf-parquet-goal --num-shards 5
```

Download and aggregate (`--methods` is required, and the plan you pasted left it out):

```bash
mkdir -p results
modal volume get rm-policy-runs eval_method/ground_truth results/
python gr_mg_evaluate/aggregate_chain_success.py \
  --chain_skeletons chain_skeletons.json \
  --results_dir results --methods ground_truth \
  --out_csv results/table_gt.csv --out_json results/table_gt.json
```

**Check:** CL1 ≈ 73.6% means the harness reproduces the paper. If it doesn't, stop and debug before Phase 1.

---

## Phase 1: Gate 0, best-of-N headroom (~$9–15)

**Question:** for the same row, do different SGE-Goal samples give *different* policy outcomes, and does picking the best one help on held-out perturbations?

> **Status: implemented.** Files: `variant_a_alignment/gate0_select_rows.py`,
> `generator/modal_gate0_generate.py`, `variant_a_alignment/gate0_analyze.py`, plus edits to
> `policy/gr_mg_evaluate/eval_method.py` and `policy/modal_calvin_eval.py`.
> `gate0_rows.json` (150 rows over 34 tasks) is already generated. Run order is in **§1.7**.
> The generator reproduces the paper's CALVIN inference exactly (from
> `inferencers/sge_goal_pipeline/`): **256 px, 10 denoising steps**, Step 2 guidance 1.0,
> Step 3 default guidance, and the predicted mask passed through JPEG as the original pipeline did.
> That makes generation about 5× cheaper than estimated below.

**Design (fits the budget):**
- **150 rows**, stratified across CALVIN tasks (about 4–5 per task), fixed random seed.
- Per row: **K_m = 3** Step 2 masks × **K_a = 2** Step 3 appearances = **6 goals**, plus the **GT goal** as a reference.
- Each goal is rolled out from **P = 3 start states**: p=0 is the original reset, and p=1,2 add small robot-joint noise.
- Rollouts: 150 × 7 × 3 = **3,150** (about 4.5 T4-hours, ~$4.5).
- Generation: 150 × (3 mask passes + 6 image passes) = **1,350 FLUX passes**. At about 5 s/pass on L40S that's about 1.9 h (~$4.5); at about 15 s/pass it's about 5.6 h (~$13). **The pilot decides.** If you're over ~8 s/pass, cut to 100 rows.

### 1.1 New file: `variant_a_alignment/gate0_select_rows.py` (local, free)

- Load `policy/hf_calvin_mapping.json`, keep rows that have a CALVIN match, group by `task`, and sample ~150 with a fixed seed.
- Write `gate0_rows.json` as `{"rows": [hf_row, ...], "seed": 0}`.
- Upload: `modal volume put -f rm-policy-runs gate0_rows.json /gate0_rows.json`

### 1.2 New file: `generator/modal_gate0_generate.py` (L40S)

A Modal app that, for each row in `gate0_rows.json`:

1. Loads the HF test row (`original_image`, `initial_edit_region_mask`, `edit_instruction`). **Use the dataset's M_i** (Step 1 is training-free; this saves cost and isolates Steps 2 and 3).
2. **Step 2:** `FluxKontextPipeline` (from `pipeline_flux_kontext_multi_imgs.py`) + `load_lora_weights("/ckpts/ckpts/calvin-fer-mask-prediction.safetensors")`. Sample K_m=3 final masks with `generator=torch.Generator().manual_seed(1000*row + j)`. Binarize: channel mean > 0.5. Reuse exactly the input conventions of the Step 2 inference script in `Inferencers/` that produced the paper's predicted masks.
3. **Step 3:** swap to `/ckpts/ckpts/calvin-image-editing.safetensors` (`unload_lora_weights()` then `load_lora_weights()`). For each mask, sample K_a=2 goals with seeds `1000*row + 10*j + k`, calling `pipe(original_image=..., initial_edit_region_mask=..., final_edit_region_mask=M_f_j, prompt=build_enhanced_prompt(instr))`. Use the **same resolution, step count and guidance as the paper's inference**.
4. Writes straight into the eval volume, in the layout `eval_method.py` already reads:
   ```
   /eval_data/gate0/m{j}a{k}/{hf_row+1}/edited_image.png
   /eval_data/gate0/m{j}a{k}/{hf_row+1}/edit_instruction.txt
   /eval_data/gate0/masks/{hf_row+1}/mf_{j}.png
   ```
5. Takes `--row-start/--row-end` and `--limit` arguments so you can pilot first and shard the rest.

Efficiency tips: run all of Step 2 for a shard first, then all of Step 3 (one LoRA swap per shard, not per row). Encode the prompt once per row and pass `prompt_embeds`. Use `torch.inference_mode()` and bf16.

```python
# skeleton
@app.function(gpu="L40S", timeout=6*3600, volumes={"/ckpts": ckpt_vol, "/eval_data": eval_vol, "/runs": runs_vol, "/hf_cache": hf_vol},
              secrets=[modal.Secret.from_name("huggingface")])
def generate_shard(row_start: int, row_end: int, km: int = 3, ka: int = 2): ...

@app.local_entrypoint()
def main(num_shards: int = 1, limit: int = 0, km: int = 3, ka: int = 2): ...
```

**Pilot, then the full run:**
```bash
cd generator
modal run modal_gate0_generate.py --limit 5            # time it; inspect the images
modal volume get rm-policy-eval-data gate0 ./gate0_pilot   # eyeball masks/goals look sane + differ
modal run --detach modal_gate0_generate.py --num-shards 2
```

**Check mask diversity on the pilot.** If the 3 masks per row are near-identical (pairwise IoU > 0.95), Step 2 is collapsing to one output and Gate 0 for the mask stage will fail. Stop and report that, because it's a finding in its own right.

### 1.3 Edit: `policy/gr_mg_evaluate/eval_method.py`

Add three arguments and a perturbation loop. Everything else stays as it is.

```python
p.add_argument("--rows_json", type=str, default="")      # evaluate only these hf_rows
p.add_argument("--variants", type=str, default="")       # comma list, e.g. "m0a0,m0a1,...,gt"
p.add_argument("--num_perturb", type=int, default=1)     # P start states per (row, variant)
p.add_argument("--perturb_std", type=float, default=0.02) # rad, on arm joints
```

- If `--rows_json` is set, iterate over that list instead of `range(row_start, row_end)`, with `row_start/row_end` slicing **into the list** for sharding.
- If `--variants` is set, loop over variants **inside** each row, so the env and model are loaded once per shard. Variant `gt` takes the goal from the HF parquet. Any other variant reads from `--method_dir/<variant>/`.
- Perturbation, applied right after `_read_npz_init`. CALVIN `robot_obs` = [tcp_pos 3, tcp_orn 3, gripper_width 1, **arm_joints 7 (idx 7:14)**, gripper_action 1]:
  ```python
  for pk in range(args.num_perturb):
      r_obs = robot_obs.copy()
      if pk > 0:
          rng = np.random.default_rng(100_000 * hf_row + pk)   # same perturbation for every variant
          r_obs[7:14] += rng.normal(0.0, args.perturb_std, 7)
      env.reset(robot_obs=r_obs, scene_obs=scene_obs)
      model.reset()
      ...
  ```
  Use the **same** perturbation for every variant of a row (common random numbers). That makes variant comparisons paired and much less noisy.
- Add `"variant"` and `"perturb"` to each JSONL record.
- **Sanity check:** run one row twice with `num_perturb=1`. The results must be identical (confirms determinism). Also confirm p=1,2 actually change the outcome on at least some rows. If they never do, raise `perturb_std` to 0.05, or also add `scene_obs` object-position noise of about 5 mm.

### 1.4 Edit: `policy/modal_calvin_eval.py`

- Pass the new arguments through `eval_method_shard` (`rows_json`, `variants`, `num_perturb`, `perturb_std`).
- Add a local entrypoint `run_gate0(num_shards=8, num_perturb=3, variants="m0a0,m0a1,m1a0,m1a1,m2a0,m2a1,gt")` that splits the row list into shards and calls `eval_method_shard.starmap`, with `method_dir=/eval_data/gate0` and the parquet available for `gt`.
- Optional saving: change `eval_method_shard` to `cpu=2, memory=16384`, and check that one shard doesn't OOM first.

```bash
cd policy
# pilot: 5 rows, all variants, 3 perturbations (~105 rollouts, ~10 min on 1 T4)
modal run modal_calvin_eval.py::run_gate0 --num-shards 1 --limit 5
# full
modal run --detach modal_calvin_eval.py::run_gate0 --num-shards 8
modal volume get rm-policy-runs eval_method/gate0 results/
```

### 1.5 New file: `variant_a_alignment/gate0_analyze.py` (local, free)

Inputs: `results/gate0/per_sample_*.jsonl` and `gate0/masks/`. Outputs: `results/gate0_report.json` plus a printed table.

Compute:
1. **Single-sample SGE-Goal rate**: the mean over rows, the 6 variants and the P perturbations. This is your realistic baseline. Compare it with **GT** on the same rows and perturbations.
2. **Mixed-outcome fraction**: the share of rows whose 6 variants don't all agree (using the per-variant mean over P).
3. **Cross-fitted best-of-6** (the headline number). For each row and each held-out perturbation h: choose `v* = argmax_v mean_{p≠h} success(v,p)` (break ties at random), then record `success(v*, h)`. Headroom = mean(best-of-6 cross-fit) − single-sample rate. Get a 95% CI by **bootstrapping over rows** (2,000 resamples).
4. **Stage split**: the variance of the per-mask mean (mask stage) vs the within-mask variance (appearance stage). Also the cross-fitted best mask, choosing on the mask average. This tells you whether Step 2 or Step 3 matters.
5. **Mask diversity**: the mean pairwise IoU of the 3 M_f per row.
6. The same numbers broken down per task, to see where the headroom sits.

Also report: the **oracle (non-cross-fitted) best-of-6**, so the optimism gap is visible, and how often the generated goals **beat GT** on the same rows.

### 1.6 Gate 0 decision rule

| Result | Decision |
|---|---|
| Cross-fitted headroom **≥ +5 pts** with CI > 0, **and ≥ 30%** mixed rows | **GO**: Variant A has a signal → Phase 2A. |
| Headroom +2 to +5 pts | **Weak**: Phase 2A only as a small add-on; lead the paper with Variant C. |
| Headroom < +2 pts, or masks collapsed | **NO-GO on GR-MG**: drop A on GR-MG → Phase 2C (policy adaptation), or repeat Gate 0 on GHIL-Glue (GT 39.0 vs SGE 29.2, more room). |

Whatever the outcome, Gate 0 is a reportable result: "how much does sample choice matter for a frozen policy".

### 1.7 Run order (copy-paste)

**One-time setup:**
```bash
# The FLUX.1-Kontext-dev weights are gated: create the secret the scripts expect (once)
modal secret create huggingface-secret HF_TOKEN=hf_xxx
# The LoRAs must be on rm-policy-ckpts (as /ckpts/ckpts/*.safetensors or /ckpts/*.safetensors)
modal volume ls rm-policy-ckpts ckpts
```

**Generation (L40S):**
```bash
cd generator
modal run modal_gate0_generate.py::prefetch          # CPU only, ~10 min, caches ~35 GB of weights
modal run modal_gate0_generate.py --limit 5          # pilot: prints s/pass for Step 2 and Step 3
modal volume get rm-policy-eval-data gate0 ../results/gate0_pilot_imgs   # look at them!
modal run --detach modal_gate0_generate.py --num-shards 2
```
Before the full run, check the pilot images: the masks should be blob-shaped, the goals should show the instructed object moved, and the 3 masks per row should not be identical. Re-running skips rows that are already finished, so an interrupted run can simply be restarted.

**Rollouts (T4):**
```bash
cd ../policy
modal run modal_calvin_eval.py::run_gate0 --num-shards 1 --limit 5       # pilot: 105 rollouts
modal volume get rm-policy-runs eval_method/gate0 ../results/gate0_pilot
python ../variant_a_alignment/gate0_analyze.py --results ../results/gate0_pilot
#   check "(row,variant) flipped by perturbation" > 0; if 0%, rerun with --perturb-std 0.05
modal run --detach modal_calvin_eval.py::run_gate0 --num-shards 8          # full: 3,150 rollouts
```

The full run overwrites the pilot's `per_sample_shard00.jsonl`, so nothing gets counted twice. If you want to keep the pilot output, pass `--run-tag gate0_pilot` to the pilot command.

**Analysis (local, free):**
```bash
cd ..
modal volume get rm-policy-runs eval_method/gate0 results/
modal volume get rm-policy-eval-data gate0/masks results/gate0_masks
python variant_a_alignment/gate0_analyze.py --results results/gate0 --masks results/gate0_masks
```
The script prints the headroom with CIs, the stage split, mask IoU, a per-task table and a **VERDICT** line using the §1.6 rule.

---

### 1.8 Running the rollouts on the college A6000 (free)

CALVIN rollouts are CPU-bound (PyBullet) and GR-MG uses only a few GB of VRAM, so they make up about $13 of the $15 full run but need no big GPU. One A6000 can run about 8 eval workers in parallel; CPU cores are the real limit, so use `--workers ≈ cores / 2`. Generation stays on Modal (L40S, about $1.5, already proven). The A6000 could run it too, but that's a second environment to set up for $1.5.

The server needs only about **1.4 GB**: a 59 MB CALVIN subset (env config + the 214 start frames), the 1.3 GB GR-MG checkpoint and the test parquet. `export_calvin_subset` has already written the subset to `rm-policy-runs:/calvin_subset_task_D_D.tar.gz`.

```bash
# --- on the server, once ---
cd ~/SGE-Goal-Cleaned/sge_followup_variant_a && git pull
bash variant_a_alignment/setup_local_env.sh ~/calvin        # conda env 'gate0', ~15 min
conda activate gate0
python -m modal token new                                    # if this machine isn't logged in yet

mkdir -p data/ckpts/gr_mg data/eval_data
python -m modal volume get rm-policy-runs /calvin_subset_task_D_D.tar.gz data/
tar -xzf data/calvin_subset_task_D_D.tar.gz -C data/
python -m modal volume get rm-policy-ckpts "/gr_mg/epoch=47-step=83712.ckpt" data/ckpts/gr_mg/
python -m modal volume get rm-policy-hf-cache /calvin_test.parquet data/
python -m modal volume get rm-policy-eval-data /gate0 data/eval_data/

# --- sanity check: re-run the Modal pilot rows; results must match Modal exactly ---
python variant_a_alignment/run_gate0_local.py --calvin_repo ~/calvin --rows 8,10,21,27,33 \
    --num_perturb 3 --ep_len 360 --workers 3 --out results/gate0_local_check
python variant_a_alignment/gate0_analyze.py --results results/gate0_local_check
#   expected: single 78.9%, GT 100%, headroom +19.8 (same as the Modal pilot)

# --- full run (after the 100-row generation has finished on Modal) ---
python -m modal volume get rm-policy-eval-data /gate0 data/eval_data/      # re-pull new goals
python variant_a_alignment/run_gate0_local.py --calvin_repo ~/calvin --workers 8
python variant_a_alignment/gate0_analyze.py --results results/gate0 --masks data/eval_data/gate0/masks
```

Run the full job inside `tmux` or `screen` so it survives an SSH disconnect. Each worker logs to `results/gate0/logs/shardNN.log`.

## Phase 2A: Real Variant A (only if Gate 0 = GO)

**Budget note:** this doesn't fit in the remaining credit. Use next month's refresh or the lab GPUs.

Do this in order of cheapness. **Skip L_beh at first**: backpropagating through a 12B FLUX + VAE + GR-MG is the most expensive and riskiest part.

1. **A-select (no training).** Train a small success predictor (e.g. a ResNet-18 on `[I_o, goal]` + task id) on the Gate 0 rollouts. At test time, generate N=6 and pick the highest-scoring one. CPU/T4, ~$1. This is also the proposal's A-select baseline.
2. **Flow-RWR / Flow-DPO on real data.** Gate 0 has already produced real preference pairs (with the same mask, different appearance → Step 3 pairs; across masks → Step 2 pairs). Rewrite `train_a_flow_dpo.py` to:
   - load the real FLUX Kontext transformer + Step 3 LoRA (trainable) and a frozen copy as the reference (or disable the adapter for the reference pass),
   - VAE-encode the winner/loser goal images, sample σ, use the same conditioning as training, and compute ℓ_θ − ℓ_ref per Eq. 5 of the proposal,
   - keep `L_edit` on dataset tuples as an anchor (otherwise the model forgets).
   Reuse the model-loading and loss code from `generator/flux_kontext_lora_finetuning.py`, not the toy `VelocityPredictor`.
   H100, about 1–2 h, ~$5–9.
3. **Evaluate on held-out rows** (not the Gate 0 rows). Single sample from the DPO-tuned LoRA vs the original LoRA, both on the same ~300 held-out rows × 3 perturbations. ~$3–4.
4. **L_beh (dense)**, only if 1–3 show gains. Rewrite `train_a_dense.py` to use the real FLUX + LoRA, the one-step x̂₀ at σ < 0.35, a differentiable VAE decode, and a resize to 200. Load the real GR-MG from `policy/gr_mg_policy/model/` with the checkpoint's `state_dict`. Use the proposal's loss `D(π(o, Î), π(o, I_e))`, which needs no action labels. GR-MG also needs the gripper camera and state history, so load them from `episode_<start>.npz` using `hf_calvin_mapping.json`.

## Phase 2C: Policy adaptation (if Gate 0 = NO-GO)

- Fine-tune GR-MG on single end-goal conditioning (π_IM) using `policy/gr_mg_policy/main.py`, with goals mixed from GT frames and SGE-Goal outputs.
- Evaluate SGE-Goal and GT goals under π_IM using the same `run_method` harness.
- Lab GPUs strongly preferred: policy fine-tuning is many GPU-hours.

---

## 2. Budget tracker

| Step | GPU | Est. cost | Cumulative |
|---|---|---|---|
| 0.3 GT baseline (skip if it already exists) | 5× T4 | $0–1.6 | $1.6 |
| 1.7 Prefetch weights | CPU | ~$0.1 | $1.7 |
| 1.2 Generation pilot (5 rows) | L40S | ~$0.3 (mostly model load) | $2.0 |
| 1.3–1.4 Rollout pilot (5 rows) | T4 | $0.2 | $2.2 |
| 1.2 Generation full (150 rows, 1,350 passes at 256 px / 10 steps) | 2× L40S | ~$1.5–3 | $5 |
| 1.4 Rollouts full (3,150) | 8× T4 | $4–5 | $10 |
| Re-runs / debugging buffer | — | ~$5 | **~$15** |

**Measured in the pilot (2026-10-15):** generation is about 2.0–2.1 s per FLUX pass on L40S. Rollouts take about 8–40 s on success, but a failure runs the whole episode (about 2 min at `ep_len=360`). That works out to about 35–40 s per rollout on average, so 2,900 rollouts would cost about $27, **over budget**. Revised full run: **100 rows** (mislabelled rows excluded) × 7 goals × **2 start states**, at **`--ep-len 240`** (keeps about 98% of successes). That's about 1,400 rollouts, roughly $13–14, plus about $1.5 for generation. The full 1087-row GT baseline costs about $8, not $1.5, so skip it this month: Gate 0 already includes GT on the same rows.

With generation this cheap, the rollouts dominate. If the pilot projects well under budget, raise rows to 200 or perturbations to 4 rather than adding variants.

If either pilot projects more than these numbers, **cut rows to 100** rather than cutting perturbations. Perturbations are what make the cross-fit valid.

---

## 3. Files at a glance

| File | Action |
|---|---|
| `variant_a_alignment/train_a_dense.py` | Add the mock guard (Phase 0). Rewrite in Phase 2A step 4. |
| `variant_a_alignment/factorial_rollouts.py` | Add the mock guard. Superseded by the Gate 0 scripts. |
| `variant_a_alignment/train_a_flow_dpo.py` | Add the mock guard. Rewrite in Phase 2A step 2. |
| `README.md` | Mark §5.2–5.5 as not implemented. |
| `variant_a_alignment/gate0_select_rows.py` | **New**: stratified row sample. |
| `generator/modal_gate0_generate.py` | **New**: L40S, K_m × K_a goals per row → eval volume. |
| `policy/gr_mg_evaluate/eval_method.py` | **Edit**: `--rows_json`, `--variants`, `--num_perturb`, `--perturb_std`. |
| `policy/modal_calvin_eval.py` | **Edit**: pass the new arguments through and add a `run_gate0` entrypoint. Optionally cpu=2 / 16 GiB. |
| `variant_a_alignment/gate0_analyze.py` | **New**: cross-fitted headroom, mixed fraction, stage split, mask IoU, bootstrap CIs. |
