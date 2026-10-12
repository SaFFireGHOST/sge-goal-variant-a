# Local extensions: final-goal-image evaluation on CALVIN-D

This fork of [bytedance/GR-MG](https://github.com/bytedance/GR-MG) adds two
things on top of the upstream code, leaving the model and training pipeline
untouched:

1. **User-supplied final-goal-image evaluation.** A new
   `--goal_image_source manifest` mode in [evaluate/eval.py](evaluate/eval.py)
   lets you feed one image per test sample as the *fixed* final goal image,
   reused at every env step of that subtask (no sub-goal re-planning).
2. **Per-sample visualisation.** [evaluate/visualize.py](evaluate/visualize.py)
   renders an mp4 of the CALVIN simulation for any chosen test sample given a
   final goal image.

The original IP2P-based behaviour is still available via
`--goal_image_source ip2p`.

## Files added / changed

- [evaluate/goal_image_provider.py](evaluate/goal_image_provider.py) — new.
  Goal-image providers: `IP2PGoalImageProvider`, `ManifestGoalImageProvider`,
  `StaticGoalImageProvider`.
- [evaluate/eval.py](evaluate/eval.py) — refactored CLI; new flags
  `--goal_image_source`, `--goal_images_manifest`, `--num_sequences`,
  `--run_tag`, `--dataset_path` (now required and explicit).
- [evaluate/eval.sh](evaluate/eval.sh) — driver script with `DATASET_PATH`,
  `GOAL_IMAGE_SOURCE`, `GOAL_IMAGES_MANIFEST`, `RUN_TAG`, `NUM_SEQUENCES`
  environment variables. Defaults point at CALVIN-D (`task_D_D`).
- [evaluate/visualize.py](evaluate/visualize.py) — new. Single-sample mp4 renderer.
- [evaluate/example_goal_images_manifest.json](evaluate/example_goal_images_manifest.json) — manifest schema example.

## Test-sample identification

CALVIN's official long-horizon evaluation enumerates 1000 test sequences via
`calvin_agent.evaluation.multistep_sequences.get_sequences(num_sequences)`.
Each sequence is a chain of (typically 5) language-instructed subtasks. We
identify a single test sample by the pair `(sequence_idx, subtask_idx)`.

This is the same enumeration used by both `eval.py` and `visualize.py`, so a
manifest entry `"3_0"` and the visualisation invocation
`--sequence_idx 3 --subtask_idx 0` refer to *exactly* the same test sample
(provided you keep `--num_sequences` consistent).

## Goal-image manifest formats

The manifest is JSON. Two shapes are accepted by `ManifestGoalImageProvider`
(see [evaluate/goal_image_provider.py](evaluate/goal_image_provider.py)):

```jsonc
// 1. Keyed by "<sequence_idx>_<subtask_idx>"
{
  "0_0": "goals/seq0_sub0.png",
  "0_1": "goals/seq0_sub1.png",
  ...
}
```

```jsonc
// 2. Flat list, indexed as sequence_idx * subtasks_per_sequence + subtask_idx
{
  "subtasks_per_sequence": 5,
  "images": [
    "goals/seq0_sub0.png",
    "goals/seq0_sub1.png",
    ...
  ]
}
```

Paths can be absolute or relative to the manifest file. Images of any size
are auto-resized to 200×200 (CALVIN's static-camera shape).

## Evaluating GR-MG on CALVIN-D with user-supplied goal images

```bash
export DATASET_PATH=/data/calvin/task_D_D
export CKPT_DIR=/ckpts/gr_mg                    # holds epoch=47-step=83712.ckpt
export EPOCHS=(47)
export GOAL_IMAGE_SOURCE=manifest
export GOAL_IMAGES_MANIFEST=/runs/gt_final/manifest.json
export RUN_TAG=gt_final                         # or ip2p_edited / magicbrush / ...
export NUM_SEQUENCES=1000

bash evaluate/eval.sh policy/config/train.json
```

Results land in `${CKPT_DIR}/task_D_D_manifest_47_epoch_gt_final/`
(`success_rate.txt`, `results.json`). Re-run with a different
`GOAL_IMAGES_MANIFEST` and `RUN_TAG` to compare goal-image sources side by side.

## Visualising one test sample

```bash
python evaluate/visualize.py \
    --dataset_path  /data/calvin/task_D_D \
    --ckpt_dir      /ckpts/gr_mg \
    --epoch         47 \
    --config_path   policy/config/train.json \
    --sequence_idx  3 \
    --subtask_idx   0 \
    --num_sequences 1000 \
    --goal_image    /runs/gt_final/goals/seq3_sub0.png \
    --output        /tmp/seq3_sub0.mp4
```

The mp4 shows three side-by-side panels per frame: static camera, gripper
camera, and the fixed final goal image. Rollout stops as soon as the CALVIN
task oracle reports success (toggle with `--no_stop_on_success`).

`imageio` (with `imageio-ffmpeg`) is preferred for mp4 writing; falls back to
`opencv-python`, then to per-frame PNG dumps.

## Notes / caveats

- GR-MG was trained with progress-conditioned goal images that often depict
  intermediate (not final) configurations. Feeding a fixed final goal image
  for the entire rollout is a slight distribution shift — but it is well
  defined, respects the "single final-goal image" constraint, and stays
  comparable across all goal-image sources (ground-truth final frame, edited
  images from any image-editing model, etc.).
- The `task_D_D` split is the CALVIN-D environment. The same flags also work
  for `task_ABC_D` and `task_ABCD_D`; just point `DATASET_PATH` accordingly.
- For `visualize.py` with `subtask_idx > 0`, the env is reset to the *sequence*
  initial state rather than the chained end-state of the previous subtask. To
  reproduce the exact chained start state, run all preceding subtasks first
  (not done here to keep visualisation cheap).

## Evaluating on the Hugging Face dataset

The companion HF dataset
[`SuyashAjitChavan2002/Calvin-Dataset-Final`](https://huggingface.co/datasets/SuyashAjitChavan2002/Calvin-Dataset-Final)
ships 1087 test rows derived from `task_D_D/validation`, each row being:

| HF column | meaning |
|---|---|
| `original_image` | rgb_static at the segment's start frame (the "before" scene) |
| `edited_image`   | rgb_static at the segment's end frame (the **final goal image**) |
| `edit_instruction` | natural-language instruction (paraphrase) |
| `id` | author key, format `episode_<E>_atomic_<A>` (no direct CALVIN frame info) |

To evaluate GR-MG on these rows we first build a **1-to-1 mapping** from each
HF row to its source CALVIN-D segment by hashing `rgb_static` at every val
segment's start/end frame and matching against the HF images.

### Step 1 — build the mapping

```bash
python evaluate/build_hf_calvin_mapping.py \
    --calvin_val_dir /data/calvin/task_D_D/validation \
    --hf_parquet     /tmp/calvin_test.parquet \
    --output         /runs/hf_calvin_mapping.json
```

Omit `--hf_parquet` to download it from the HF hub into `~/.cache/calvin_hf/`.
Output JSON contains, per row, `start_frame`, `end_frame`, canonical CALVIN
`task` name, official `calvin_ann`, and the `match_method` (`"exact"` for an
md5 hit, `"nn_l2"` for the JPEG-tolerant fallback).

### Step 2 — run evaluation

```bash
python evaluate/eval_hf.py \
    --dataset_path  /data/calvin/task_D_D \
    --ckpt_dir      /ckpts/gr_mg \
    --epoch         47 \
    --config_path   policy/config/train.json \
    --mapping_json  /runs/hf_calvin_mapping.json \
    --hf_parquet    /tmp/calvin_test.parquet \
    --output_dir    /runs/eval_hf_gt_final
```

For each matched HF row the env is reset from `episode_<start_frame>.npz`,
the policy is conditioned on the HF `edited_image` (the final goal) and the
HF `edit_instruction`, and success is decided by the official CALVIN task
oracle on the recovered task name.

Outputs in `--output_dir`:

- `success_rate.txt` — running success rate, one line per sample
- `per_sample.jsonl` — full per-row record
- `summary.json` — overall and per-task success rates

To compare goal-image sources (ground-truth `edited_image` vs. images produced
by an image-editing model), produce a *modified* parquet with the same `id`s
and replaced `edited_image` payloads, point `--hf_parquet` at it, and re-run
Step 2 with a different `--output_dir`. Step 1 only needs to be re-run if the
*original* images change.

To use the canonical CALVIN annotation instead of the HF paraphrase, pass
`--instruction_source calvin`.

## Running on Modal (cloud GPU)

The local GPU on this machine is shared. Use [`modal_app.py`](modal_app.py) to
run the same scripts on Modal-managed GPUs. Modal logic only — the underlying
eval scripts in `evaluate/` are unchanged.

### One-time setup (on whichever machine drives Modal)

```bash
pip install modal

# Authenticate. These credentials live in ~/.modal.toml; do this on a machine
# whose home dir you trust.
modal token set --token-id <YOUR_TOKEN_ID> --token-secret <YOUR_TOKEN_SECRET>
```

The Modal app declares four persistent volumes (created on first use):

| Volume | Purpose | Container mount |
|---|---|---|
| `rm-policy-calvin-data` | CALVIN-D dataset (`task_D_D/`) | `/data` |
| `rm-policy-ckpts` | GR-MG / IP2P / MAE checkpoints | `/ckpts` |
| `rm-policy-runs` | mapping JSON, eval outputs, mp4s, custom parquets | `/runs` |
| `rm-policy-hf-cache` | cached HF parquet | `/hf_cache` |

### Step-by-step

1. **Diagnose the image** (verifies CUDA, CALVIN imports, volumes):

   ```bash
   modal run modal_app.py::setup_check
   ```

2. **Upload CALVIN-D** (once, ~150 GB depending on split):

   ```bash
   modal run modal_app.py::upload_calvin_split \
       --local-path /path/to/calvin/dataset/task_D_D \
       --split-name task_D_D
   ```

3. **Upload checkpoints** (once each):

   ```bash
   # GR-MG policy
   modal run modal_app.py::upload_ckpt \
       --local-path ./epoch=47-step=83712.ckpt \
       --remote-name gr_mg/epoch=47-step=83712.ckpt

   # (Optional) IP2P goal-image generator, only if you'll use --goal-image-source ip2p
   modal run modal_app.py::upload_ckpt \
       --local-path ./goal_gen.ckpt \
       --remote-name goal_gen.ckpt
   ```

4. **Pre-fetch the HF parquet** into the cache volume (small, fast):

   ```bash
   modal run modal_app.py::download_hf_parquet
   ```

5. **Build the HF↔CALVIN mapping** (CPU only, no GPU needed):

   ```bash
   modal run modal_app.py::build_mapping --split-name task_D_D
   # produces rm-policy-runs:/hf_calvin_mapping.json
   ```

6. **Evaluate on the 1087 HF rows** with the HF `edited_image` as the goal:

   ```bash
   modal run modal_app.py::eval_hf \
       --split-name task_D_D \
       --ckpt-remote gr_mg/epoch=47-step=83712.ckpt \
       --epoch 47 \
       --run-tag gt_final
   # results in rm-policy-runs:/eval_hf_task_D_D_gt_final_epoch47/
   ```

7. **Compare a different goal-image source** (e.g. images produced by an
   image-editing model). Build a parquet with the same `id`s but with the
   `edited_image` column replaced, upload it to the runs volume, then:

   ```bash
   modal volume put rm-policy-runs ./my_edited.parquet edited/my_edited.parquet
   modal run modal_app.py::eval_hf \
       --custom-parquet-remote edited/my_edited.parquet \
       --run-tag magicbrush
   ```

8. **(Optional) Standard CALVIN LH-MTLC eval** with a manifest of goal images:

   ```bash
   modal volume put rm-policy-runs ./manifest.json manifest.json
   modal volume put rm-policy-runs ./goals goals    # directory of goal pngs
   modal run modal_app.py::eval_calvin \
       --split-name task_D_D \
       --goal-image-source manifest \
       --goal-images-manifest-remote manifest.json
   ```

9. **Visualise one test sample** and download the mp4 to your local box:

   ```bash
   modal run modal_app.py::visualize \
       --goal-image ./my_goal.png \
       --output     ./seq3_sub0.mp4 \
       --sequence-idx 3 --subtask-idx 0
   ```

### Inspecting / fetching results

```bash
# List what's on the runs volume.
modal volume ls rm-policy-runs

# Pull an entire output directory back locally.
modal volume get rm-policy-runs eval_hf_task_D_D_gt_final_epoch47 ./eval_hf_gt_final
```

### Notes

- All eval functions request `gpu="A10G"`. Bump to `"A100"` or
  `"A100-80GB"` in `modal_app.py` if your account allows and you want the
  ~12-hour 1087-row eval to finish faster.
- `build_mapping` runs on CPU only (no GPU billing) since it's pure numpy/md5.
- The Modal image installs CALVIN with submodules at build time, so the
  first run will be slow. Subsequent runs reuse the cached image layers.


