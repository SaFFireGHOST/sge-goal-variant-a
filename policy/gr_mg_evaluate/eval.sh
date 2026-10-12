#!/usr/bin/env bash
# Evaluate GR-MG on CALVIN.
#
# Two evaluation modes are supported:
#
#   1. ip2p     — original GR-MG behaviour (goal images generated on the fly
#                 by InstructPix2Pix, re-planned every 20 env steps).
#   2. manifest — user-supplied final goal images, one per test sample,
#                 held fixed for the entire subtask rollout. This is the mode
#                 to use for the "single final-goal image" study.
#
# Edit the variables below, then run:
#   bash evaluate/eval.sh policy/config/train.json
#
# The first positional arg ($1) is the policy config JSON. Any extra args are
# ignored (kept for backward compat with the original launch script).

set -euo pipefail

cd "$(dirname "$0")/.."

# -----------------------------------------------------------------------------
# Required user paths — EDIT THESE.
# -----------------------------------------------------------------------------
# Path to the CALVIN dataset split. Use task_D_D for the CALVIN-D test set.
export DATASET_PATH="${DATASET_PATH:-/PATH/TO/calvin/dataset/task_D_D}"

# Directory containing the GR-MG policy ckpt (e.g. epoch=47-step=83712.ckpt).
export CKPT_DIR="${CKPT_DIR:-/PATH/TO/gr_mg_policy_ckpt_dir}"

# Which checkpoint epoch to load.
export EPOCHS=("${EPOCHS[@]:-47}")

# -----------------------------------------------------------------------------
# Goal-image source.
#   GOAL_IMAGE_SOURCE=ip2p     -> use the IP2P goal generator.
#                                 Requires SD_CKPT to be set.
#   GOAL_IMAGE_SOURCE=manifest -> use a user-supplied JSON manifest of goal
#                                 images. Requires GOAL_IMAGES_MANIFEST to be set.
# -----------------------------------------------------------------------------
export GOAL_IMAGE_SOURCE="${GOAL_IMAGE_SOURCE:-manifest}"
export SD_CKPT="${SD_CKPT:-/PATH/TO/goal_gen.ckpt}"
export GOAL_IMAGES_MANIFEST="${GOAL_IMAGES_MANIFEST:-/PATH/TO/goal_images_manifest.json}"

# Tag appended to the output dir, lets you keep multiple goal-image sources
# side-by-side (e.g. "gt_final", "ip2p_edited", "magicbrush_edited").
export RUN_TAG="${RUN_TAG:-gt_final}"

# Number of long-horizon CALVIN test sequences (1000 = official LH-MTLC count).
export NUM_SEQUENCES="${NUM_SEQUENCES:-1000}"

export MESA_GL_VERSION_OVERRIDE=3.3

CONFIG_PATH="${1:?Usage: bash evaluate/eval.sh <policy_config.json>}"

echo "[eval.sh] DATASET_PATH       = ${DATASET_PATH}"
echo "[eval.sh] CKPT_DIR           = ${CKPT_DIR}"
echo "[eval.sh] EPOCHS             = ${EPOCHS[*]}"
echo "[eval.sh] GOAL_IMAGE_SOURCE  = ${GOAL_IMAGE_SOURCE}"
echo "[eval.sh] RUN_TAG            = ${RUN_TAG}"
echo "[eval.sh] NUM_SEQUENCES      = ${NUM_SEQUENCES}"
echo "[eval.sh] CONFIG_PATH        = ${CONFIG_PATH}"

EXTRA_ARGS=()
if [[ "${GOAL_IMAGE_SOURCE}" == "ip2p" ]]; then
    EXTRA_ARGS+=(--ip2p_ckpt_path "${SD_CKPT}")
elif [[ "${GOAL_IMAGE_SOURCE}" == "manifest" ]]; then
    EXTRA_ARGS+=(--goal_images_manifest "${GOAL_IMAGES_MANIFEST}")
else
    echo "Unknown GOAL_IMAGE_SOURCE='${GOAL_IMAGE_SOURCE}'" >&2
    exit 2
fi

COUNTER=-1
for epoch in "${EPOCHS[@]}"; do
    COUNTER=$((COUNTER + 1))
    export CUDA_VISIBLE_DEVICES="${COUNTER}"
    python3 evaluate/eval.py \
        --dataset_path "${DATASET_PATH}" \
        --ckpt_dir "${CKPT_DIR}" \
        --epoch "${epoch}" \
        --goal_image_source "${GOAL_IMAGE_SOURCE}" \
        --num_sequences "${NUM_SEQUENCES}" \
        --run_tag "${RUN_TAG}" \
        --config_path "${CONFIG_PATH}" \
        "${EXTRA_ARGS[@]}" &
done
wait
