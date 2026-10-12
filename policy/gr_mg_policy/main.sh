#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

GPUS_PER_NODE="${GPUS_PER_NODE:-1}" # number of gpus per machine
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1:29500}" # master address and port
NNODES="${NNODES:-1}" # number of machines
JOB_ID="${JOB_ID:-107}"

CONFIG_FILE="${1:-policy/config/train.json}"

torchrun \
    --nproc_per_node "$GPUS_PER_NODE" \
    --nnodes "$NNODES" \
    --node_rank 0 \
    --rdzv_endpoint "$MASTER_ADDR" \
    --rdzv_id "$JOB_ID" \
    --rdzv_backend c10d \
    policy/main.py \
    --config "$CONFIG_FILE" \
    --devices "$GPUS_PER_NODE" \
    --num_nodes "$NNODES"