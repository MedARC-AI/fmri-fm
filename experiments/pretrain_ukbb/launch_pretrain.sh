#!/bin/bash

export OMP_NUM_THREADS=8

ROOT="${HOME}/fmri-fm"
cd $ROOT

set -a
source .env
set +a

EXP_NAME="pretrain_ukbb"
EXP_DIR="experiments/${EXP_NAME}"

name="${EXP_NAME}/01_n1800/pretrain"

uv run torchrun --standalone --nproc_per_node=1 \
    src/flat_mae/main_pretrain.py \
    --cfg-path "${EXP_DIR}/pretrain.yaml" \
    --overrides \
    name="${name}" \
    notes="pretrain on 1800 ukbb shards; stream from r2" \
    output_dir="${EXP_DIR}/checkpoints"
