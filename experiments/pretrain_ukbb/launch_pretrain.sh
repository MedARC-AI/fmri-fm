#!/bin/bash

export OMP_NUM_THREADS=8

ROOT="${HOME}/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="pretrain_ukbb"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/checkpoints"

# save output to persistent shared storage
SHARE_DIR="/teamspace/gcs_folders/share/fmri-fm/connor"
SHARE_OUT_DIR="${SHARE_DIR}/${OUT_DIR}"
mkdir -p ${SHARE_OUT_DIR} 2>/dev/null
ln -s ${SHARE_OUT_DIR} ${OUT_DIR} 2>/dev/null

name="${EXP_NAME}/01_n1800/pretrain"

uv run torchrun --standalone --nproc_per_node=1 \
    src/flat_mae/main_pretrain.py \
    --cfg-path "${EXP_DIR}/pretrain.yaml" \
    --overrides \
    name="${name}" \
    notes="pretrain on 1800 ukbb shards; stream from r2" \
    output_dir="${EXP_DIR}/checkpoints"
