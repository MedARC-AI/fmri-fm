#!/bin/bash

if [[ -z $1 || $1 == "-h" || $1 == "--help" ]]; then
    echo "launch_eval.sh JOBID TASKID"
    exit
fi

JOBID=$1
TASKID=$2

export OMP_NUM_THREADS=8

ROOT="${HOME}/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="pretrain_mix"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/checkpoints"

# fill with the name of your home folder on lightning
SHARE_USER=${SHARE_USER:-volunteer}
SHARE_DIR="/teamspace/gcs_folders/share/fmri-fm/${SHARE_USER}"

# save output to persistent shared storage
SHARE_OUT_DIR="${SHARE_DIR}/${OUT_DIR}"
mkdir -p ${SHARE_OUT_DIR} 2>/dev/null
ln -sn ${SHARE_OUT_DIR} ${OUT_DIR} 2>/dev/null

keys=(
    hcp_ukbb_n1800
    hcp_ukbb_n3600
)
key=${keys[JOBID]}

tasks=(
    hcp_task
    nsd_clip_class
    ukbb_sex
)
task=${tasks[TASKID]}

name="${EXP_NAME}/${key}/${task}_eval"
config="${EXP_DIR}/config/${task}_eval.yaml"

notes="${task} eval for hcp ukbb mix pretraining run"

ckpt="${EXP_DIR}/checkpoints/${EXP_NAME}/${key}/pretrain/checkpoint-last.pth"

echo uv run torchrun --standalone --nproc_per_node=1 \
    src/flat_mae/main_probe.py \
    --cfg-path "${config}" \
    --overrides \
    name="${name}" \
    notes="${notes}" \
    pretrain_ckpt="${ckpt}" \
    output_dir="${OUT_DIR}"
