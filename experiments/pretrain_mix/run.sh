#!/bin/bash

set -euo pipefail

cd "${HOME}/fmri-fm/experiments/pretrain_mix"

taskid=2

for ii in {0..3}; do
    bash launch_eval.sh $ii $taskid
done
