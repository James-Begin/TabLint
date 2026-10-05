#!/bin/bash
# Full C4 run: 60 targets/dataset, 500 context rows, 100 optimisation steps.
cd ~/forensics
mkdir -p logs
launch() { # gpu dataset shard nshards
  CUDA_VISIBLE_DEVICES=$1 nohup uv run python c4_audit.py $2 500 60 $3 $4 100 0 \
    > logs/c4_$2_$3.log 2>&1 < /dev/null &
}
launch 0 credit-g 0 4; launch 1 credit-g 1 4; launch 2 credit-g 2 4; launch 3 credit-g 3 4
launch 4 diabetes 0 2; launch 5 diabetes 1 2
launch 6 taiwan 0 2;   launch 7 taiwan 1 2
