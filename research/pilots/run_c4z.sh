#!/bin/bash
# C4 variant z (standardised surrogate inputs + masked non-finite gradient entries).
cd ~/forensics
launch() { CUDA_VISIBLE_DEVICES=$1 nohup uv run python c4_audit.py $2 500 60 $3 $4 100 0 z \
  > logs/c4z_$2_$3.log 2>&1 < /dev/null & }
launch 0 credit-g 0 3; launch 1 credit-g 1 3; launch 2 credit-g 2 3
launch 3 diabetes 0 2; launch 4 diabetes 1 2
launch 5 taiwan 0 3;   launch 6 taiwan 1 3;   launch 7 taiwan 2 3
