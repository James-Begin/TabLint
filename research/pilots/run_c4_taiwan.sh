#!/bin/bash
cd ~/forensics
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s nohup uv run python c4_audit.py taiwan 500 60 $s 4 100 0 \
    > logs/c4_taiwan_$s.log 2>&1 < /dev/null &
done
