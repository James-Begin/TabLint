#!/bin/bash
# Detection on confirmation reports: one sequential process per GPU, 8 shards.
cd ~/forensics
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. nohup timeout 3600 uv run python experiments/detect_poison.py \
    results/confirmation --shard $g --nshards 8 > logs/detect_$g.log 2>&1 < /dev/null &
done
