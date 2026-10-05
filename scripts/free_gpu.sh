#!/bin/sh
# Usage: GPU=$(sh scripts_free_gpu.sh) || exit 1   -> prints one GPU index whose memory.used < 500 MiB, else fails.
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 500 {print $1; found=1; exit} END {if (!found) exit 1}'
