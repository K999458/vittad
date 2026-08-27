#!/usr/bin/env bash
# Train the TAD detector. Usage: bash scripts/train_tad.sh [data_root] [output_dir]
set -euo pipefail

DATA_ROOT=${1:-data/tad}
OUTPUT_DIR=${2:-outputs/tad}

python main.py \
  --config_file configs/DINO/custom_dino.py \
  --dataset_file tad \
  --coco_path "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}"
