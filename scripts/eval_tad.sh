#!/usr/bin/env bash
# Evaluate a trained checkpoint and dump visualizations.
# Usage: bash scripts/eval_tad.sh <checkpoint> [data_root] [output_dir]
set -euo pipefail

CHECKPOINT=${1:?usage: bash scripts/eval_tad.sh <checkpoint> [data_root] [output_dir]}
DATA_ROOT=${2:-data/tad}
OUTPUT_DIR=${3:-outputs/tad_eval}

python pre.py \
  --config_file configs/DINO/custom_dino.py \
  --dataset_file tad \
  --coco_path "${DATA_ROOT}" \
  --resume "${CHECKPOINT}" \
  --eval \
  --output_dir "${OUTPUT_DIR}" \
  --visualization_dir "${OUTPUT_DIR}/visualizations"
