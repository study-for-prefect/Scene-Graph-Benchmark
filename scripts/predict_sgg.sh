#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/wxm/miniconda3/envs/scene_graph_benchmark/bin/python}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

INPUT_DIR="${1:-input_dir}"
OUTPUT_DIR="${2:-custom_output_sgdet/predictions/$(basename "$INPUT_DIR")}"
BOX_THRESH="${BOX_THRESH:-0.5}"
REL_THRESH="${REL_THRESH:-0.5}"
TOP_RELS="${TOP_RELS:-12}"

"$PYTHON_BIN" tools/relation_test_net.py \
  --config-file configs/sgdet_X_101_train.yaml \
  TEST.CUSTUM_EVAL True \
  TEST.CUSTUM_PATH "$INPUT_DIR" \
  DETECTED_SGG_DIR "$OUTPUT_DIR" \
  TEST.IMS_PER_BATCH 1

"$PYTHON_BIN" tools/visualize_custom_sgg.py \
  --prediction-json "$OUTPUT_DIR/custom_prediction.json" \
  --info-json "$OUTPUT_DIR/custom_data_info.json" \
  --output-dir "$OUTPUT_DIR/vis" \
  --box-thresh "$BOX_THRESH" \
  --rel-thresh "$REL_THRESH" \
  --top-rels "$TOP_RELS"

echo "JSON: $OUTPUT_DIR/custom_prediction.json"
echo "Readable JSON: $OUTPUT_DIR/vis/readable_predictions.json"
echo "Visualization: $OUTPUT_DIR/vis"
