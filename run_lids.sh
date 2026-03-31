#!/bin/bash
cd "$(dirname "$0")"
OUTPUT_DIR="output"

echo "=== Fixing None eval entries (marking as fail) ==="
python fix_none_evals.py

echo ""
echo "=== Merging qwen25_7b_lids results ==="
python -m eval_system_bench \
    --merge_only \
    --infer_model_name qwen25_7b_lids \
    --output_dir ${OUTPUT_DIR} \
    --world_size 4
