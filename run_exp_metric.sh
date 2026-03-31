#!/bin/bash
cd "$(dirname "$0")"
output="./output"

echo "=== Metrics: qwen25_7b_vllm ==="
python plot/eval_output.py \
    --infer_model_name qwen25_7b_vllm \
    --output_dir ${output}

echo ""
echo "=== Metrics: qwen25_7b_lids ==="
python plot/eval_output.py \
    --infer_model_name qwen25_7b_lids \
    --output_dir ${output}
