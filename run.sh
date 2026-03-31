#!/bin/bash
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=2,3,4,5
OUTPUT_DIR="output"
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
LIDS_WORLD_SIZE=${#GPUS[@]}

echo "=== Evaluating qwen25_7b_vllm (verifier: gpt4o, max_threads=20) ==="
echo "Ensure vLLM server is running: bash servers/run_vllm_qwen25_7b.sh"
python -m eval_system_bench \
    --infer_model_name qwen25_7b_vllm \
    --output_dir ${OUTPUT_DIR} \
    --max_threads 20

echo ""
echo "=== Evaluating qwen25_7b_lids (verifier: gpt4o, ${LIDS_WORLD_SIZE} processes in parallel) ==="
for rank in $(seq 0 $((LIDS_WORLD_SIZE - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$rank]} python -m eval_system_bench \
        --infer_model_name qwen25_7b_lids \
        --output_dir ${OUTPUT_DIR} \
        --max_threads 1 \
        --rank ${rank} \
        --world_size ${LIDS_WORLD_SIZE}
done
wait

echo ""
echo "=== Merging qwen25_7b_lids results ==="
python -m eval_system_bench \
    --merge_only \
    --infer_model_name qwen25_7b_lids \
    --output_dir ${OUTPUT_DIR} \
    --world_size ${LIDS_WORLD_SIZE}
