#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

model="Qwen/Qwen2.5-7B-Instruct"
vllm serve $model \
    --dtype bfloat16 \
    --quantization bitsandbytes \
    --port 33619 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --api-key custom-key \
    --trust-remote-code
