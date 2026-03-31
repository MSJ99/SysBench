import json
import os
from collections import Counter

def check_cache(path):
    if not os.path.exists(path):
        print(f"  NOT FOUND: {path}")
        return
    data = json.load(open(path, encoding="utf-8"))
    total = len(data)
    none_count = sum(1 for d in data if d.get("eval_results") is None)
    dup_ids = [sid for sid, cnt in Counter(d["system_id"] for d in data).items() if cnt > 1]
    print(f"  {os.path.basename(path)}: total={total}, eval_results=None: {none_count}, duplicate system_ids: {len(dup_ids)}")

print("=== qwen25_7b_lids ===")
for i in range(4):
    check_cache(f"output/qwen25_7b_lids/qwen25_7b_lids_eval_rank{i}_cache.json")

print("\n=== qwen25_7b_lids (merged eval.json) ===")
check_cache("output/qwen25_7b_lids/qwen25_7b_lids_eval.json")

print("\n=== qwen25_7b_vllm ===")
check_cache("output/qwen25_7b_vllm/qwen25_7b_vllm_eval_cache.json")
check_cache("output/qwen25_7b_vllm/qwen25_7b_vllm_eval.json")
