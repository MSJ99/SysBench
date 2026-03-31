import json
import os

def fix_none_evals(eval_path):
    data_list = json.load(open(eval_path, encoding="utf-8"))
    fixed = 0
    for data in data_list:
        if data.get("eval_results") is not None:
            continue

        eval_results = {}
        messages = [m for m in data["infer_results"] if m["role"] == "user"]
        for message in messages:
            prompt = message["content"]
            criteria = data["prompt_infos"][prompt]["criteria"]
            eval_results[prompt] = {
                "评判理由": "eval skipped: response too large for evaluator token limit",
                "评判结果": {k: "否" for k in criteria},
                "eval_pattern": "",
                "response": "",
                "criteria": criteria,
                "retry_time": -1,
            }
            for m in data["infer_results"]:
                if m["role"] == "assistant":
                    eval_results[prompt]["response"] = m["content"]

        data["eval_results"] = eval_results
        fixed += 1

    if fixed > 0:
        json.dump(data_list, open(eval_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"Fixed {fixed} entries in {os.path.basename(eval_path)}")
    else:
        print(f"No None entries in {os.path.basename(eval_path)}")

for rank in range(4):
    path = f"output/qwen25_7b_lids/qwen25_7b_lids_eval_rank{rank}_cache.json"
    if os.path.exists(path):
        fix_none_evals(path)
    path = f"output/qwen25_7b_lids/qwen25_7b_lids_eval_rank{rank}.json"
    if os.path.exists(path):
        fix_none_evals(path)
