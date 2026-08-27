#!/usr/bin/env python3
"""环境工程评测：基座 / SFT / SFT+DPO / 量化 四组对比。

无模型时也可对预测 JSONL 做三层验证，便于先检查验证器本身。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from env_validators import evaluate_response, instruction_constraints, score_response  # noqa: E402


def load_jsonl(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def metrics_from_eval(rows: list) -> dict:
    buckets = defaultdict(list)
    fabricated = 0
    uncertainty_ok = 0
    uncertainty_n = 0
    for row in rows:
        ev = row["eval"]
        buckets["instruction_following"].append(1.0 if ev["format"]["passed"] else 0.0)
        calc = ev["calculation"]
        if not calc.get("skipped"):
            buckets["calculation"].append(1.0 if calc["passed"] else 0.0)
            unit = calc.get("checks", {}).get("load_unit")
            if unit is not None:
                buckets["unit"].append(1.0 if unit else 0.0)
        headers = ev["format"]["checks"].get("headers")
        if headers is not None:
            buckets["field_completeness"].append(1.0 if headers else 0.0)
        know = ev["knowledge"]["checks"]
        if "no_fabricated_standard" in know:
            fabricated += 0 if know["no_fabricated_standard"] else 1
        if "uncertainty" in know:
            uncertainty_n += 1
            uncertainty_ok += 1 if know["uncertainty"] else 0
        buckets["layer_score"].append(score_response(ev))
    def avg(key):
        xs = buckets[key]
        return round(sum(xs) / len(xs), 4) if xs else None
    n = len(rows)
    return {
        "n": n,
        "instruction_following_accuracy": avg("instruction_following"),
        "calculation_accuracy": avg("calculation"),
        "unit_accuracy": avg("unit"),
        "field_completeness": avg("field_completeness"),
        "fabricated_standard_rate": round(fabricated / n, 4) if n else None,
        "uncertainty_handling_rate": round(uncertainty_ok / uncertainty_n, 4) if uncertainty_n else None,
        "mean_layer_score": avg("layer_score"),
    }


def evaluate_predictions(pred_path: Path, test_rows: list) -> list:
    preds = {r.get("query"): r.get("response") or r.get("output") or r.get("prediction") for r in load_jsonl(pred_path)}
    out = []
    for item in test_rows:
        q = item["query"]
        resp = preds.get(q, "")
        gold = item.get("gold") or {}
        fmt = gold.get("format_constraints") or instruction_constraints(item.get("instruction", ""))
        ev = evaluate_response(resp, query=q, gold=gold, format_constraints=fmt)
        out.append({"query": q, "category": item.get("category"), "eval": ev, "response": resp})
    return out


def generate_with_model(model_path: str, test_rows: list, max_new_tokens: int = 512) -> list:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16
    )
    out = []
    t0 = time.time()
    for item in test_rows:
        prompt = item["query"]
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.2, do_sample=False)
        resp = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        gold = item.get("gold") or {}
        ev = evaluate_response(resp, query=prompt, gold=gold, format_constraints=gold.get("format_constraints"))
        out.append({"query": prompt, "category": item.get("category"), "eval": ev, "response": resp})
    latency = (time.time() - t0) / max(len(test_rows), 1)
    for row in out:
        row["latency_s"] = latency
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default=str(ROOT / "data" / "environmental_engineering_test.jsonl"))
    parser.add_argument("--pred-file", help="已有预测 JSONL，每行含 query 与 response")
    parser.add_argument("--model-path", help="待评测模型目录")
    parser.add_argument("--tag", default="manual")
    parser.add_argument("--out-dir", default=str(ROOT / "eval_results"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    test_rows = load_jsonl(Path(args.test_file))
    if args.limit:
        test_rows = test_rows[: args.limit]

    if args.pred_file:
        detailed = evaluate_predictions(Path(args.pred_file), test_rows)
    elif args.model_path:
        detailed = generate_with_model(args.model_path, test_rows)
    else:
        raise SystemExit("请提供 --pred-file 或 --model-path")

    summary = metrics_from_eval(detailed)
    summary["tag"] = args.tag
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.tag}_detailed.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in detailed), encoding="utf-8"
    )
    (out_dir / f"{args.tag}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
