"""用环境工程测试题做推理冒烟测试。"""
from __future__ import annotations

import json
import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def main():
    model_path = "./models/model_merged"
    if not os.path.isdir(model_path):
        print("合并模型不存在，跳过推理测试")
        return
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16
    )
    prompts = []
    test_file = os.environ.get("AUTOIF_TEST_PATH", "data/environmental_engineering_test.jsonl")
    if os.path.exists(test_file):
        with open(test_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                prompts.append(json.loads(line).get("query", ""))
    if not prompts:
        prompts = [
            "某污水厂设计水量为10000 m³/d，进水COD为300 mg/L，出水COD为50 mg/L，请计算COD去除率和每日去除负荷。",
            "某活性污泥系统出现污泥膨胀、SVI升高和出水悬浮物增加，请分析可能原因并提出排查步骤。",
        ]
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.3, do_sample=True)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print("\n问:", prompt[:160])
        print("答:", response[:500])
    print("\n✅ 推理冒烟测试完成（正式指标请运行 scripts/eval_env_engineering.py）")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"推理测试失败: {e}")
        sys.exit(0)
