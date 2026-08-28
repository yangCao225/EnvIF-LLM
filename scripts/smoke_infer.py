"""用环境工程测试题做推理冒烟测试。支持基座或基座+LoRA。"""
from __future__ import annotations

import argparse
import json
import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="./models/student")
    parser.add_argument("--adapter", default="./adapters/envif-qwen2.5-1.5b-sft")
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    if not os.path.isdir(args.base):
        print(f"基座不存在: {args.base}，请先 python scripts/train_envif_lora.py")
        return
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16
    )
    adapter_cfg = os.path.join(args.adapter, "adapter_config.json")
    if os.path.isfile(adapter_cfg):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"已加载 LoRA: {args.adapter}")
    else:
        print("未找到 LoRA，仅用基座推理")

    prompts = []
    test_file = os.environ.get("AUTOIF_TEST_PATH", "data/environmental_engineering_test.jsonl")
    if os.path.exists(test_file):
        with open(test_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= args.n:
                    break
                prompts.append(json.loads(line).get("query", ""))
    if not prompts:
        prompts = [
            "某污水厂设计水量为10000 m³/d，进水COD为300 mg/L，出水COD为50 mg/L，请计算COD去除率和每日去除负荷。",
        ]
    model.eval()
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=280, temperature=0.2, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print("\n问:", prompt[:160])
        print("答:", response[:600])
    print("\n推理冒烟完成。正式指标: python scripts/eval_env_engineering.py --pred-file ...")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"推理测试失败: {e}")
        sys.exit(1)
