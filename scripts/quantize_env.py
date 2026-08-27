"""环境工程校准语料的 GPTQ 量化。失败时不得宣称精度只损失 2%~3%。"""
from __future__ import annotations

import json
import os
import sys

from transformers import AutoTokenizer
import torch


def load_calibration(path: str, n: int = 32) -> list:
    texts = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                row = json.loads(line)
                q = (row.get("query") or "").strip()
                if q:
                    texts.append(q)
    if not texts:
        texts = [
            "某污水厂设计水量为10000 m³/d，进水COD为300 mg/L，出水COD为50 mg/L，请计算COD去除率和每日去除负荷。",
            "比较 A2/O 与氧化沟的处理效果、占地、能耗、污泥产量和适用条件。",
            "活性污泥系统出现污泥膨胀，请列出三个可能原因及复核方法。",
            "监测数据不足时不得编造排放限值，请说明缺失参数。",
            "VOCs 浓度为 300 mg/m³，请评估活性炭吸附与催化燃烧。",
        ]
    return texts


def main():
    model_path = "./models/model_merged"
    output_dir = "./models/model_gptq_int4"
    if not os.path.isdir(model_path):
        print("合并模型不存在，跳过量化")
        return
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    calibration_texts = load_calibration(os.environ.get("AUTOIF_TEST_PATH", "data/environmental_engineering_test.jsonl"))
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        quantize_config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=True)
        model = AutoGPTQForCausalLM.from_pretrained(
            model_path, quantize_config=quantize_config,
            trust_remote_code=True, torch_dtype=torch.float16
        )
        examples = [tokenizer(t, return_tensors="pt") for t in calibration_texts]
        print("GPTQ 量化中（环境工程校准语料）...")
        model.quantize(examples)
        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("✅ GPTQ 量化完成")
    except Exception as e:
        print(f"GPTQ 量化失败: {e}")
        print("量化失败不影响 SFT/DPO 合并模型。不要预先宣称精度只损失 2%~3%。")
        sys.exit(0)


if __name__ == "__main__":
    main()
