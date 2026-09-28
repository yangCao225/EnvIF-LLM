#!/usr/bin/env python3
"""现场演示答题：加载 1.5B + LoRA，打印 3 道代表性问答。默认 SFT v2。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

QUESTIONS = [
    {
        "id": "demo-cod",
        "title": "污水计算",
        "prompt": (
            "某污水厂设计水量为10000 m³/d，进水COD为300 mg/L，出水COD为50 mg/L，"
            "请计算COD去除率和每日去除负荷。\n\n"
            "【指令约束】必须写出公式和代入；结果保留两位小数；负荷单位必须是 kg/d；"
            "未给排放标准不得判定达标。"
        ),
        "note": "口头金标：去除率 83.33%；负荷 10000×250×0.001=2500.00 kg/d。",
    },
    {
        "id": "demo-fgd",
        "title": "脱硫工艺比较",
        "prompt": (
            "临海热电厂含硫烟气拟做脱硫改造。请比较石灰石-石膏湿法与半干法。\n\n"
            "【指令约束】必须按优先、备选、不推荐输出；必须涉及脱硫或石膏；"
            "必须评估二次污染与运行安全；不得编造排放限值。"
        ),
        "note": "期望：优先湿法，备选半干法，不推荐高烟囱稀释。",
    },
    {
        "id": "demo-scr",
        "title": "脱硝工艺比较",
        "prompt": (
            "北港烧结车间含氮烟气拟做脱硝。请比较 SCR 与 SNCR。\n\n"
            "【指令约束】必须按优先、备选、不推荐输出；必须涉及脱硝或氨逃逸；"
            "不得编造排放限值。"
        ),
        "note": "期望：优先 SCR，备选 SNCR，提及氨逃逸。",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(ROOT / "models" / "student"))
    parser.add_argument("--adapter", default=str(ROOT / "adapters" / "envif-qwen2.5-1.5b-sft-v2"))
    parser.add_argument("--max-new-tokens", type=int, default=320)
    args = parser.parse_args()

    base = Path(args.base)
    adapter = Path(args.adapter)
    if not (base / "config.json").is_file():
        raise SystemExit(f"基座不存在: {base}")
    if not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"LoRA 不存在: {adapter}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("加载 Qwen2.5-1.5B + SFT LoRA（演示，不是专家模型）", flush=True)
    tok = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    print(f"adapter: {adapter}", flush=True)

    rows = []
    for i, item in enumerate(QUESTIONS, 1):
        print("\n" + "=" * 64, flush=True)
        print(f"{i}/3  {item['title']}", flush=True)
        print(f"提示：{item['note']}", flush=True)
        print(f"问：{item['prompt'].split(chr(10))[0]}", flush=True)
        messages = [{"role": "user", "content": item["prompt"]}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        resp = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        print(f"答：{resp}", flush=True)
        rows.append({"id": item["id"], "title": item["title"], "prompt": item["prompt"], "response": resp})

    dest = ROOT / "eval_results" / "demo_qa.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    print(f"\n已写入 {dest}", flush=True)


if __name__ == "__main__":
    main()
