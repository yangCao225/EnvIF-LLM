#!/usr/bin/env python3
"""用基座或基座+LoRA 生成 EnvIF-Bench / 领域测试预测，供评测脚本打分。"""
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


def load_jsonl(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prompt_of(item: dict) -> str:
    return (item.get("prompt") or item.get("query") or "").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("bench", "test"), default="bench")
    parser.add_argument("--base", default=str(ROOT / "models" / "student"))
    parser.add_argument("--adapter", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--task", default="", help="只跑指定 task 的题，例如 air_pollution")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--8bit", dest="load_8bit", action="store_true", default=False)
    parser.add_argument("--fp16", dest="load_8bit", action="store_false")
    args = parser.parse_args()

    src = (
        ROOT / "data" / "envif_bench" / "envif_bench.jsonl"
        if args.source == "bench"
        else ROOT / "data" / "environmental_engineering_test.jsonl"
    )
    items = load_jsonl(src)
    if args.task:
        items = [x for x in items if x.get("task") == args.task]
    if args.n:
        items = items[: args.n]
    if not items:
        raise SystemExit(f"空数据: {src}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not Path(args.base).is_dir():
        raise SystemExit(f"基座不存在: {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    kw = dict(trust_remote_code=True, low_cpu_mem_usage=True, device_map={"": 0})
    if args.load_8bit:
        kw["load_in_8bit"] = True
        kw["device_map"] = "auto"
    else:
        kw["torch_dtype"] = torch.float16
    print(f"加载基座 8bit={args.load_8bit}: {args.base}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, **kw)
    adapter = args.adapter.strip()
    if adapter and Path(adapter, "adapter_config.json").is_file():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        print(f"已加载 LoRA: {adapter}")
    model.eval()

    if args.tag:
        tag = args.tag
    else:
        al = adapter.replace("\\", "/").lower()
        if "dpo" in al:
            tag = "sft_dpo_v2" if "v2" in al else "sft_dpo"
        elif adapter:
            tag = "sft_v2" if "v2" in al else "sft"
        else:
            tag = "base"
    out = Path(args.out) if args.out else ROOT / "eval_results" / f"{args.source}_{tag}_preds.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, item in enumerate(items, 1):
        prompt = prompt_of(item)
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        resp = tok.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        row = {
            "id": item.get("id") or f"{args.source}-{i:04d}",
            "query": item.get("query") or prompt,
            "prompt": prompt,
            "response": resp,
            "tag": tag,
        }
        rows.append(row)
        print(f"[{i}/{len(items)}] {row['id']}  {len(resp)} chars", flush=True)

    out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    print(f"已写入 {out}  n={len(rows)}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "code"))
    main()
