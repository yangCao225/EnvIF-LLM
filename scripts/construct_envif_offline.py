#!/usr/bin/env python3
"""用 EnvIF-TLR 离线构造 SFT / 分类型 DPO，不调用教师模型。

产出可供 LlamaFactory 训练的文件；Loss 仍须 GPU 训练后才写入 experiment_summary。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from env_validators import evaluate_response, score_response  # noqa: E402
from envif_tlr import build_pair, chosen_response, pick_instruction, typed_negatives  # noqa: E402


def load_jsonl(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_seeds() -> list[str]:
    p = ROOT / "sample_data" / "seed_instruction_环境工程.txt"
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    train = load_jsonl(ROOT / "data" / "environmental_engineering_queries.jsonl")
    seeds = load_seeds()
    sft = []
    pairs = []
    skipped = 0
    for row in train:
        query = row["query"]
        gold = row.get("gold") or {}
        cat = row.get("category") or ""
        inst = pick_instruction(cat, seeds)
        chosen = chosen_response(query, gold, cat)
        ev = evaluate_response(chosen, query=query, gold=gold, format_constraints=gold.get("format_constraints"))
        sc = score_response(ev)
        if sc < 0.72:
            skipped += 1
            continue
        sft.append({
            "instruction": f"{query} {inst}",
            "input": "",
            "output": chosen,
            "history": [],
            "category": cat,
            "layer_score": sc,
            "source": "envif_tlr_offline",
        })
        for neg in typed_negatives(query, gold, chosen)[:3]:
            pairs.append(build_pair(query, inst, chosen, neg["text"], neg["error_type"], gold=gold, category=cat))

    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    sft_path = out / "IF_sft_data.json"
    dpo_path = out / "dpo_pairs.jsonl"
    sft_path.write_text(json.dumps(sft, ensure_ascii=False, indent=2), encoding="utf-8")
    with dpo_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    meta = {
        "mode": "envif_tlr_offline",
        "sft": len(sft),
        "dpo_pairs": len(pairs),
        "skipped_low_score": skipped,
        "note": "由带单位计算层 + 分类型负样本构造，不是教师模型蒸馏；LoRA Loss 仍为 null。",
    }
    (out / "envif_offline_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
