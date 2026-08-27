#!/usr/bin/env python3
"""将 AutoIF 产物转为 LlamaFactory 可用格式。

SFT: 直接使用 output/IF_sft_data.json（字段 instruction/input/output）。
DPO: 保留 ShareGPT 偏好格式 conversations/chosen/rejected，不做错误二次转换。
"""
from __future__ import annotations

import json
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
from config_loader import get, resolve_path  # noqa: E402


def _wrap_sharegpt_message(value):
    if isinstance(value, dict) and "value" in value:
        return {"from": value.get("from", "gpt"), "value": value.get("value", "")}
    return {"from": "gpt", "value": value if isinstance(value, str) else str(value)}


def prepare_sft(dest_dir: str) -> int:
    src = resolve_path(get("data.sft_data"), os.path.join(ROOT, "output", "IF_sft_data.json"))
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "autoif_sft.json")

    if os.path.isfile(src):
        shutil.copyfile(src, dest)
        with open(dest, encoding="utf-8") as f:
            data = json.load(f)
        n = len(data)
        empty = sum(1 for x in data if not str(x.get("output", "")).strip())
        print(f"SFT 数据: {n} 条 (来自 IF_sft_data.json, 空答案 {empty} 条)")
        return n

    fallback = os.path.join(ROOT, "output", "query_score_filter.jsonl")
    if not os.path.isfile(fallback):
        raise FileNotFoundError("找不到 IF_sft_data.json 或 query_score_filter.jsonl")

    rows = []
    with open(fallback, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            output = (item.get("response") or "").strip()
            if not output:
                ans = item.get("gpt-answer", "")
                if isinstance(ans, list) and ans:
                    output = str(ans[0]).strip()
            query = (item.get("query") or "").strip()
            instruction = (item.get("instruction") or "").strip()
            if not query or not instruction or not output:
                continue
            combined = f"{query} {instruction}"
            rows.append({"instruction": combined, "input": "", "output": output, "history": []})
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"SFT 数据: {len(rows)} 条 (由 query_score_filter.jsonl 的 response 字段构建)")
    return len(rows)


def prepare_dpo(dest_dir: str) -> int:
    src = resolve_path(get("data.dpo_pairs"), os.path.join(ROOT, "output", "dpo_pairs.jsonl"))
    os.makedirs(dest_dir, exist_ok=True)
    dest_jsonl = os.path.join(dest_dir, "autoif_dpo.jsonl")
    dest_json = os.path.join(dest_dir, "autoif_dpo.json")
    if not os.path.isfile(src):
        raise FileNotFoundError(src)

    rows = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            conv = item.get("conversations")
            if not conv:
                human = item.get("instruction") or item.get("query") or item.get("prompt") or ""
                conv = [{"from": "human", "value": human}]
            rows.append({
                "conversations": conv,
                "chosen": _wrap_sharegpt_message(item.get("chosen", "")),
                "rejected": _wrap_sharegpt_message(item.get("rejected", "")),
            })
    with open(dest_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(dest_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"DPO 数据: {len(rows)} 对 (ShareGPT conversations/chosen/rejected)")
    return len(rows)


def register_datasets(dest_dir: str):
    info_path = os.path.join(dest_dir, "dataset_info.json")
    info = {}
    if os.path.isfile(info_path):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
    info["autoif_sft"] = {
        "file_name": "autoif_sft.json",
        "formatting": "alpaca",
        "columns": {"prompt": "instruction", "query": "input", "response": "output"},
    }
    info["autoif_dpo"] = {
        "file_name": "autoif_dpo.json",
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
        },
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"已注册数据集: {info_path}")


def main():
    dest = os.path.join(ROOT, "LlamaFactory", "data")
    prepare_sft(dest)
    prepare_dpo(dest)
    register_datasets(dest)


if __name__ == "__main__":
    main()
