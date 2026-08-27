"""构造环境工程硬负样本：单位错误、计算错误、只给结论、虚构标准等。"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from utils import logger, DPO_PAIRS_PATH, QUERY_SCORE_FILTER_PATH  # noqa: E402
from envif_tlr import typed_negatives  # noqa: E402


def _chosen_from_item(item: dict) -> str:
    resp = (item.get("response") or "").strip()
    if resp:
        return resp
    return ""


def hard_negatives_for(query: str, gold: dict, chosen: str) -> list:
    if not chosen:
        return []
    return [row["text"] for row in typed_negatives(query, gold or {}, chosen)]


def main():
    if not os.path.exists(QUERY_SCORE_FILTER_PATH):
        logger.warning("缺少 query_score_filter.jsonl，跳过硬负样本扩充")
        return
    extra = 0
    with open(QUERY_SCORE_FILTER_PATH, encoding="utf-8") as src, \
         open(DPO_PAIRS_PATH, "a", encoding="utf-8") as dst:
        for line in src:
            item = json.loads(line)
            gold = item.get("gold") or {}
            query = item.get("query") or ""
            inst = item.get("instruction") or ""
            chosen = _chosen_from_item(item)
            for neg in hard_negatives_for(query, gold, chosen)[:2]:
                human = f"{query}\n\n【格式要求】{inst}".strip()
                pair = {
                    "conversations": [{"from": "human", "value": human}],
                    "chosen": {"from": "gpt", "value": chosen},
                    "rejected": {"from": "gpt", "value": neg},
                    "error_type": "env_hard_negative",
                }
                dst.write(json.dumps(pair, ensure_ascii=False) + "\n")
                extra += 1
    logger.info(f"已追加 {extra} 条环境工程硬负样本到 {DPO_PAIRS_PATH}")


if __name__ == "__main__":
    main()
