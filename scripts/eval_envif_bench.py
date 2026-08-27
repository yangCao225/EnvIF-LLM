#!/usr/bin/env python3
"""评测 EnvIF-Bench。可对模型预测或内置 reference（oracle）打分。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from envif_bench import aggregate, evaluate_item  # noqa: E402


def load_jsonl(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default=str(ROOT / "data" / "envif_bench" / "envif_bench.jsonl"))
    parser.add_argument("--pred-file", help="JSONL，字段 id 或 query + response")
    parser.add_argument("--oracle", action="store_true", help="用样本自带 reference 作答，检查 verifier 自洽")
    parser.add_argument("--tag", default="manual")
    parser.add_argument("--out-dir", default=str(ROOT / "eval_results"))
    args = parser.parse_args()

    bench = load_jsonl(Path(args.bench))
    preds = {}
    if args.pred_file:
        for row in load_jsonl(Path(args.pred_file)):
            key = row.get("id") or row.get("query")
            preds[key] = row.get("response") or row.get("output") or ""

    detailed = []
    for item in bench:
        if args.oracle:
            resp = item.get("reference") or ""
        else:
            resp = preds.get(item["id"], preds.get(item["query"], ""))
        detailed.append({**evaluate_item(item, resp), "response": resp})

    summary = aggregate(detailed)
    summary["tag"] = args.tag
    summary["mode"] = "oracle" if args.oracle else "pred"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"envif_bench_{args.tag}_detailed.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in detailed), encoding="utf-8"
    )
    (out_dir / f"envif_bench_{args.tag}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
