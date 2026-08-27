#!/usr/bin/env python3
"""层次 ablation：验证 TLR 相对单层规则能否把 chosen 排在 typed rejected 之上。

这是不依赖 GPU 的对照实验，回答「验证器算不算科研贡献」：
贡献不在正则本身，而在分层奖励能否按错误类型分开偏好对。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from env_validators import evaluate_response, instruction_constraints  # noqa: E402
from envif_tlr import ABLATION_SETS, tlr_score  # noqa: E402


def load_jsonl(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_human(text: str) -> tuple[str, str]:
    if "【格式要求】" in text:
        q, inst = text.split("【格式要求】", 1)
        return q.strip(), inst.strip()
    return text.strip(), ""


def main():
    pairs_path = ROOT / "output" / "dpo_pairs.jsonl"
    if not pairs_path.exists():
        raise SystemExit("请先运行 python scripts/construct_envif_offline.py")
    pairs = load_jsonl(pairs_path)
    # 控制规模，保证可复现且几秒内跑完
    if len(pairs) > 800:
        pairs = pairs[:: max(1, len(pairs) // 800)][:800]

    wins = {layers: defaultdict(int) for layers in ABLATION_SETS}
    total = {layers: defaultdict(int) for layers in ABLATION_SETS}

    for pair in pairs:
        query, inst = parse_human(pair["conversations"][0]["value"])
        chosen = pair["chosen"]["value"]
        rejected = pair["rejected"]["value"]
        et = pair.get("error_type") or "other"
        gold = pair.get("gold") or {}
        fmt = gold.get("format_constraints") or instruction_constraints(inst)
        ev_c = evaluate_response(chosen, query=query, gold=gold, format_constraints=fmt)
        ev_r = evaluate_response(rejected, query=query, gold=gold, format_constraints=fmt)
        for layers in ABLATION_SETS:
            sc = tlr_score(ev_c, layers)
            sr = tlr_score(ev_r, layers)
            total[layers][et] += 1
            total[layers]["all"] += 1
            if sc > sr:
                wins[layers][et] += 1
                wins[layers]["all"] += 1

    report = {"n_pairs": len(pairs), "ranking_win_rate": {}}
    for layers in ABLATION_SETS:
        name = "+".join(layers)
        report["ranking_win_rate"][name] = {}
        for k, n in sorted(total[layers].items()):
            report["ranking_win_rate"][name][k] = round(wins[layers][k] / n, 4) if n else None

    out = ROOT / "output" / "envif_ablation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    eval_dir = ROOT / "eval_results"
    eval_dir.mkdir(exist_ok=True)
    (eval_dir / "envif_ablation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
