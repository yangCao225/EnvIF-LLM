#!/usr/bin/env python3
"""从 logs/ 与 output/ 汇总真实实验数字，供 README 和 PDF 引用。"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "experiment_summary.json"


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return len(data) if isinstance(data, list) else 1
        except Exception:
            return 0
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def parse_log_metrics(log_dir: Path) -> dict:
    metrics = {}
    for name in ["sft_train.log", "dpo_train.log", "step8.log", "step9.log", "dpo2.log"]:
        p = log_dir / name
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        losses = [float(x) for x in re.findall(r"'loss':\s*([0-9.]+)", text)]
        if not losses:
            losses = [float(x) for x in re.findall(r"\bloss[:=]\s*([0-9.]+)", text, re.I)]
        if losses:
            metrics[name.replace(".log", "_last_loss")] = losses[-1]
            metrics[name.replace(".log", "_first_loss")] = losses[0]
        kept = re.findall(r"保留:\s*(\d+)/(\d+)", text)
        if kept:
            metrics["filter_kept"], metrics["filter_total"] = map(int, kept[-1])
        pairs = re.findall(r"共 (\d+) 个偏好对", text)
        if pairs:
            metrics["dpo_pairs_log"] = int(pairs[-1])
        sft_n = re.findall(r"共 (\d+) 条", text)
        if sft_n and "step9" in name:
            metrics["sft_from_log"] = int(sft_n[-1])
    return metrics


def source_counts(path: Path) -> dict:
    from collections import Counter
    c = Counter()
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            c[row.get("source", "template")] += 1
    return dict(c)


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    log_dir = ROOT / "logs"
    summary = {
        "status": "not_run" if not (ROOT / "output").exists() or not any((ROOT / "output").glob("*")) else "partial",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": "文档与讲义只允许引用本文件中的数字。未跑通的 GPU 训练字段保持 null。SFT/DPO 条数若来自 EnvIF-TLR 离线构造，不是 LoRA Loss。",
        "method": "EnvIF-TLR (Typed Layered Reward on AutoIF)",
        "domain": "环境工程",
        "counts": {
            "seed_instructions": count_lines(ROOT / "sample_data" / "seed_instruction_环境工程.txt"),
            "domain_queries": count_lines(ROOT / "data" / "environmental_engineering_queries.jsonl"),
            "heldout_test": count_lines(ROOT / "data" / "environmental_engineering_test.jsonl"),
            "augment_instructions": count_lines(ROOT / "output" / "augment_instructions.txt"),
            "eval_func_rft": count_lines(ROOT / "output" / "eval_func_rft.jsonl"),
            "cross_validation": count_lines(ROOT / "output" / "cross_validation.jsonl"),
            "back_trans_filter": count_lines(ROOT / "output" / "back_trans_filter.jsonl"),
            "query_rft": count_lines(ROOT / "output" / "query_rft.jsonl"),
            "query_rft_score": count_lines(ROOT / "output" / "query_rft_score.jsonl"),
            "query_score_filter": count_lines(ROOT / "output" / "query_score_filter.jsonl"),
            "sft": count_lines(ROOT / "output" / "IF_sft_data.json"),
            "dpo_pairs": count_lines(ROOT / "output" / "dpo_pairs.jsonl"),
            "envif_bench": count_lines(ROOT / "data" / "envif_bench" / "envif_bench.jsonl"),
        },
        "query_source": {
            "train": source_counts(ROOT / "data" / "environmental_engineering_queries.jsonl"),
            "test": source_counts(ROOT / "data" / "environmental_engineering_test.jsonl"),
        },
        "construction": load_json(ROOT / "output" / "envif_offline_meta.json"),
        "training": {
            "sft_learning_rate": 5.0e-5,
            "dpo_learning_rate": 5.0e-6,
            "sft_last_loss": None,
            "dpo_last_loss": None,
            "sft_time": None,
            "dpo_time": None,
        },
        "eval": {
            "base": None,
            "sft": None,
            "sft_dpo": None,
            "quantized": None,
            "tlr_ablation": load_json(ROOT / "output" / "envif_ablation.json"),
            "envif_bench_oracle": load_json(ROOT / "eval_results" / "envif_bench_oracle_summary.json"),
        },
        "deploy": {
            "merged_model_size": None,
            "quantized_model_size": None,
            "latency_ms": None,
        },
        "log_metrics": parse_log_metrics(log_dir),
    }
    if (log_dir / "sft_train.log").exists() or (log_dir / "dpo_train.log").exists():
        summary["status"] = "from_logs"
    elif summary["counts"]["sft"] or summary["counts"]["dpo_pairs"]:
        summary["status"] = "envif_offline_ready"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {OUT}")
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
