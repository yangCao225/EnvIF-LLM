#!/usr/bin/env python3
"""在本机跑通环境工程全流程（无 7B 教师、无 vLLM、无 LlamaFactory 训练）。

对应 run_all.sh 在 6GB Windows 上能完成的部分：
  1. CPU 测试 + EnvIF-TLR 离线 SFT/DPO
  2. 导出 LlamaFactory 数据格式（即使尚未安装 LlamaFactory）
  3. 层次 ablation + Bench oracle
  4. 若有 CUDA：冒烟推理 + 309 条领域测试（基座 / SFT / DPO）
  5. 写入 output/experiment_summary.json

不会伪造 AutoIF 九步的 query_rft 条数。那些仍需要 AutoDL 上的 7B 教师。
已有 SFT/DPO LoRA 默认直接用，不覆盖。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def run(cmd: list[str], log: Path | None = None) -> int:
    print(f"\n>> {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(f"\n$ {' '.join(cmd)}\nexit={proc.returncode}\n")
    if proc.returncode != 0:
        print(f"失败 exit={proc.returncode}: {' '.join(cmd)}", flush=True)
    return proc.returncode


def py(*args: str) -> list[str]:
    return [sys.executable, *args]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--skip-test-eval", action="store_true", help="只做 Bench/冒烟，不跑 309 条领域测试")
    parser.add_argument("--test-n", type=int, default=0, help="领域测试条数，0=全部 309")
    parser.add_argument("--train", action="store_true", help="再训 SFT（默认使用已有 adapter）")
    args = parser.parse_args()

    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    pipe_log = logs / "local_pipeline.log"
    meta = {
        "mode": "envif_tlr_local",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "teacher_7b": False,
        "vllm": False,
        "llamafactory_train": False,
        "note": "本机全流程。数据来自 EnvIF-TLR 离线构造，不是 7B 教师蒸馏。",
        "steps": {},
    }

    print("============================================")
    print("  本机全流程（EnvIF-TLR + 已有 LoRA）")
    print("  不是 AutoDL 7B 教师九步")
    print("============================================", flush=True)

    steps = [
        ("cpu_tests", py("tests/test_env_validators.py")),
        ("cpu_tests_tlr", py("tests/test_envif_tlr.py")),
        ("cpu_tests_bench", py("tests/test_envif_bench.py")),
        ("construct", py("scripts/construct_envif_offline.py")),
        ("prepare_lf", py("scripts/prepare_llamafactory_data.py")),
        ("ablate", py("scripts/ablate_envif_layers.py")),
        ("bench_oracle", py("scripts/eval_envif_bench.py", "--oracle", "--tag", "oracle")),
    ]
    for name, cmd in steps:
        code = run(cmd, pipe_log)
        meta["steps"][name] = code == 0
        if code != 0 and name.startswith("cpu_tests"):
            raise SystemExit(f"CPU 测试失败: {name}")

    cuda = False
    if not args.skip_gpu:
        try:
            import torch

            cuda = bool(torch.cuda.is_available())
        except Exception:
            cuda = False
    meta["cuda"] = cuda
    student = ROOT / "models" / "student"
    sft = ROOT / "adapters" / "envif-qwen2.5-1.5b-sft"
    dpo = ROOT / "adapters" / "envif-qwen2.5-1.5b-dpo"

    if args.train and cuda and student.joinpath("config.json").is_file():
        meta["steps"]["train_sft"] = (
            run(
                py("scripts/train_envif_lora.py", "--out", str(sft), "--epochs", "1", "--fp16", "--max-len", "512"),
                pipe_log,
            )
            == 0
        )

    if cuda and student.joinpath("config.json").is_file():
        meta["steps"]["smoke"] = (
            run(py("scripts/smoke_infer.py", "--base", str(student), "--adapter", str(sft), "--n", "3"), pipe_log)
            == 0
        )
        if not args.skip_test_eval:
            jobs = [
                ("base", ""),
                ("sft", str(sft) if sft.joinpath("adapter_config.json").is_file() else ""),
                ("sft_dpo", str(dpo) if dpo.joinpath("adapter_config.json").is_file() else ""),
            ]
            for tag, adapter in jobs:
                infer_cmd = py(
                    "scripts/infer_envif.py",
                    "--source",
                    "test",
                    "--base",
                    str(student),
                    "--fp16",
                    "--tag",
                    tag,
                )
                if adapter:
                    infer_cmd += ["--adapter", adapter]
                if args.test_n:
                    infer_cmd += ["--n", str(args.test_n)]
                ok_i = run(infer_cmd, pipe_log) == 0
                pred = ROOT / "eval_results" / f"test_{tag}_preds.jsonl"
                ok_e = False
                if ok_i and pred.is_file():
                    ok_e = (
                        run(
                            py(
                                "scripts/eval_env_engineering.py",
                                "--pred-file",
                                str(pred),
                                "--tag",
                                f"test_{tag}",
                            ),
                            pipe_log,
                        )
                        == 0
                    )
                meta["steps"][f"test_{tag}"] = ok_i and ok_e
    else:
        meta["steps"]["smoke"] = False
        meta["gpu_skip_reason"] = "no_cuda_or_no_student" if not args.skip_gpu else "skip_gpu"

    run(py("scripts/build_experiment_summary.py"), pipe_log)
    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    (ROOT / "output" / "local_pipeline_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run(py("scripts/build_experiment_summary.py"), pipe_log)
    print("\n本机全流程结束。meta -> output/local_pipeline_meta.json", flush=True)
    print(json.dumps(meta["steps"], ensure_ascii=False, indent=2), flush=True)
    failed = [k for k, v in meta["steps"].items() if not v]
    if failed:
        print("未成功步骤:", ", ".join(failed), flush=True)
        # GPU 失败不让整个脚本以非 0 退出：CPU 数据链路仍算跑通
        if any(not k.startswith("test_") and k != "smoke" and k != "train_sft" for k in failed):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
