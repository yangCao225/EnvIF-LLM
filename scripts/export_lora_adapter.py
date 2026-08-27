#!/usr/bin/env python3
"""Copy a LlamaFactory / PEFT LoRA folder into adapters/ for release.

Does not invent weights. Source must already contain adapter_config.json
and adapter_model.safetensors (or .bin).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REQUIRED_ANY = ("adapter_model.safetensors", "adapter_model.bin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="trained LoRA directory")
    parser.add_argument("--dst", required=True, help="release directory under adapters/")
    args = parser.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    cfg = src / "adapter_config.json"
    weight = next((src / n for n in REQUIRED_ANY if (src / n).exists()), None)
    if not src.is_dir() or not cfg.exists() or weight is None:
        raise SystemExit(
            f"未找到可发布的 LoRA。需要 {cfg.name} 以及 {' 或 '.join(REQUIRED_ANY)}。\n"
            "请先在 AutoDL 上完成 SFT/DPO，再导出。"
        )
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg, dst / cfg.name)
    shutil.copy2(weight, dst / weight.name)
    for extra in ("README.md", "tokenizer_config.json"):
        p = src / extra
        if p.exists():
            shutil.copy2(p, dst / extra)
    print(f"copied adapter -> {dst}")


if __name__ == "__main__":
    main()
