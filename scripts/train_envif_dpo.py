#!/usr/bin/env python3
"""在已有 SFT LoRA 上继续训 EnvIF DPO（离线偏好对，不是教师蒸馏）。

默认面向 6GB：8bit + 从 SFT adapter 接着训，max_len=512。OOM 时减序列或减样本。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parents[1]
DPO_PATH = ROOT / "output" / "dpo_pairs.jsonl"
BASE_DIR = ROOT / "models" / "student"
SFT_ADAPTER = ROOT / "adapters" / "envif-qwen2.5-1.5b-sft"
ADAPTER_DIR = ROOT / "adapters" / "envif-qwen2.5-1.5b-dpo"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def load_pairs(max_pairs: int = 0) -> list[dict]:
    rows = []
    with DPO_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            conv = item.get("conversations") or []
            prompt = ""
            if conv:
                prompt = (conv[0].get("value") or "").strip()
            chosen = ((item.get("chosen") or {}).get("value") or "").strip()
            rejected = ((item.get("rejected") or {}).get("value") or "").strip()
            if prompt and chosen and rejected:
                rows.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "error_type": item.get("error_type"),
                    }
                )
    if max_pairs:
        # 按错误类型轮转抽样，避免只剩 format_error
        buckets: dict[str, list] = {}
        for row in rows:
            buckets.setdefault(row.get("error_type") or "other", []).append(row)
        out, i = [], 0
        keys = list(buckets)
        while len(out) < max_pairs and keys:
            key = keys[i % len(keys)]
            if buckets[key]:
                out.append(buckets[key].pop(0))
            else:
                keys.remove(key)
                continue
            i += 1
        rows = out
    if not rows:
        raise SystemExit(f"DPO 为空: {DPO_PATH}")
    print(f"DPO 偏好对 {len(rows)}")
    return rows


def try_train(
    base_path: str,
    rows: list,
    max_len: int,
    use_8bit: bool,
    out_dir: Path,
    sft_adapter: Path,
    epochs: float,
    lr: float,
) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kw = dict(trust_remote_code=True, device_map="auto")
    if use_8bit:
        kw["load_in_8bit"] = True
    else:
        kw["torch_dtype"] = torch.float16
    print(f"加载模型 8bit={use_8bit} max_len={max_len}")
    model = AutoModelForCausalLM.from_pretrained(base_path, **kw)
    model.config.use_cache = False
    if use_8bit:
        model = prepare_model_for_kbit_training(model)
    if sft_adapter.joinpath("adapter_config.json").is_file():
        model = PeftModel.from_pretrained(model, str(sft_adapter), is_trainable=True)
        print(f"从 SFT adapter 继续: {sft_adapter}")
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    def to_chat(prompt: str, answer: str | None = None) -> str:
        msgs = [{"role": "user", "content": prompt}]
        if answer is None:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        msgs.append({"role": "assistant", "content": answer})
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    ds = Dataset.from_list(
        [
            {
                "prompt": to_chat(r["prompt"]),
                "chosen": r["chosen"],
                "rejected": r["rejected"],
            }
            for r in rows
        ]
    )

    args = DPOConfig(
        output_dir=str(ROOT / "models" / "model_c_dpo"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="epoch",
        fp16=not use_8bit,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        max_length=max_len,
        max_prompt_length=max(128, max_len // 2),
        max_completion_length=max(128, max_len // 2),
        beta=0.1,
        seed=42,
        precompute_ref_log_probs=True,
    )
    trainer = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds, processing_class=tok)
    result = trainer.train()
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    cfg_path = out_dir / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["base_model_name_or_path"] = MODEL_ID
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    hist = trainer.state.log_history or []
    step_losses = [h["loss"] for h in hist if "loss" in h]
    try:
        started = sft_adapter.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        started = str(sft_adapter).replace("\\", "/")
    meta = {
        "base_model": MODEL_ID,
        "finetuning": "dpo_lora",
        "started_from": started,
        "lora_rank": 8,
        "epochs": epochs,
        "learning_rate": lr,
        "n_pairs": len(rows),
        "max_length": max_len,
        "load_in_8bit": use_8bit,
        "dataset": "output/dpo_pairs.jsonl",
        "source": "envif_tlr_offline",
        "train_loss": getattr(result, "training_loss", None),
        "last_step_loss": step_losses[-1] if step_losses else None,
        "note": "Offline typed DPO on EnvIF pairs. Not teacher-distilled. Not an expert model.",
    }
    (out_dir / "envif_train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已保存 DPO LoRA -> {out_dir}")
    return out_dir


def main():
    import argparse

    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pairs", type=int, default=400, help="0=全部；默认 400 以免 6GB 训太久")
    parser.add_argument("--sft-adapter", default=str(SFT_ADAPTER))
    parser.add_argument("--out", default=str(ADAPTER_DIR), help="DPO LoRA 输出目录，v2 请另指定以免覆盖 v1")
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU")
    print("cuda", torch.cuda.get_device_name(0))
    if not BASE_DIR.joinpath("config.json").is_file():
        raise SystemExit("请先跑 scripts/train_envif_lora.py 下载基座")
    rows = load_pairs(args.max_pairs)
    sft_adapter = Path(args.sft_adapter)
    out_dir = Path(args.out)
    last_err = None
    for use_8bit, max_len in ((True, 512), (True, 384), (True, 256)):
        try:
            try_train(str(BASE_DIR), rows, max_len, use_8bit, out_dir, sft_adapter, args.epochs, args.lr)
            return
        except Exception as e:
            last_err = e
            print(f"失败 8bit={use_8bit} max_len={max_len}: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
    raise SystemExit(f"DPO 训练失败: {last_err}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "code"))
    main()
