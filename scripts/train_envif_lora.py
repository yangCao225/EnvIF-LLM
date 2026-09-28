#!/usr/bin/env python3
"""在本机训练 EnvIF LoRA（默认 Qwen2.5-1.5B-Instruct + 已有 SFT 数据）。

面向 6GB 笔记本：优先 8bit + gradient checkpointing；OOM 时自动降序列长度。
权重写到 adapters/envif-qwen2.5-1.5b-sft/，这是可发布的 LoRA，不是完整基座。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")

ROOT = Path(__file__).resolve().parents[1]
SFT_PATH = ROOT / "output" / "IF_sft_data.json"
BASE_DIR = ROOT / "models" / "student"
ADAPTER_DIR = ROOT / "adapters" / "envif-qwen2.5-1.5b-sft"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def download_base() -> str:
    cfg = BASE_DIR / "config.json"
    if cfg.exists():
        print(f"使用已有基座: {BASE_DIR}")
        return str(BASE_DIR)
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"下载 {MODEL_ID} -> {BASE_DIR}")
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        path = snapshot_download(MODEL_ID, local_dir=str(BASE_DIR))
        return path
    except Exception as e:
        print(f"ModelScope 失败 ({e})，改走 hf-mirror")
        from huggingface_hub import snapshot_download as hf_dl
        return hf_dl(MODEL_ID, local_dir=str(BASE_DIR), resume_download=True)


def load_sft(path: Path | None = None) -> list[dict]:
    sft_path = path or SFT_PATH
    data = json.loads(sft_path.read_text(encoding="utf-8"))
    rows = []
    for item in data:
        inst = (item.get("instruction") or "").strip()
        out = (item.get("output") or "").strip()
        if inst and out:
            rows.append({"instruction": inst, "output": out})
    if not rows:
        raise SystemExit(f"SFT 为空: {sft_path}")
    print(f"SFT 样本 {len(rows)}")
    return rows


def tokenize_rows(rows, tokenizer, max_len: int):
    import torch
    toks = []
    for row in rows:
        user = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["instruction"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": row["instruction"]},
                {"role": "assistant", "content": row["output"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        full_ids = tokenizer(full, truncation=True, max_length=max_len, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(user, truncation=True, max_length=max_len, add_special_tokens=False)["input_ids"]
        labels = list(full_ids)
        n_prompt = min(len(prompt_ids), len(labels))
        labels[:n_prompt] = [-100] * n_prompt
        toks.append({
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        })
    return toks


def try_train(
    base_path: str,
    rows: list,
    max_len: int,
    use_8bit: bool,
    out_dir: Path,
    epochs: float,
    lr: float,
    resume: str = "",
) -> Path:
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 先分词再加载权重，避免 CPU 上同时摊开模型和长文本
    tokenized = tokenize_rows(rows, tokenizer, max_len)

    kw = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    if use_8bit:
        kw["load_in_8bit"] = True
        kw.pop("device_map", None)
        kw["device_map"] = "auto"
    else:
        kw["torch_dtype"] = torch.float16
    print(f"加载模型 8bit={use_8bit} max_len={max_len} device_map={kw.get('device_map')}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(base_path, **kw)
    model.config.use_cache = False

    from torch.utils.data import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

    if use_8bit:
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    resume = (resume or "").strip()
    if resume and Path(resume, "adapter_config.json").is_file():
        model = PeftModel.from_pretrained(model, resume, is_trainable=True)
        print(f"续训 LoRA: {resume}", flush=True)
    else:
        lora = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    class TorchListDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    ds = TorchListDataset(tokenized)
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8)

    out_tmp = ROOT / "models" / "model_c_sft"
    args = TrainingArguments(
        output_dir=str(out_tmp),
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
        dataloader_num_workers=0,
        max_grad_norm=1.0,
        seed=42,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    result = trainer.train()
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    cfg_path = out_dir / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["base_model_name_or_path"] = MODEL_ID
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    hist = trainer.state.log_history or []
    step_losses = [h["loss"] for h in hist if "loss" in h]
    prior_epochs = 0.0
    if resume:
        prev_meta = Path(resume) / "envif_train_meta.json"
        if prev_meta.is_file():
            try:
                prior_epochs = float(json.loads(prev_meta.read_text(encoding="utf-8")).get("epochs") or 0)
            except Exception:
                prior_epochs = 0.0
    meta = {
        "base_model": MODEL_ID,
        "finetuning": "lora",
        "lora_rank": 8,
        "lora_alpha": 16,
        "epochs": prior_epochs + epochs,
        "this_run_epochs": epochs,
        "prior_epochs": prior_epochs or None,
        "learning_rate": lr,
        "n_sft": len(rows),
        "max_length": max_len,
        "load_in_8bit": use_8bit,
        "dataset": "output/IF_sft_data.json",
        "resume_adapter": resume or None,
        "train_loss": getattr(result, "training_loss", None),
        "last_step_loss": step_losses[-1] if step_losses else None,
        "note": (
            "EnvIF SFT v2 LoRA: wastewater + monitoring + air (SO2/NOx, FGD/SCR, O2 correction). "
            "Not a full expert model. Does not overwrite SFT v1."
            + (f" Continued from {resume}." if resume else "")
            if "v2" in str(out_dir)
            else "EnvIF SFT LoRA on wastewater/monitoring instruction-following data. Not a full expert model."
        ),
    }
    (out_dir / "envif_train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 LoRA -> {out_dir}")
    return out_dir


def main():
    import argparse

    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ADAPTER_DIR), help="LoRA 输出目录，v2 请另指定以免覆盖 v1")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--sft-path", default=str(SFT_PATH))
    parser.add_argument("--max-len", type=int, default=0, help=">0 时只试这一档，避免同进程反复加载")
    parser.add_argument("--fp16", action="store_true", help="不用 8bit，降低 CPU/页面文件压力")
    parser.add_argument("--resume", default="", help="已有 LoRA 目录，续训而不覆盖 v1")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU")
    print("cuda", torch.cuda.get_device_name(0))
    base = download_base()
    rows = load_sft(Path(args.sft_path))
    out_dir = Path(args.out)
    last_err = None
    if args.max_len:
        configs = ((not args.fp16, args.max_len),)
    elif args.fp16:
        configs = ((False, 768), (False, 512))
    else:
        configs = ((True, 1024), (True, 768), (False, 512))
    for use_8bit, max_len in configs:
        try:
            try_train(
                base, rows, max_len, use_8bit, out_dir, args.epochs, args.lr,
                resume=args.resume,
            )
            return
        except Exception as e:
            last_err = e
            print(f"失败 8bit={use_8bit} max_len={max_len}: {type(e).__name__}: {e}")
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            continue
    raise SystemExit(f"训练失败: {last_err}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "code"))
    main()
