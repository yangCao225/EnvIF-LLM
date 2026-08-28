# EnvIF LoRA adapters

Release location for instruction-following LoRA trained on the EnvIF SFT set.
Base model is **Qwen2.5-1.5B-Instruct**. Full 1.5B/7B checkpoints are **not** in this repo.

## Current release

| Adapter | Path | Status |
| --- | --- | --- |
| SFT LoRA | `adapters/envif-qwen2.5-1.5b-sft/` | **trained** (2 epochs, 798 samples, r=8, lr=5e-5, mean train loss 0.524) |
| DPO LoRA | `adapters/envif-qwen2.5-1.5b-dpo/` | not trained yet (needs GPU preference run) |

Files that matter for loading:

- `adapter_config.json`
- `adapter_model.safetensors` (~35 MB)

This adapter improves **structured instruction following** on wastewater/monitoring templates. It does **not** make a 1.5B model an environmental-engineering expert.

## Train / re-train on a 6GB GPU

```bash
python scripts/train_envif_lora.py
```

Downloads the 1.5B base into `models/student/` (gitignored), then writes the adapter here.

## Inference

```bash
python scripts/smoke_infer.py \
  --base ./models/student \
  --adapter ./adapters/envif-qwen2.5-1.5b-sft
```

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, "adapters/envif-qwen2.5-1.5b-sft")
```

## Export from LlamaFactory (AutoDL)

```bash
python scripts/export_lora_adapter.py --src ./models/model_c_sft --dst ./adapters/envif-qwen2.5-1.5b-sft
python scripts/export_lora_adapter.py --src ./models/model_c_dpo --dst ./adapters/envif-qwen2.5-1.5b-dpo
```
