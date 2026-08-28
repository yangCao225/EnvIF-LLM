---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
license: mit
pipeline_tag: text-generation
tags:
  - lora
  - instruction-following
  - wastewater
  - environmental-monitoring
---

# EnvIF SFT LoRA (Qwen2.5-1.5B)

LoRA adapter trained on `output/IF_sft_data.json` (798 EnvIF-TLR SFT samples).
Base model: **Qwen/Qwen2.5-1.5B-Instruct**.

This adapter is for **structured instruction following** on wastewater / monitoring templates.
It is **not** an environmental-engineering expert model.

## Files to load

- `adapter_config.json`
- `adapter_model.safetensors` (~35 MB)

## Training (this snapshot)

- LoRA r=8, alpha=16, dropout=0.05
- 2 epochs, batch 1 × accum 8, lr 5e-5, cosine
- 8-bit load, max length 1024
- mean train loss **0.524**; last-step loss **0.028** (template-heavy SFT; last-step drop can overfit)

DPO adapter is not included.

## Load

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, "adapters/envif-qwen2.5-1.5b-sft")
```
