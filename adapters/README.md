# EnvIF LoRA adapters

This folder is the **release location** for instruction-following LoRA adapters
trained on the EnvIF dataset (Qwen2.5-1.5B student by default).

## Current status

GPU SFT/DPO has not been finished in the public snapshot, so **binary adapter
weights are not in git**. After you run `bash run_all.sh --domain 环境工程` on
AutoDL, export the adapter here:

```text
adapters/
  envif-qwen2.5-1.5b-sft/
    adapter_config.json
    adapter_model.safetensors
  envif-qwen2.5-1.5b-dpo/
    adapter_config.json
    adapter_model.safetensors
```

Then either:

1. Upload weights with Git LFS (`git lfs track "*.safetensors"`), or
2. Push the folder to Hugging Face and put the model card URL in this README.

Do not commit full 7B/1.5B base models. Only LoRA deltas belong here.

## Export from LlamaFactory

```bash
python scripts/export_lora_adapter.py \
  --src ./models/model_c_sft \
  --dst ./adapters/envif-qwen2.5-1.5b-sft

python scripts/export_lora_adapter.py \
  --src ./models/model_c_dpo \
  --dst ./adapters/envif-qwen2.5-1.5b-dpo
```

## Inference demo (when weights exist)

```bash
python scripts/smoke_infer.py \
  --base ./models/student \
  --adapter ./adapters/envif-qwen2.5-1.5b-dpo
```
