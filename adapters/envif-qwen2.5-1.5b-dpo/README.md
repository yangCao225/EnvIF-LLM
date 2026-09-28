---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
license: mit
pipeline_tag: text-generation
tags:
  - lora
  - dpo
  - instruction-following
  - wastewater
---

# EnvIF DPO LoRA (Qwen2.5-1.5B)

Continues from `adapters/envif-qwen2.5-1.5b-sft` on **400** stratified offline DPO pairs
(`output/dpo_pairs.jsonl`, source `envif_tlr_offline`). Not teacher-distilled.

## This snapshot

- 1 epoch, lr 5e-6, max_len 512, 8bit, r=8
- mean train loss **7.4e-6**; last-step loss **0.0**
- `rewards/accuracies` stayed 1.0: typed rejected answers are already far from chosen, so DPO barely moves the adapter

This is **not** an environmental-engineering expert model. EnvIF-Bench (`eval.envif_bench_sft_dpo`): prompt_accuracy 0.2353, below SFT 0.3529. Do not treat DPO loss≈0 as an improvement.
