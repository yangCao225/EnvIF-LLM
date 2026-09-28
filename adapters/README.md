# EnvIF LoRA

基座 **Qwen2.5-1.5B-Instruct**（在 gitignored 的 `models/student/`）。这里只放 LoRA，不是完整模型，也不是环境工程专家模型。数字以 `output/experiment_summary.json` 为准。

| Adapter | 状态 |
| --- | --- |
| `envif-qwen2.5-1.5b-sft/` | 已训。污水+监测，798 条，2 epoch。勿覆盖 |
| `envif-qwen2.5-1.5b-sft-v2/` | 已训。含大气污染，1020 条，累计 3 epoch。现场演示默认用这个 |
| `envif-qwen2.5-1.5b-dpo/` | 已训，从 SFT v1 续训。Bench 未超过 SFT，不要当默认 |

加载需要 `adapter_config.json` 和 `adapter_model.safetensors`。用法见仓库根目录 `README.md`。
