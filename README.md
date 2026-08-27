# 面向污水处理与环境监测的 AutoIF 环境工程大模型指令遵循优化系统

**Open-source release:** EnvIF dataset · EnvIF-Bench · evaluation scripts · CPU demo.  
LoRA adapter **layout and export script** are included; trained weights are published only after GPU SFT/DPO (see `adapters/README.md`).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

本项目不是简单将通用问答模型更换为环保词汇，而是构建了环境工程领域查询集、可执行工程计算验证器和专业偏好数据，通过 AutoIF、LoRA 与 DPO 提升模型在污水处理计算、工艺比较、异常诊断和监测报告生成中的结构化指令遵循能力。

> 基于论文 [Self-play with Execution Feedback](https://arxiv.org/abs/2406.13542) (ICLR 2025 Spotlight)  
> 主线：**污水处理与水污染控制**；第二场景：**环境监测**；废气 / 固废 / 噪声为扩展。

## 开源内容

| 组件 | 路径 | 说明 |
| --- | --- | --- |
| EnvIF 数据集 | `data/*.jsonl`，`output/IF_sft_data.json`，`output/dpo_pairs.jsonl` | 领域查询 + 离线 SFT/DPO（来源标明 `envif_tlr_offline`） |
| EnvIF-Bench | `data/envif_bench/` | 按 content / format / numerical / domain / multi 标注的指令遵循评测 |
| Evaluation | `scripts/eval_envif_bench.py`，`scripts/eval_env_engineering.py` | Bench 与领域测试评测 |
| Demo | `scripts/demo_env_autoif.py` | 无 GPU：三层验证 + chosen/rejected |
| LoRA adapter | `adapters/`，`scripts/export_lora_adapter.py` | 发布目录与导出脚本；权重需训练后放入 |

实验数字一律以 `output/experiment_summary.json` 为准。GPU 上的 SFT/DPO Loss 未跑完前保持 `null`。离线构造的 SFT/DPO 条数可以引用，但必须标明来源是 EnvIF-TLR，不是教师模型蒸馏。

---

## 方法增量（EnvIF-TLR）

本项目不另起预训练算法，而是在 AutoIF 的可执行反馈上增加 **Typed Layered Reward**：

1. **计算层改为带单位的量匹配**：`2500 t/d` 不能靠裸数字蒙成 `2500 kg/d`。
2. **知识边界**：未给标准时禁止编造限值，也禁止「已经达标」这类虚假确定性。
3. **按错误类型构造 DPO 负样本**：`calc_error` / `unit_error` / `fabricated_standard` / `false_certainty`。
4. **层次 ablation**：比较单层与三层奖励能否把 chosen 排在 typed rejected 之上。结果写入 `output/envif_ablation.json`。

不依赖 GPU 的复现：

```bash
python tests/test_env_validators.py
python tests/test_envif_tlr.py
python scripts/generate_env_queries.py
python scripts/construct_envif_offline.py
python scripts/ablate_envif_layers.py
python scripts/build_experiment_summary.py
```

与训练查询集分离的指令遵循评测：

```bash
python scripts/build_envif_bench.py
python scripts/eval_envif_bench.py --oracle --tag oracle
```

详见 `data/envif_bench/README.md`。约束类型为 content / format / numerical / domain / multi。

---

## 任务定义

| 项 | 说明 |
| --- | --- |
| 输入 | 水质参数、污染源信息、工艺条件、监测数据或环境工程案例 |
| 输出 | 污染分析、工程计算、工艺比较、异常诊断、监测报告和治理建议 |
| 强制要求 | 给出单位、公式、计算过程、假设条件、不确定性说明 |
| 知识边界 | 未提供适用标准、地区或年份时，不得自行编造标准限值 |
| 目标 | 优化「结构化专业回答能力」，而不是宣称少量微调就能掌握全部环境工程知识 |

推荐覆盖比例：污水处理 40%，环境监测 20%，大气污染 15%，固废 10%，噪声 5%，环评与清洁生产 10%。

---

## 快速开始

```bash
cd /root/autodl-tmp
unzip AutoIF-LLM.zip && cd AutoIF-LLM
bash setup.sh
bash run_all.sh --domain 环境工程
# 或后台
nohup bash run_all.sh --domain 环境工程 > run.log 2>&1 &
```

指定领域查询集（推荐）：

```bash
bash run_all.sh --domain 环境工程 \
  --query-file data/environmental_engineering_queries.jsonl \
  --test-file data/environmental_engineering_test.jsonl
```

---

## 数据与验证

| 文件 | 当前规模 | 说明 |
| --- | ---: | --- |
| `sample_data/seed_instruction_环境工程.txt` | 见 summary | 可程序验证的格式/计算/知识边界约束 |
| `data/environmental_engineering_queries.jsonl` | 见 `experiment_summary.json` | 领域查询，含 `category` / `gold` |
| `data/envif_bench/` | 见 summary | EnvIF-Bench：按 content/format/numerical/domain/multi 标注的指令遵循评测 |

三层验证（`code/env_validators.py`）：

1. **格式**：标题、表格、单位、字数、流程符号、「信息不足」前缀  
2. **计算**：去除率、污染负荷、HRT 等，允许相对误差  
3. **依据**：不得虚构限值、缺参数时声明不确定性、工艺与污染物匹配

COD 示例的参考结果：

- 去除率 `((300-50)/300 = 83.33%)`
- 日去除负荷 `(10000 × (300-50) × 0.001 = 2500 kg/d)`

本地检查验证器：

```bash
python tests/test_env_validators.py
python scripts/demo_env_autoif.py
```

---

## 已修复的数据链路

| 原问题 | 现处理 |
| --- | --- |
| `--domain 环境工程` 无模板 | 已在 `extended_domains.py` 注册，并支持「污水处理」「环境监测」别名 |
| 查询生成覆盖生活/编程 | 默认读取领域查询集；LLM 兜底也强制环境工程 |
| SFT 读 `gpt-answer` 得到空答案 | `run_all.sh` 使用 `output/IF_sft_data.json`；字段为 `response` |
| DPO 误读 `instruction/query` | 保留 ShareGPT 的 `conversations/chosen/rejected` |
| DPO 丢失环境问题 | human 侧为「环境问题 + 【格式要求】」 |
| 配置未驱动代码 | `code/utils.py` / `run_all.sh` 从 `configs/pipeline_config.yaml` 读阈值与学习率 |
| 覆盖原始种子且不恢复 | 使用独立 `seed_instruction_环境工程.txt`，`trap` 只负责关闭 vLLM |
| DPO 学习率文档打架 | 统一为 YAML 中的 `5e-6` |

---

## 评测

比较四组：**基座 / 环境工程 SFT / SFT+DPO / 量化**。核心指标见 `scripts/eval_env_engineering.py`：指令遵循准确率、计算正确率、单位正确率、字段完整率、虚构标准率、不确定性处理率、量化前后变化、延迟。

```bash
python scripts/eval_env_engineering.py --model-path ./models/student --tag base
python scripts/eval_env_engineering.py --model-path ./models/model_c_sft --tag sft
python scripts/eval_env_engineering.py --model-path ./models/model_merged --tag sft_dpo
python scripts/eval_env_engineering.py --model-path ./models/model_gptq_int4 --tag quantized
python scripts/build_experiment_summary.py
```

量化校准语料改为环境工程文本；**不要预先宣称精度只损失 2%～3%**。

---

## 训练超参（唯一来源：YAML）

| 参数 | SFT | DPO |
| --- | --- | --- |
| 学习率 | `5e-5` | `5e-6` |
| 轮数 | 3 | 2 |
| LoRA rank | 8 | 8 |
| 序列长度 | 2048 | 2048 |

---

## 项目结构（关键增量）

```
code/env_validators.py              # 三层验证器（带单位量 + 错误类型）
code/envif_tlr.py                   # Typed Layered Reward 与分类型负样本
code/env_seed_validators.py         # 种子指令确定性验证函数
scripts/construct_envif_offline.py  # 无教师模型的 SFT/DPO 构造
scripts/ablate_envif_layers.py      # 层次 ablation
scripts/generate_env_queries.py     # 模板 + 运行摘录案例
code/envif_bench.py                 # EnvIF-Bench 约束类型与 verifier
scripts/build_envif_bench.py
scripts/eval_envif_bench.py
data/envif_bench/                   # 指令遵循评测集
```

操作步骤见 `AutoIF环境工程操作指南.pdf`。实验数字只引用 `output/experiment_summary.json`。
