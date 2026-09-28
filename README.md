# 面向污水处理、环境监测与大气污染的 AutoIF 环境工程指令遵循优化

**Open-source:** EnvIF 数据集 · EnvIF-Bench · CPU 演示 · **SFT LoRA v1 / v2** · **DPO LoRA v1**.  
完整 1.5B / 7B 权重不在仓库里。数字只引用 `output/experiment_summary.json`（`generated_at`: **2026-09-28T14:31:10**）。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![cpu-tests](https://github.com/yangCao225/EnvIF-LLM/actions/workflows/cpu-tests.yml/badge.svg)](https://github.com/yangCao225/EnvIF-LLM/actions/workflows/cpu-tests.yml)

仓库：https://github.com/yangCao225/EnvIF-LLM

本项目**不是**环境工程专家大模型。它用可执行验证器约束 Qwen2.5-1.5B：污水计算、监测报告、含硫含氮烟气题必须写对单位、公式和知识边界。方法增量是 **EnvIF-TLR**（Typed Layered Reward），叠在 [AutoIF](https://arxiv.org/abs/2406.13542) 的可执行反馈上，不另起预训练。

三条并列主线：**污水处理**、**环境监测**、**大气污染（SO2 / NOx / 基准氧 / FGD / SCR）**。固废、噪声、环评是扩展。大气污染不是污水题的边角料。

离线 SFT / DPO 条数来自 TLR 构造，不是 7B 教师蒸馏，也不是 LoRA loss。DPO last-step loss 为 0 不代表变强。

---

## 怎么用

```bash
git clone https://github.com/yangCao225/EnvIF-LLM.git
cd EnvIF-LLM
python scripts/demo_env_autoif.py
```

这条不需要 GPU。它会走同一套三层验证器，现场区分对错：

| 题 | 金标准 | 会被打回 |
| --- | --- | --- |
| 水量 10000 m³/d，COD 300→50 mg/L | 去除率 **83.33%**，负荷 **2500.00 kg/d**（`Q×ΔC×0.001`） | 只给结论、写成 t/d、编造 GB 18918 |
| 烟气 6850 m³/h，SO2 115→20 mg/m³ | 负荷 **15.62 kg/d**（`Qg×ΔC×24×10^{-6}`） | 套用废水 `×0.001` |
| 只给 COD、没有水量和标准 | 必须以「信息不足」开头 | 说已经达标 |

演示脚本写出的 `eval_results/demo_preds.jsonl` 只是十几条格式样例。**不要拿它去评 389 条测试集**，那个分数不是模型成绩。

有显存时看 SFT v2 现场答题（COD / 脱硫 / 脱硝）：

```bash
python scripts/demo_envif_qa.py
```

6GB 笔记本常因页面文件不足（Windows **1455**）或 CUDA 非法指令中断。打不开时直接看已经生成的答案：`eval_results/bench_sft_v2_preds.jsonl`。脱硫优先石灰石-石膏湿法，脱硝优先 SCR。

重打已有预测：

```bash
python scripts/eval_envif_bench.py --pred-file eval_results/bench_sft_v2_preds.jsonl --tag sft_v2
python scripts/build_experiment_summary.py
```

国内拉基座：`HF_ENDPOINT=https://hf-mirror.com`，放到 gitignored 的 `models/student/`。

---

## 方法（EnvIF-TLR）

1. **计算层带单位**：`2500 t/d` 不能蒙成 `2500 kg/d`。污水 `Q×C×0.001`；烟气 `Qg×C×24×10^{-6}`，禁止套废水公式。基准氧必须写出 `C'=C×(21-O2,s)/(21-O2,m)`，任意乘号不算过。
2. **知识边界**：题目没给标准，禁止编造限值和「已经达标」。
3. **按错误类型造 DPO 负样本**：`calc_error` / `unit_error` / `fabricated_standard` / `false_certainty`。
4. **层次 ablation**：三层同时启用时，chosen 相对 typed rejected 的 ranking win rate 为 **1.0**（`output/envif_ablation.json`）。

Oracle 的 prompt / constraint 准确率是 **1.0**，只说明打分器自洽。

---

## 数据规模

| 项 | 数量 |
| --- | ---: |
| 种子指令 | 52 |
| 训练查询 | 1020 |
| 独立测试 | 389 |
| 离线 SFT | 1020 |
| 离线 DPO 对 | 2667 |
| EnvIF-Bench | 79（75 可自动打分） |
| 7B 教师 `query_rft` | 0 |

训练查询里的大气类别（`categories.train`）：`air_calculation` **180**、`air_process` **72**、`air_diagnosis` **50**，合计 **302**。测试集对应 **64 / 24 / 20**，合计 **108**。SFT v1 用的是更早的 798 条，不含这条大气主线；SFT v2 才用满 1020 条。

气账：

1. 烟气负荷：`Qg(m³/h)×C(mg/m³)×24×10^{-6} = kg/d`
2. 基准氧：只有题目给出参考氧含量才折算；没给排放标准不得判达标。

---

## 评测

**72 题（无大气任务）和 79 题（含 7 道 `air_pollution`）不能直接横比。** 309 条领域测试 `eval.base / sft / sft_dpo` 仍是 **null**。量化体积和延迟是 **null**，不要写「只掉 2%～3%」。

EnvIF-Bench v1（72 题 / 68 可自动打分）：

| 指标 | 基座 1.5B | SFT v1 | SFT+DPO v1 |
| --- | ---: | ---: | ---: |
| prompt_accuracy | 0.3676 | 0.3529 | 0.2353 |
| constraint_accuracy | 0.5407 | 0.6453 | 0.6221 |
| numerical 族 | 0.0909 | 0.3636 | 0.1818 |

SFT v1 抬的是带单位计算的格式，不是整题通过率。DPO 的 Bench 没有超过 SFT。

EnvIF-Bench SFT v2（79 题 / 75 可自动打分）：

| 指标 | SFT v2 |
| --- | ---: |
| prompt_accuracy | 0.3733 |
| constraint_accuracy | 0.6684 |
| numerical 族 | 0.2000 |
| numerical.value | 0.3636 |
| air_pollution | 0.4286（3/7） |

同一套 **7** 道大气题（`eval.envif_bench_*_air`，2026-09-26 本机 fp16；SFT v2 取自上面的 `by_task`）：

| 模型 | 整题通过 | numerical.formula | numerical.value |
| --- | ---: | ---: | ---: |
| 基座 | 0.1429（1/7） | 0.00 | 0.1667 |
| SFT v1 | 0.1429（1/7） | 0.50 | 0.00 |
| SFT+DPO v1 | 0.1429（1/7） | 0.25 | 0.00 |
| SFT v2 | **0.4286（3/7）** | — | — |

v2 能排石灰石-石膏湿法 / SCR。烟气负荷和基准氧算术仍经常错。不要默认用 DPO。

---

## 本机权重

`configs/pipeline_config.yaml` 里的 3 epoch / 2048 / batch 4 是 AutoDL + LlamaFactory 的预留，不要和下面这组混写。

| 检查点 | 数据 | 设定 | mean / last loss |
| --- | --- | --- | ---: |
| `adapters/envif-qwen2.5-1.5b-sft` | 污水+监测，798 条 | 2 epoch，max_len 1024，8bit | 0.524 / 0.0278 |
| `adapters/envif-qwen2.5-1.5b-sft-v2` | 三主线，1020 条 | 累计 3 epoch，max_len 768，fp16 | 0.085 / 0.0184 |
| `adapters/envif-qwen2.5-1.5b-dpo` | 从 v1 续训，400 对 | 1 epoch，max_len 512 | ≈0 / 0 |

不要覆盖 v1 和 v2。DPO v2 未训。算错的 SFT 上不要再叠 DPO。

可选续训请写到新目录，例如：

```bash
python scripts/train_envif_lora.py --out adapters/envif-qwen2.5-1.5b-sft-v3 --resume adapters/envif-qwen2.5-1.5b-sft-v2 --epochs 2 --fp16 --max-len 768 --lr 2e-5
```

---

## 两条路径

| 路径 | 现状 | 入口 |
| --- | --- | --- |
| 本机 1.5B + EnvIF-TLR | CPU 测试、构造、ablation、oracle 已跑通；LoRA 与 Bench 已有数字 | `python scripts/run_local_pipeline.py --skip-gpu` |
| AutoDL 7B 教师 + vLLM + LlamaFactory | 本机未跑。无教师权重、无 vLLM | `bash run_all.sh`（约 40GB+ GPU） |

答辩和开源以本机路径为准。

无 GPU 复现验证器与数据：

```bash
python tests/test_env_validators.py
python tests/test_envif_tlr.py
python tests/test_envif_bench.py
python scripts/generate_env_queries.py
python scripts/construct_envif_offline.py
python scripts/ablate_envif_layers.py
python scripts/eval_envif_bench.py --oracle --tag oracle
```

---

## 目录

```
code/env_validators.py              # 三层验证（污水负荷、烟气负荷、基准氧）
code/envif_tlr.py                   # Typed Layered Reward
code/envif_bench.py                 # Bench verifier
scripts/air_domain.py               # 大气查询
scripts/demo_env_autoif.py          # CPU 演示
scripts/demo_envif_qa.py            # GPU 三题
scripts/infer_envif.py
scripts/train_envif_lora.py
data/envif_bench/                   # 79 题；envif_bench_air.jsonl 为其中 7 题
adapters/envif-qwen2.5-1.5b-sft/
adapters/envif-qwen2.5-1.5b-sft-v2/
adapters/envif-qwen2.5-1.5b-dpo/
output/experiment_summary.json      # 唯一数字来源
docs/EnvIF环境工程操作说明.md
```

---

## 诚实清单

| 项 | 状态 |
| --- | --- |
| SFT v1 / v2 | 已训、已评；v2 的 FGD/SCR 排序可用，气账和基准氧不稳 |
| DPO v1 | 已训；7 道大气题和 72 题 Bench 都没有超过对应 SFT |
| DPO v2、7B 教师九步、量化、309 条领域测试 | 未完成，summary 中为 null 或 0 |
| 现场加载 1.5B | 6GB 上不稳定；演示优先用 CPU 脚本或已有 `bench_sft_v2_preds.jsonl` |
