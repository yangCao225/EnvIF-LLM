# 面向污水处理与环境监测的 EnvIF 操作说明

项目全称：面向污水处理与环境监测的 AutoIF 环境工程大模型指令遵循优化系统

仓库：https://github.com/yangCao225/EnvIF-LLM

代码目录：`AutoIF-LLM/`

实验数字唯一来源：`output/experiment_summary.json`。报告、PPT、本说明里的条数、Loss、Bench 指标只许抄这一份。未跑通的字段保持空白或写「未跑」，禁止抄旧材料里互相矛盾的 37 / 406 条，也禁止预先写「量化只掉 2%～3%」。

本文替换已删除的《AutoIF 环境工程操作指南》。旧指南按 AutoDL + 7B 教师全流程来写。那条路径代码还在，但本机没有 7B / vLLM / LlamaFactory，**不能在这台 6GB 笔记本上跑通教师九步**。当前可复现的是：本机 1.5B LoRA + EnvIF-TLR 离线数据 + EnvIF-Bench；入口为 `python scripts/run_local_pipeline.py`。

---

## 0. 先读这一页

### 0.1 这个项目做什么

本项目不是把通用问答模型换成环保词汇，而是：

1. 用可程序验证的环境工程种子指令，接在 AutoIF 的可执行反馈上；
2. 用领域查询集（三条主线：污水处理、环境监测、大气污染含硫/含氮烟气），而不是生活 / 编程 / 法律题；
3. 用「格式 + 带单位计算 + 知识边界」三层验证构造 SFT / DPO；
4. 在 Qwen2.5-1.5B-Instruct 上训 LoRA，评测结构化指令遵循，而不是宣称模型已经成为环境工程专家。

方法增量名为 **EnvIF-TLR**（Typed Layered Reward）：计算层按单位匹配量，未给标准时禁止编造限值和虚假确定性，并按错误类型构造难负样本。

### 0.2 这个项目不做什么

- 不宣称训练出「环境工程专家大模型」。
- 不把 Loss 下降当成效果证明。DPO last-step loss 为 0 尤其不能当成功。
- 不在未提供标准名称、适用对象、年份时让模型编造排放限值。
- 不要再用法律 / 医疗 / 金融当主演示路径。

### 0.3 两条路径，不要混着写

| 路径 | 现状 | 适用 |
| --- | --- | --- |
| 本机 6GB GPU：1.5B + LoRA + EnvIF-TLR | CPU 全流程已跑通；LoRA 与 Bench 已有数字。冒烟若报 1455 是内存不够，不是脚本没写 | 开源复现、答辩演示 |
| AutoDL：7B 教师 + vLLM + LlamaFactory | **本机跑不了**。无 7B 权重、无 vLLM、无 LlamaFactory。`query_rft` 保持 0 | 需要 A800/A100 40GB+ 再跑 |

答辩和 README 以本机路径为准。AutoDL 命令见第 6 节，不要把未跑的 Loss 写进表格。

### 0.4 当前规模（抄自 summary，生成时刻见该文件）

| 项 | 数量 |
| --- | ---: |
| 种子指令 | 52 |
| 训练查询 | 1020（其中大气污染计算/工艺/诊断约 302 条，加烟气监测后大气为第三条主线） |
| 独立测试查询 | 389 |
| EnvIF-Bench | 79（75 条可自动打分；`air_pollution` 任务 7 条） |
| 离线 SFT | 1020 |
| 离线 DPO 对 | 2667 |
| 已发布 SFT LoRA | 在更早一版 798 条上训，2 epoch；**尚未用本版大气主线重训** |
| 已发布 DPO LoRA | 400 对、1 epoch，从 SFT adapter 接着训 |

离线 SFT / DPO 条数来自 EnvIF-TLR 构造，**不是**教师 7B 蒸馏，也**不是** LoRA Loss。上表以 `output/experiment_summary.json` 为准。

---

## 1. 获取代码与环境

### 1.1 克隆

```bash
git clone https://github.com/yangCao225/EnvIF-LLM.git
cd EnvIF-LLM
```

若你本地文件夹名仍是 `AutoIF-LLM`，在该目录操作即可。完整 1.5B / 7B 基座不进仓库。

### 1.2 本机依赖

- Windows 10/11 或 Linux，Python 3.10+（本机实测 Anaconda + CUDA）
- 无 GPU：可跑测试、离线构造、Bench oracle
- 有 GPU：RTX 3060 Laptop 6GB 可训 1.5B LoRA（8bit 或 fp16）
- Hugging Face 镜像（国内）：`HF_ENDPOINT=https://hf-mirror.com`

基座下载到 `models/student/`（已 gitignore）。不要把完整权重推进 GitHub。

### 1.3 开源里有什么、没有什么

| 有 | 没有 |
| --- | --- |
| 查询集、Bench、验证器、离线 SFT/DPO 数据 | 完整 1.5B / 7B 权重 |
| SFT / DPO LoRA（各约 35 MB） | 教师蒸馏后的 SFT 条数 |
| CPU 测试与 GitHub Actions | 量化部署实测数字 |

---

## 2. 无 GPU：先确认仓库能跑

在项目根目录：

```bash
python tests/test_env_validators.py
python tests/test_envif_tlr.py
python tests/test_envif_bench.py
python scripts/demo_env_autoif.py
python scripts/eval_envif_bench.py --oracle --tag oracle
python scripts/construct_envif_offline.py
python scripts/ablate_envif_layers.py
python scripts/build_experiment_summary.py
```

应看到：验证器测试 OK；oracle 的 prompt / constraint 准确率为 1.0（说明打分器自洽，不是模型能力）；三层奖励把 chosen 排在 typed rejected 之上的 win rate 为 1.0（见 `output/envif_ablation.json`）。

COD 口头例题（介绍项目时建议现场算）：

- 水量 10000 m³/d，进水 COD 300 mg/L，出水 50 mg/L
- 去除率 (300−50)/300 = 83.33%
- 日去除负荷 10000×(300−50)×0.001 = 2500 kg/d

只给结论、算成 50%、写成 2500 t/d、编造 GB 18918 限值，都应被三层验证打回。

---

## 3. 本机 6GB：训练、推理、评测（主路径）

### 3.1 基座

首次训练会把 `Qwen/Qwen2.5-1.5B-Instruct` 下到 `models/student/`。也可事先用 ModelScope / hf-mirror 放好，目录里要有 `config.json`。

### 3.2 训 SFT LoRA（已发布的 v1）

```bash
python scripts/train_envif_lora.py
```

默认写入 `adapters/envif-qwen2.5-1.5b-sft/`，**不要随便改 `--out` 覆盖它**。本机记录：r=8，α=16，2 epoch，lr=5e-5，max_len=1024，8bit；mean train loss 约 0.524，last-step 约 0.028。last-step 很低说明模板过拟合，不是专家能力。

内存紧张（Windows 页面文件错误 1455）时：

```bash
python scripts/train_envif_lora.py --out adapters/envif-qwen2.5-1.5b-sft-v2 --epochs 1 --fp16 --max-len 512
```

先关掉占内存的聊天 / 浏览器。v2 必须换输出目录，保留 v1 作对照。

### 3.3 推理 EnvIF-Bench

```bash
python scripts/infer_envif.py --source bench --base ./models/student --fp16
python scripts/infer_envif.py --source bench --base ./models/student --adapter ./adapters/envif-qwen2.5-1.5b-sft --fp16
python scripts/infer_envif.py --source bench --base ./models/student --adapter ./adapters/envif-qwen2.5-1.5b-dpo --fp16
```

默认 `max_new_tokens=384`。6GB 上请用 `--fp16`，不要默认 8bit 解码（会极慢）。

### 3.4 打分并写回 summary

```bash
python scripts/eval_envif_bench.py --pred-file eval_results/bench_base_preds.jsonl --tag base
python scripts/eval_envif_bench.py --pred-file eval_results/bench_sft_preds.jsonl --tag sft
python scripts/eval_envif_bench.py --pred-file eval_results/bench_sft_dpo_preds.jsonl --tag sft_dpo
python scripts/build_experiment_summary.py
```

领域测试集（309 条）脚本是 `scripts/eval_env_engineering.py`。基座 / SFT / DPO 在该测试集上的 summary 目前仍为 null，未跑就不要填。

### 3.5 DPO（已发布的 v1）

```bash
python scripts/train_envif_dpo.py
```

默认 400 对、1 epoch、从 SFT adapter 继续，写入 `adapters/envif-qwen2.5-1.5b-dpo/`。mean loss 约 7.4e-6，last-step 为 0，rewards/accuracies 全程 1.0：负样本当时太容易分开，**不代表变强**。

从 SFT v2 训 DPO v2（不覆盖 v1）：

```bash
python scripts/train_envif_dpo.py --sft-adapter adapters/envif-qwen2.5-1.5b-sft-v2 --out adapters/envif-qwen2.5-1.5b-dpo-v2 --max-pairs 400
```

**部署和演示默认用 SFT v1，不要默认加载 DPO。** Bench 上 DPO 低于 SFT。

### 3.6 本机超参（已发布 adapter，不是 YAML 里的 7B 设定）

| 项 | SFT v1 | DPO v1 |
| --- | --- | --- |
| 基座 | Qwen2.5-1.5B-Instruct | 同左，从 SFT LoRA 接着训 |
| 学习率 | 5e-5 | 5e-6 |
| 轮数 | 2 | 1 |
| LoRA rank / alpha | 8 / 16 | 8 / 16 |
| 序列长度 | 1024 | 512 |
| 量化加载 | 8bit | 8bit |
| 有效 batch | 1 × accum 8 | 1 × accum 8 |

YAML `configs/pipeline_config.yaml` 里的 3 epoch / 2048 / batch 4 是给 AutoDL + LlamaFactory 预留的，**不要和上表混写成同一组实验**。

---

## 4. 数据与 EnvIF-TLR

### 4.1 关键文件

| 路径 | 作用 |
| --- | --- |
| `sample_data/seed_instruction_环境工程.txt` | 可程序验证的种子约束 |
| `data/environmental_engineering_queries.jsonl` | 训练查询，含 category / gold |
| `data/environmental_engineering_test.jsonl` | 独立测试查询，与训练 query 无重叠 |
| `data/envif_bench/envif_bench.jsonl` | 指令遵循评测，按 content / format / numerical / domain / multi 标注 |
| `output/IF_sft_data.json` | 离线 SFT |
| `output/dpo_pairs.jsonl` | 离线 DPO（ShareGPT：conversations / chosen / rejected） |
| `code/env_validators.py` | 三层验证 |
| `code/envif_tlr.py` | chosen 模板 + 分类型负样本 + 层次分 |
| `scripts/construct_envif_offline.py` | 无教师时构造 SFT/DPO |

覆盖生成查询（会改 data 文件，慎用）：

```bash
python scripts/generate_env_queries.py
```

场景目标比例：污水处理 38%，监测 18%，大气 22%，固废 9%，噪声 4%，环评与清洁生产 9%。实际条数以 `experiment_summary.json` 为准。

### 4.2 三层分别查什么

1. **格式**：标题、表格、单位、字数、流程箭头、「信息不足」前缀、两位小数、公式
2. **计算**：去除率、污染负荷（kg/d 与 t/d 不能靠裸数字蒙对）、HRT（须 ×24）、数量级检查
3. **依据**：未给标准不得写 GB 限值；禁止「已经达标 / 可以直接排放」；缺参须声明信息不足

### 4.3 当前离线构造在防什么

v1 DPO 负样本过易（例如「该厂运行正常，无需按格式作答」），模型会输出格式完整但算错的数。数据已重生为更长 chosen 和难近误：

- 水量用 `Q`，池容才用 `V`；负荷必须写 `×0.001` 和 `kg/d`
- HRT 必须 `V/Q×24`，不能漏乘 24
- 难负样本：漏乘 0.001、HRT 漏乘 24、把流量写成 `V=` 再乱套 BOD、把 kg/d 写成 t/d、虚构标准、虚假确定性
- 构造时按错误类型去重，少用 format_error

重生：

```bash
python scripts/construct_envif_offline.py
```

这只更新 `output/` 里的 json，**不会**自动覆盖已发布 LoRA。要用新数据必须另存 adapter 目录再训。

---

## 5. 已测 EnvIF-Bench（诚实读表）

72 条，68 条可自动打分。数字来自 `eval_results/envif_bench_*_summary.json`，已汇总进 `experiment_summary.json`。

| 指标 | 基座 1.5B | SFT LoRA | SFT+DPO |
| --- | ---: | ---: | ---: |
| prompt_accuracy | 0.3676 | 0.3529 | 0.2353 |
| constraint_accuracy | 0.5407 | 0.6453 | 0.6221 |
| numerical 族 | 0.0909 | 0.3636 | 0.1818 |
| content 族 | 0.3333 | 0.1667 | 0.1667 |
| multi 族 | 0.4091 | 0.1364 | 0.0455 |

怎么说：

- SFT 主要抬的是**带单位计算的格式**（公式、两位小数、数量级），不是整题通过率；prompt_accuracy 甚至略低于基座。
- 基座在解释类 / 多约束题上仍更好。SFT 把输出收成短模板后，content 和 multi 下降。
- 离线 DPO **没有超过 SFT**：整题通过率和 numerical 都更差。部分回答看起来完整，数字是错的（例如负荷混进 BOD/COD 公式）。
- 当前最好用的计算格式检查点是 **SFT v1**，不是 DPO。

oracle 准确率为 1.0 只说明 verifier 与 reference 一致，不能当作成绩。

---

## 6. 全流程怎么跑

### 6.1 本机（当前能跑通的）

```bash
python scripts/run_local_pipeline.py
```

等价于 `run_all.sh` 在没有 7B / vLLM 时的自动回退。已经实测：CPU 测试、离线构造 795/1924、LlamaFactory 数据导出、TLR ablation、Bench oracle 均成功。冒烟推理在系统内存不足时会报 Windows 1455，不伪造成功。

309 条领域测试需要先能加载 1.5B。内存够时不要加 `--skip-test-eval`。

### 6.2 AutoDL 7B 教师（本机没有这套环境）

需要 40～80GB 卡、教师 Qwen2.5-7B-Instruct、vLLM、LlamaFactory。项目放数据盘：

```text
/root/autodl-tmp/AutoIF-LLM
```

```bash
bash setup.sh
nohup bash run_all.sh --domain 环境工程 \
  --query-file data/environmental_engineering_queries.jsonl \
  --test-file data/environmental_engineering_test.jsonl \
  > run.log 2>&1 &
tail -f run.log
```

若实例上还没装 vLLM 或没有教师权重，`run_all.sh` 会自动改走 `scripts/run_local_pipeline.py`，**不会假装九步已经跑完**。`query_rft` 等字段有数字之前，不要写进论文表格。

---

## 7. 常见问题

**页面文件太小 / os error 1455**  
Windows 提交内存不够映射约 3GB 的 `model.safetensors`。关掉 QQ / 微信 / 豆包 / 多余 Edge，或 `--fp16 --max-len 512` 换目录训 v2。不要覆盖 v1。

**DPO loss 已经是 0，是不是训好了？**  
不是。负样本太容易时 DPO 没有新信号，Bench 还可能变差。

**SFT last-step loss 接近 0**  
短模板过拟合。看 Bench，不要看这条曲线交差。

**推理极慢**  
不要用 8bit generate。`--fp16` 在 6GB 上可以跑 1.5B。

**GitHub 网页推送失败**  
国内可用 SSH 443：`ssh://git@ssh.github.com:443/yangCao225/EnvIF-LLM.git`。不要把 `models/` 推上去。

**想证明有效，贴什么？**  
EnvIF-Bench 的 constraint / numerical，加上「未给标准不编造限值」的抽检。不要只贴 Loss，也不要说模型已经懂环境工程。

**309 条领域测试有数字吗？**  
summary 里 `eval.base / sft / sft_dpo` 在加载 1.5B 成功并跑完 `run_local_pipeline.py`（不要 `--skip-test-eval`）之前保持 null。本机若报 1455，这一项就还没数字。

---

## 8. 目录（操作时会碰到的）

```text
AutoIF-LLM/
  README.md
  docs/EnvIF环境工程操作说明.md
  EnvIF环境工程操作说明.pdf
  configs/pipeline_config.yaml
  sample_data/seed_instruction_环境工程.txt
  data/environmental_engineering_queries.jsonl
  data/environmental_engineering_test.jsonl
  data/envif_bench/
  code/env_validators.py
  code/envif_tlr.py
  code/1_RFT.py … 9_sft_data_construction.py
  scripts/construct_envif_offline.py
  scripts/train_envif_lora.py
  scripts/train_envif_dpo.py
  scripts/infer_envif.py
  scripts/eval_envif_bench.py
  scripts/build_experiment_summary.py
  adapters/envif-qwen2.5-1.5b-sft/
  adapters/envif-qwen2.5-1.5b-dpo/
  tests/
  output/experiment_summary.json
  eval_results/
  models/          # 本地基座，不入库
```

底层仍是 AutoIF 九步 + LoRA / DPO。改过的是领域数据、带单位验证器和离线数据组装，不是从零写训练框架。

---

## 9. 介绍时可以怎么说

一句话：不是把通用模型换成环保词，而是给 AutoIF 接上环境工程查询集、可执行计算验证器和分类型偏好数据，看污水处理计算、工艺比较、异常诊断和监测报告里的结构化指令遵循有没有变好。

三个必答题：

1. 为什么选污水处理？计算可验证，流程和报告字段固定，契合 AutoIF。
2. 和套 Prompt 有何不同？Prompt 管不住算错和编造标准；这里用程序抽带单位的数字对 gold，DPO 再拉开错误单位和虚构限值。
3. 模型懂环境工程了吗？没有。1.5B LoRA 学的是按约束把计算和格式写完整；专业事实仍要标准文本或人工抽检。SFT 抬了计算格式，DPO v1 没有再抬，内容类题基座仍更好。

---

## 10. 答辩前检查清单

- [ ] 数字只引用 `output/experiment_summary.json`
- [ ] 说清离线 795 / 1924 来自 EnvIF-TLR，不是 7B 教师蒸馏
- [ ] 已发布 LoRA 是 1.5B，不是 7B
- [ ] Bench 表包含基座 / SFT / DPO，且承认 DPO 未超过 SFT
- [ ] 演示默认加载 SFT LoRA
- [ ] 未跑的量化、309 条评测、教师全流程保持「未跑」
- [ ] 未把 `models/` 完整基座提交到 GitHub
