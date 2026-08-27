# EnvIF-Bench：环境工程领域指令遵循评测集

面向污水处理、环境监测与污染控制的**指令遵循**评测，而不是开放知识问答。设计对齐 IFEval / FollowBench：先给可执行约束，再按约束类型出分。

数据文件：`envif_bench.jsonl`。规模以 `envif_bench_meta.json` 为准。

## 任务覆盖

| task | 含义 |
| --- | --- |
| `wastewater_treatment` | 污水处理工艺、流程、诊断 |
| `environmental_monitoring` | 监测报告、点位、达标判断 |
| `pollution_control` | 气/渣/声等污染控制 |
| `environmental_calculation` | 去除率、负荷、HRT、F/M 等 |
| `safety_constraint` | 药剂、应急、二次污染与安全 |
| `format_constraint` | 以表格、标题、步骤等格式为主 |
| `multi_step_reasoning` | 计算 + 流程 + 知识边界等组合 |
| `terminology_explanation` | 专业术语解释 |

## 指令类型（constraint family）

| 类型 | 含义 | 自动 verifier 示例 |
| --- | --- | --- |
| `content` | 必须出现指定内容/术语/字段 | `content.must_include` `content.explain_term` `content.missing_params` |
| `format` | 结构与版式 | `format.table` `format.headers` `format.arrow_process` `format.n_causes` `format.insufficient_prefix` `format.length` `format.n_steps` |
| `numerical` | 数值、单位、公式、小数位 | `numerical.value` `numerical.two_decimals` `numerical.formula` `numerical.magnitude` |
| `domain` | 专业边界 | `domain.no_fabricated_standard` `domain.no_false_certainty` `domain.allowed_units` `domain.safety` `domain.ph_dimensionless` `domain.process_match` |
| `multi` | 同一题含两类及以上约束 | 上述原子约束的合取 |

一条样本有 `constraints[]`。`constraint_family=multi` 表示类型多于一种。`auto_verifiable=false` 的题目没有程序打分，留给人工。

## 指标

- **prompt_accuracy**：该题全部自动约束都通过
- **constraint_accuracy**：原子约束通过率
- 再按 `by_family` / `by_task` / `by_verifier` 分组

## 生成与评测

```bash
python scripts/build_envif_bench.py
python scripts/eval_envif_bench.py --oracle --tag oracle
python scripts/eval_envif_bench.py --pred-file your_preds.jsonl --tag model
```

预测文件每行：`{"id":"envifbench-0001","response":"..."}` 或用 `query` 对齐。

与 `data/environmental_engineering_*.jsonl` 的关系：Bench 是**独立评测集**，生成时会丢掉与训练/测试查询重复的句子。训练请仍用领域查询 JSONL；汇报指令遵循请引用 EnvIF-Bench。
