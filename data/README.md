领域查询与独立测试集由 `python scripts/generate_env_queries.py` 生成。

- `environmental_engineering_queries.jsonl`：训练/合成用查询（可含 `gold`）
- `environmental_engineering_test.jsonl`：合成评测用，与上一文件查询文本无重叠
- `envif_bench/`：**EnvIF-Bench** 指令遵循评测集（按 content/format/numerical/domain/multi 标注，部分带自动 verifier）

字段：`query`, `category`, `difficulty`, `required_knowledge`, `split`, 可选 `gold`。

场景占比目标：污水处理 38%、监测 18%、大气 22%（含硫含氮烟气为大气主线）、固废 9%、噪声 4%、环评清洁生产 9%。扩充时请按同一比例分层，并继续保持训练/测试隔离。
