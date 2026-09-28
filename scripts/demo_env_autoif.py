#!/usr/bin/env python3
"""不依赖 GPU 的环境工程 AutoIF 演示：同一道题走三层验证 + 评测指标。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from env_validators import evaluate_response, gas_load_kg_d, pollution_load_kg_d, removal_rate, score_response
from env_seed_validators import funcs_for_instruction
from utils import compile_eval_func


def banner(title: str):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def show_eval(name: str, result: dict):
    calc = result["calculation"]["checks"]
    know = result["knowledge"]["checks"]
    fmt = result["format"]["checks"]
    print(f"\n[{name}] 总分={score_response(result):.2f}  三层通过={result['passed']}")
    print(f"  格式: {result['format']['passed']}  {fmt or '（本题无额外格式约束）'}")
    print(f"  计算: {result['calculation']['passed']}  {calc}")
    print(f"  依据: {result['knowledge']['passed']}  {know}")


def main():
    banner("1. 项目现在在做什么")
    print("输入: 水质参数 / 烟气工况 / 工艺条件 / 监测数据")
    print("输出: 带单位、公式、过程、假设的结构化回答")
    print("主线: 污水处理、环境监测、大气污染（含硫含氮烟气）")
    print("边界: 没给标准时不得编造限值")
    print("验证: 格式 → 计算 → 专业依据")

    banner("2. 种子指令（可程序验证，不是“请分析污染”）")
    seed = ROOT / "sample_data" / "seed_instruction_环境工程.txt"
    lines = [x.strip() for x in seed.read_text(encoding="utf-8").splitlines() if x.strip()]
    for i, line in enumerate(lines[:6], 1):
        print(f"  {i}. {line}")
    print(f"  ... 共 {len(lines)} 条")

    banner("3. 领域查询（训练集里抽 3 条）")
    queries = []
    with (ROOT / "data" / "environmental_engineering_queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    for row in queries[:3]:
        print(f"  [{row['category']}] {row['query'][:90]}")
        if row.get("gold"):
            g = row["gold"]
            print(f"      gold: 去除率={g.get('removal_rate_pct')}  负荷={g.get('load_kg_d')}")

    q = "某污水厂设计水量为10000 m³/d，进水COD为300 mg/L，出水COD为50 mg/L，请计算COD去除率和每日去除负荷。"
    gold = {
        "params": {"Q": 10000, "Cin": 300, "Cout": 50},
        "removal_rate_pct": round(removal_rate(300, 50), 2),
        "load_kg_d": round(pollution_load_kg_d(10000, 250), 2),
        "require_two_decimals": True,
        "require_magnitude_check": True,
        "knowledge_constraints": {"forbid_fabricated_standard": True},
    }

    banner("4. 同一道 COD 题：参考答案怎么算")
    print(q)
    print(f"  去除率 = (300-50)/300 × 100% = {gold['removal_rate_pct']}%")
    print(f"  负荷   = 10000 × (300-50) × 0.001 = {gold['load_kg_d']} kg/d")

    answers = {
        "chosen 正确过程": (
            "已知条件：Q=10000 m³/d，进水COD=300 mg/L，出水COD=50 mg/L。\n"
            "去除率=(300-50)/300×100%=83.33%。\n"
            "每日去除负荷=10000×(300-50)×0.001=2500.00 kg/d。\n"
            "数量级合理性检查：日去除 2.5 t COD，对万吨级污水厂合理。未提供排放标准，需核实现行标准。"
        ),
        "rejected 只给结论": "该厂 COD 去除效果较好，已经达标。",
        "rejected 算错": (
            "进水300 mg/L，出水50 mg/L，去除率=50.00%。负荷=100.00 kg/d。数量级合理。"
        ),
        "rejected 单位错误": (
            "去除率=(300-50)/300=83.33%。每日去除负荷=2500.00 t/d。数量级检查通过。"
        ),
        "rejected 虚构标准": (
            "去除率83.33%，负荷2500 kg/d。按 GB 18918-2002 一级A限值 COD 为 50 mg/L，已经达标。"
        ),
    }

    banner("5. 三层验证器如何区分 chosen / rejected")
    results = {}
    for name, text in answers.items():
        results[name] = evaluate_response(text, query=q, gold=gold)
        show_eval(name, results[name])

    banner("5b. 大气污染例题：烟气负荷不得套用废水公式")
    q_air = "临港燃煤热电烟气量Qg=6850 m³/h，进口SO2=115 mg/m³，出口SO2=20 mg/m³。请计算去除率和每日去除负荷。"
    gold_air = {
        "params": {"Qg": 6850, "Cin": 115, "Cout": 20},
        "removal_rate_pct": 82.61,
        "load_kg_d": round(gas_load_kg_d(6850, 95), 2),
        "require_two_decimals": True,
        "require_magnitude_check": True,
        "knowledge_constraints": {"forbid_fabricated_standard": True, "process_family": "so2"},
    }
    print(q_air)
    print(f"  去除率 = (115-20)/115 × 100% = {gold_air['removal_rate_pct']}%")
    print(f"  负荷   = 6850 × 95 × 24 × 10^{-6} = {gold_air['load_kg_d']} kg/d")
    air_ok = (
        "符号约定：烟气量用 Qg（m³/h）。已知进口 C_in=115 mg/m³、出口 C_out=20 mg/m³、烟气量 Qg=6850 m³/h。"
        "去除率=(115-20)/115×100%=82.61%。"
        "每日去除负荷=Qg×(C_in-C_out)×24×10^{-6}=6850×(115-20)×24×10^{-6}=15.62 kg/d。"
        "该负荷为含硫烟气脱硫削减量。数量级合理性检查通过。未提供排放标准，需核实现行标准。"
    )
    air_bad = (
        "把烟气量当成水量。负荷=Q×(C_in-C_out)×0.001=6850×(115-20)×0.001=650.75 kg/d。已经达标。"
    )
    show_eval("烟气 chosen", evaluate_response(air_ok, query=q_air, gold=gold_air))
    show_eval("套用废水公式", evaluate_response(air_bad, query=q_air, gold=gold_air))

    banner("6. 缺参数题：必须以“信息不足”开头")
    q2 = "监测数据仅给出 COD=300 mg/L，未给出水量和执行标准，请判断是否达标。"
    gold2 = {
        "format_constraints": {"must_start_insufficient": True},
        "knowledge_constraints": {"require_insufficient": True, "forbid_fabricated_standard": True},
    }
    a_ok = "信息不足。缺少水量、采样时间、检测方法及排放标准年份/适用对象，不能给出确定结论，需核实现行标准。"
    a_bad = "按照一级A标准已经达标，建议直接排放。"
    show_eval("正确拒答", evaluate_response(a_ok, query=q2, gold=gold2))
    show_eval("编造限值", evaluate_response(a_bad, query=q2, gold=gold2))

    banner("7. 种子指令自带的 Python 验证函数")
    inst = "计算污染物去除率时，必须依次给出进水浓度、出水浓度、计算公式和百分比结果。"
    fn = compile_eval_func(funcs_for_instruction(inst)[0])
    print("指令:", inst)
    print("  正确回答 →", fn(answers["chosen 正确过程"]))
    print("  只给结论 →", fn(answers["rejected 只给结论"]))

    # 写出评测预测文件，供官方评测脚本使用
    out_dir = ROOT / "eval_results"
    out_dir.mkdir(exist_ok=True)
    pred_path = out_dir / "demo_preds.jsonl"
    test_path = ROOT / "data" / "environmental_engineering_test.jsonl"
    preds = []
    with test_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 12:
                break
            item = json.loads(line)
            gold_item = item.get("gold") or {}
            params = gold_item.get("params") or {}
            if gold_item.get("removal_rate_pct") is not None:
                cin, cout = params.get("Cin"), params.get("Cout")
                qv = params.get("Q")
                qg = params.get("Qg")
                rate = gold_item["removal_rate_pct"]
                load = gold_item.get("load_kg_d")
                if qg is not None:
                    resp = (
                        f"已知进口{cin} mg/m³、出口{cout} mg/m³、烟气量Qg={qg} m³/h。"
                        f"去除率=({cin}-{cout})/{cin}×100%={rate:.2f}%。"
                    )
                    if load is not None:
                        resp += f"每日去除负荷={qg}×({cin}-{cout})×24×10^{-6}={load:.2f} kg/d。"
                    resp += "数量级合理性检查通过。未提供标准，需核实现行标准。"
                else:
                    resp = (
                        f"已知进水{cin} mg/L、出水{cout} mg/L"
                        + (f"、水量{qv} m³/d。" if qv else "。")
                        + f"去除率=({cin}-{cout})/{cin}×100%={rate:.2f}%。"
                    )
                    if load is not None:
                        resp += f" 每日去除负荷={qv}×({cin}-{cout})×0.001={load:.2f} kg/d。"
                    resp += " 数量级合理性检查：结果与水量、浓度差匹配。未提供标准，需核实现行标准。"
            elif gold_item.get("o2_corrected") is not None:
                c, om, os = params.get("C"), params.get("O2m"), params.get("O2s")
                corr = gold_item["o2_corrected"]
                resp = (
                    f"C'={c}×(21−{os})/(21−{om})={corr:.2f} mg/m³。实测 {c} mg/m³。"
                    "未提供排放标准，需核实现行标准。"
                )
            elif gold_item.get("format_constraints", {}).get("must_start_insufficient"):
                resp = "信息不足。列出缺失参数：水量、采样时间、执行标准名称及年份。不得给出确定结论。"
            elif gold_item.get("format_constraints", {}).get("exactly_n_causes"):
                resp = (
                    "原因1：丝状菌膨胀。证据：SVI升高。复核方法：镜检丝状菌。\n"
                    "原因2：二沉池负荷过高。证据：出水SS增加。复核方法：核算表面负荷。\n"
                    "原因3：营养比失调。证据：污泥结构松散。复核方法：复核N、P投加。"
                )
            else:
                resp = (
                    "设计输入：按题给条件。关键假设：未给标准时不编造限值。\n"
                    "计算过程：先列已知量再代入公式。推荐方案：结合水质选择。风险点：二次污染与运行安全。需核实现行标准。"
                )
            preds.append({"query": item["query"], "response": resp, "category": item.get("category")})
    with pred_path.open("w", encoding="utf-8") as f:
        for row in preds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n已写出 {pred_path}（{len(preds)} 条格式样例，不是模型预测）")
    print("不要拿这 12 条去评 389 条测试集。模型成绩看 EnvIF-Bench：")
    print("  python scripts/eval_envif_bench.py --pred-file eval_results/bench_sft_v2_preds.jsonl --tag sft_v2")


if __name__ == "__main__":
    main()
