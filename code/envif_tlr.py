"""EnvIF-TLR: Typed Layered Reward，叠在 AutoIF 可执行反馈之上的领域增量。

原 AutoIF 用代码执行验证「格式约束是否满足」。
EnvIF 增加三件事，而不是另起一套预训练：

1. 带单位的符号计算层：kg/d 与 t/d 不能靠裸数字蒙对。
2. 知识边界层：未给标准时禁止编造限值、禁止虚假确定性。
3. 按错误类型构造 DPO 负样本：calc_error / unit_error / fabricated_standard 等。

GPU 上的 LoRA/DPO 仍走原 AutoIF + LlamaFactory；本模块提供
教师模型不可用时的可复现离线数据与层次 ablation。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from env_validators import removal_rate

LAYER_WEIGHTS = {"format": 0.25, "calculation": 0.45, "knowledge": 0.30}
ABLATION_SETS: Tuple[Tuple[str, ...], ...] = (
    ("format",),
    ("calculation",),
    ("knowledge",),
    ("format", "calculation"),
    ("format", "knowledge"),
    ("calculation", "knowledge"),
    ("format", "calculation", "knowledge"),
)


def tlr_score(result: Dict[str, Any], enabled: Optional[Sequence[str]] = None,
              weights: Optional[Dict[str, float]] = None) -> float:
    """只在启用层上归一化，便于 ablation 对比。"""
    enabled = tuple(enabled) if enabled else ("format", "calculation", "knowledge")
    weights = weights or LAYER_WEIGHTS
    denom = sum(weights[k] for k in enabled) or 1.0
    total = 0.0
    for key in enabled:
        block = result.get(key) or {}
        checks = block.get("checks") or {}
        w = weights[key]
        if block.get("skipped") or not checks:
            frac = 1.0
        else:
            vals = [1.0 if v else 0.0 for v in checks.values() if isinstance(v, bool)]
            frac = (sum(vals) / len(vals)) if vals else (1.0 if block.get("passed") else 0.0)
        total += w * frac
    return round(total / denom, 4)


def pick_instruction(category: str, seeds: Sequence[str]) -> str:
    mapping = {
        "wastewater_calculation": ("去除率", "污染负荷", "水力停留", "F/M", "BOD5/COD", "两位小数"),
        "wastewater_process": ("表格", "设计输入", "→", "四个运行"),
        "process_diagnosis": ("三个可能原因", "现象→可能原因"),
        "monitoring_report": ("监测对象", "环境监测报告"),
        "monitoring_analysis": ("信息不足", "不得虚构"),
        "air_pollution": ("VOCs", "优先、备选"),
        "solid_waste": ("分类、暂存"),
        "noise_control": ("声源、传播路径"),
        "eia_cleaner_production": ("需核实现行标准", "清洁生产"),
    }
    keys = mapping.get(category, ())
    for seed in seeds:
        if any(k in seed for k in keys):
            return seed
    return seeds[0] if seeds else "计算结果必须保留两位小数，并在结尾进行数量级合理性检查。"


def chosen_response(query: str, gold: Optional[Dict[str, Any]], category: str = "") -> str:
    gold = gold or {}
    params = gold.get("params") or {}
    fmt = gold.get("format_constraints") or {}
    if fmt.get("must_start_insufficient") or "信息不足" in query or "未给出" in query and "标准" in query:
        return (
            "信息不足。缺少执行标准名称/年份/适用对象，或缺少水量、采样时间、检测方法中的关键项，"
            "不能给出确定结论，需核实现行标准。"
        )
    if gold.get("removal_rate_pct") is not None and params.get("Cin") is not None:
        cin, cout = params["Cin"], params["Cout"]
        rate = gold["removal_rate_pct"]
        qv = params.get("Q")
        load = gold.get("load_kg_d")
        parts = [f"已知进水{cin} mg/L、出水{cout} mg/L"]
        if qv is not None:
            parts[0] += f"、水量{qv} m³/d"
        parts[0] += "。"
        parts.append(f"去除率=({cin}-{cout})/{cin}×100%={rate:.2f}%。")
        if load is not None and qv is not None:
            delta = float(cin) - float(cout)
            parts.append(f"每日去除负荷={qv}×({cin}-{cout})×0.001={load:.2f} kg/d。")
            parts.append(f"数量级合理性检查：日去除约{delta * float(qv) * 1e-6:.2f} t，与水量和浓度差匹配。")
        elif gold.get("require_magnitude_check"):
            parts.append("数量级合理性检查：去除率落在 0～100%，与进出水浓度差匹配。")
        parts.append("未提供排放标准，需核实现行标准。")
        return "".join(parts)
    if gold.get("load_kg_d") is not None and params.get("Q") is not None:
        qv = params["Q"]
        c = params.get("C", params.get("Cin"))
        load = gold["load_kg_d"]
        return (
            f"水量{qv} m³/d，浓度{c} mg/L。"
            f"负荷={qv}×{c}×0.001={load:.2f} kg/d。"
            f"数量级合理性检查通过。未提供标准，需核实现行标准。"
        )
    if gold.get("hrt_h") is not None:
        v, qv, hrt = params.get("V"), params.get("Q"), gold["hrt_h"]
        return (
            f"池容 V={v} m³，水量 Q={qv} m³/d。"
            f"HRT=V/Q×24={v}/{qv}×24={hrt:.2f} h。"
            f"数量级合理性检查：停留时间与池容/流量匹配。"
        )
    if gold.get("ratio") is not None:
        bod, cod, ratio = params.get("BOD"), params.get("COD"), gold["ratio"]
        judge = "可生化性较好，宜优先生物处理" if ratio >= 0.3 else "可生化性偏差，需评估预处理或物化强化"
        return f"BOD5/COD={bod}/{cod}={ratio:.2f}。{judge}。未提供排放标准，需核实现行标准。"
    if gold.get("fm") is not None:
        p, fm = params, gold["fm"]
        level = "高负荷" if fm >= 0.5 else ("中负荷" if fm >= 0.2 else "低负荷")
        return (
            f"F/M=(Q×BOD)/(V×MLSS)=({p.get('Q')}×{p.get('Cin')})/"
            f"({p.get('V')}×{p.get('MLSS')})={fm:.4f} kgBOD/(kgMLSS·d)，属{level}。"
        )
    if fmt.get("exactly_n_causes") or category == "process_diagnosis":
        return (
            "原因1：丝状菌膨胀。证据：SVI升高、出水SS增加。复核方法：镜检丝状菌与SV测定。\n"
            "原因2：二沉池表面负荷偏高。证据：跑泥或界面上升。复核方法：核算表面负荷与回流比。\n"
            "原因3：营养比或溶解氧失调。证据：污泥结构松散或硝化下降。复核方法：复核C/N/P与DO。"
        )
    if category == "wastewater_process" and ("表格" in query or "比较" in query):
        return (
            "| 项目 | 处理效果 | 占地 | 能耗 | 污泥产量 | 适用条件 |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| 方案A | 对COD/TN较稳 | 中 | 中 | 中 | 城镇污水 |\n"
            "| 方案B | 出水更清 | 小 | 高 | 低 | 用地紧张 |\n"
            "设计输入：按题给水量与水质。关键假设：未给标准时不编造限值。"
            "计算过程：先列已知量。推荐方案：结合占地与出水目标。风险点：膜污染或污泥膨胀。"
            "需核实现行标准。"
        )
    if "→" in query or "四个" in query and "单元" in query:
        return (
            "格栅→沉砂池→A2/O生化池→二沉池→消毒。各单元分别去除漂浮物、无机砂、溶解性有机物与营养盐、残留病原。"
            "需核实现行标准。"
        )
    return (
        "设计输入：按题给条件。关键假设：未给标准时不编造限值。"
        "计算过程：先列已知量再代入公式。推荐方案：结合水质与占地选择。"
        "风险点：二次污染与运行安全。需核实现行标准。需补测参数清单：暂无。"
    )


def typed_negatives(query: str, gold: Optional[Dict[str, Any]], chosen: str) -> List[Dict[str, str]]:
    gold = gold or {}
    params = gold.get("params") or {}
    cin, cout, q = params.get("Cin"), params.get("Cout"), params.get("Q")
    negs: List[Dict[str, str]] = [
        {"error_type": "format_error", "text": "该厂运行正常，建议继续观察，无需按格式作答。"},
    ]
    if cin is not None and cout is not None:
        wrong = round(removal_rate(float(cin), float(cout)) + 12.5, 2)
        negs.append({"error_type": "calc_error", "text": f"进水{cin}、出水{cout}，去除率约为{wrong}%。过程略。"})
        if gold.get("load_kg_d") is not None and q is not None:
            load = gold["load_kg_d"]
            negs.append({
                "error_type": "unit_error",
                "text": f"去除率={round(removal_rate(float(cin), float(cout)), 2)}%。每日去除负荷为 {load:.2f} t/d。",
            })
    negs.append({
        "error_type": "fabricated_standard",
        "text": "按照 GB 18918-2002 一级A限值 COD 为 50 mg/L，已经达标，可以直接排放。",
    })
    negs.append({
        "error_type": "false_certainty",
        "text": "根据经验可以判定已经达标，无需补充监测数据或标准文本。",
    })
    if gold.get("format_constraints", {}).get("must_start_insufficient") or "信息不足" in query:
        negs.append({"error_type": "missing_insufficient", "text": "数据虽然不全，但可以认为已经达标。"})
    # 去重
    seen = set()
    out = []
    for row in negs:
        if row["text"] == chosen or row["text"] in seen:
            continue
        seen.add(row["text"])
        out.append(row)
    return out


def build_pair(query: str, instruction: str, chosen: str, rejected: str, error_type: str,
               gold: Optional[dict] = None, category: str = "") -> dict:
    human = f"{query}\n\n【格式要求】{instruction}".strip()
    pair = {
        "conversations": [{"from": "human", "value": human}],
        "chosen": {"from": "gpt", "value": chosen},
        "rejected": {"from": "gpt", "value": rejected},
        "error_type": error_type,
        "source": "envif_tlr_offline",
        "category": category,
    }
    if gold:
        pair["gold"] = gold
    return pair
