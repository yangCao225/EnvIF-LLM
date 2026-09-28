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

import re
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
        "air_pollution": ("VOCs", "优先、备选", "脱硫", "脱硝", "SO2", "NOx", "烟气", "24×10", "基准氧"),
        "air_calculation": ("烟气", "24×10", "基准氧", "SO2", "NOx", "两位小数"),
        "air_process": ("脱硫", "脱硝", "优先、备选", "石灰石", "SCR"),
        "air_diagnosis": ("三个可能原因", "氨逃逸", "脱硫"),
        "solid_waste": ("分类、暂存"),
        "noise_control": ("声源、传播路径"),
        "eia_cleaner_production": ("需核实现行标准", "清洁生产"),
    }
    keys = mapping.get(category, ())
    for seed in seeds:
        if any(k in seed for k in keys):
            return seed
    return seeds[0] if seeds else "计算结果必须保留两位小数，并在结尾进行数量级合理性检查。"


_AIR_KEYS = (
    "烟气", "废气", "SO2", "SO₂", "NOx", "NOₓ", "VOCs", "mg/m³", "mg/m3",
    "脱硫", "脱硝", "风量", "含氧", "SNCR", "SCR", "H2S",
)


def _headers(*pairs: Tuple[str, str]) -> str:
    return "".join(f"{k}：{v}" for k, v in pairs)


def _is_air_query(query: str, gold: Optional[Dict[str, Any]] = None, category: str = "") -> bool:
    gold = gold or {}
    params = gold.get("params") or {}
    if category in ("air_pollution", "air_calculation", "air_process", "air_diagnosis") or params.get("Qg") or gold.get("medium") == "gas":
        return True
    return any(k in (query or "") for k in _AIR_KEYS)


def _air_process_response(query: str) -> str:
    q = query or ""
    if any(k in q for k in ("SO2", "SO₂", "NOx", "NOₓ")) and ("脱硫" in q and "脱硝" in q):
        return (
            "优先：石灰石-石膏湿法脱硫 + SCR 脱硝（连续高硫高氮烟气）。"
            "备选：半干法脱硫 + SNCR（中低负荷或投资受限）。"
            "不推荐：只靠高烟囱稀释，或无温度窗口盲目喷氨。"
            "设计输入：按题给烟气量与浓度。关键假设：不编造排放限值或设备参数。"
            "二次污染与运行安全：石膏副产物、氨逃逸、废催化剂需单独处置。需核实现行标准。"
        )
    if any(k in q for k in ("SO2", "SO₂", "脱硫", "石灰石", "石膏")):
        return (
            "优先：石灰石-石膏湿法脱硫（高硫烟气、连续运行）。"
            "备选：半干法或循环流化床脱硫（中低硫、节水）。"
            "不推荐：仅靠高烟囱稀释。"
            "设计输入：按题给烟气量与 SO2 浓度。关键假设：不编造排放限值。"
            "二次污染与运行安全：石膏副产物、浆液泄漏与防腐。需核实现行标准。"
        )
    if any(k in q for k in ("NOx", "NOₓ", "脱硝", "SNCR", "SCR")):
        return (
            "优先：SCR 脱硝（催化剂床层、效率较高，适合连续稳定烟气）。"
            "备选：SNCR（炉膛喷氨，投资较低但温度窗口窄）。"
            "不推荐：无温度窗口时盲目喷氨。"
            "设计输入：按题给烟气量与 NOx 浓度。关键假设：不编造排放限值。"
            "二次污染与运行安全：氨逃逸与废催化剂处置。需核实现行标准。"
        )
    if any(k in q for k in ("H2S", "恶臭", "除臭")):
        return (
            "优先：化学吸收（高浓度 H2S）。"
            "备选：生物滤池（稳定低浓度恶臭）。"
            "不推荐：无收集直接高空排放。"
            "设计输入：按题给风量与浓度。关键假设：不编造设备参数。"
            "二次污染与运行安全：吸收液与废填料。需核实现行标准。"
        )
    if "RTO" in q or "rto" in q.lower():
        return (
            "优先：RTO（连续高浓度 VOCs）。"
            "备选：催化燃烧或转轮浓缩。"
            "不推荐：仅用小容量炭罐。"
            "设计输入：按题给风量与浓度。关键假设：不编造设备参数。"
            "二次污染与运行安全：爆炸下限、高温与废活性炭。需核实现行标准。"
        )
    return (
        "优先：催化燃烧（连续中高浓度 VOCs）。"
        "备选：活性炭吸附（波动或低浓度）。"
        "不推荐：无预处理直接高浓度吸附。"
        "设计输入：按题给风量与浓度。关键假设：不编造设备参数。"
        "二次污染与运行安全：关注冷凝液、废活性炭和高温爆炸风险。需核实现行标准。"
    )


def _gas_calc_response(params: Dict[str, Any], gold: Dict[str, Any]) -> str:
    qg = params.get("Qg")
    cin = params.get("Cin")
    cout = params.get("Cout")
    c = params.get("C", cin)
    load = gold.get("load_kg_d")
    rate = gold.get("removal_rate_pct")
    family = ((gold.get("knowledge_constraints") or {}).get("process_family") or "")
    family_note = {
        "so2": "该负荷为含硫烟气脱硫削减量。",
        "nox": "该负荷为含氮烟气脱硝削减量。",
        "vocs": "该负荷为吸附或催化燃烧削减量。",
    }.get(family, "该负荷为烟气污染物削减量。")
    parts = [
        "符号约定：烟气量用 Qg（m³/h），浓度用 mg/m³，不可把烟气量当成污水量做 0.001 换算。"
    ]
    if cin is not None and cout is not None:
        parts.append(f"已知进口 C_in={cin} mg/m³、出口 C_out={cout} mg/m³、烟气量 Qg={qg} m³/h。")
        if rate is not None:
            parts.append(f"去除率=(C_in-C_out)/C_in×100%=({cin}-{cout})/{cin}×100%={rate:.2f}%。")
        if load is not None:
            parts.append(
                f"每日去除负荷=Qg×(C_in-C_out)×24×10^{{-6}}={qg}×({cin}-{cout})×24×10^{{-6}}={load:.2f} kg/d。"
            )
    else:
        parts.append(f"烟气量 Qg={qg} m³/h，浓度 C={c} mg/m³。")
        if load is not None:
            parts.append(f"负荷=Qg×C×24×10^{{-6}}={qg}×{c}×24×10^{{-6}}={load:.2f} kg/d。")
    if gold.get("require_magnitude_check") or load is not None:
        tons = (float(load) / 1000.0) if load is not None else 0.0
        parts.append(f"数量级合理性检查：日去除约{tons:.2f} t，与烟气量和浓度差匹配。")
    parts.append(family_note)
    parts.append("二次污染与运行安全：脱硫石膏、氨逃逸或废催化剂需单独处置。未提供排放标准，需核实现行标准。")
    return "".join(parts)


def chosen_response(query: str, gold: Optional[Dict[str, Any]], category: str = "") -> str:
    gold = gold or {}
    params = gold.get("params") or {}
    fmt = gold.get("format_constraints") or {}
    if fmt.get("must_start_insufficient") or ("信息不足" in query and ("未给出" in query or "缺少" in query or "不得直接" in query)):
        if _is_air_query(query, gold, category):
            return (
                "信息不足。缺少执行标准名称/年份/适用对象，或缺少烟气量、含氧量、采样时间和检测方法中的关键项，"
                "不能给出确定结论，需核实现行标准。缺失参数：标准文本、烟气量或含氧量、检测方法。"
            )
        return (
            "信息不足。缺少执行标准名称/年份/适用对象，或缺少水量、采样时间、检测方法中的关键项，"
            "不能给出确定结论，需核实现行标准。缺失参数：标准文本、采样时间、检测方法。"
        )
    if re.search(r"pH", query) and ("无量纲" in query or "不得添加" in query) and "去除率" not in query:
        m = re.search(r"pH\s*[=为是]\s*([0-9.]+)", query)
        val = m.group(1) if m else "题给数值"
        return (
            f"pH={val}（无量纲，不得写成 mg/L）。"
            "若电极未校准或温度补偿异常，应复测。"
            "未提供排放标准，需核实现行标准，不能判定达标。"
        )
    if gold.get("o2_corrected") is not None or (
        params.get("O2m") is not None and params.get("O2s") is not None and params.get("C") is not None
    ):
        c = params.get("C")
        om, os = params.get("O2m"), params.get("O2s")
        corr = gold.get("o2_corrected")
        if corr is None and c is not None and om is not None and os is not None:
            corr = round(float(c) * (21.0 - float(os)) / (21.0 - float(om)), 2)
        return (
            f"定义：基准氧折算把不同含氧量下的实测浓度归一到同一基准。"
            f"公式：C' = C×(21−O2,s)/(21−O2,m)。"
            f"已知实测 C={c} mg/m³，烟气含氧 O2,m={om}%，基准氧 O2,s={os}%。"
            f"代入：C'={c}×(21−{os})/(21−{om})={corr:.2f} mg/m³。"
            f"同时保留实测浓度 {c} mg/m³，不得把未折算值当作折算值。"
            f"数量级合理性检查：折算后与实测同数量级。未提供排放标准，需核实现行标准。"
        )
    if params.get("Qg") is not None and (
        gold.get("removal_rate_pct") is not None or gold.get("load_kg_d") is not None
    ):
        return _gas_calc_response(params, gold)
    if gold.get("removal_rate_pct") is not None and params.get("Cin") is not None and not params.get("Qg"):
        cin, cout = params["Cin"], params["Cout"]
        rate = gold["removal_rate_pct"]
        qv = params.get("Q")
        load = gold.get("load_kg_d")
        parts = [f"符号约定：水量用 Q（m³/d），池容才用 V（m³）。已知进水 C_in={cin} mg/L、出水 C_out={cout} mg/L"]
        if qv is not None:
            parts[0] += f"、水量 Q={qv} m³/d"
        parts[0] += "。"
        parts.append(f"去除率=(C_in-C_out)/C_in×100%=({cin}-{cout})/{cin}×100%={rate:.2f}%。")
        if load is not None and qv is not None:
            delta = float(cin) - float(cout)
            parts.append(
                f"每日去除负荷=Q×(C_in-C_out)×0.001={qv}×({cin}-{cout})×0.001={load:.2f} kg/d。"
            )
            parts.append(f"数量级合理性检查：日去除约{delta * float(qv) * 1e-6:.2f} t，与水量和浓度差匹配。")
        elif gold.get("require_magnitude_check"):
            parts.append("数量级合理性检查：去除率落在 0～100%，与进出水浓度差匹配。")
        parts.append("未提供排放标准，需核实现行标准。")
        return "".join(parts)
    if gold.get("load_kg_d") is not None and params.get("Q") is not None and not params.get("Qg"):
        qv = params["Q"]
        c = params.get("C", params.get("Cin"))
        load = gold["load_kg_d"]
        return (
            f"符号约定：水量用 Q（m³/d），不要把流量写成 V。"
            f"水量 Q={qv} m³/d，浓度 C={c} mg/L。"
            f"负荷=Q×C×0.001={qv}×{c}×0.001={load:.2f} kg/d。"
            f"数量级合理性检查通过。未提供标准，需核实现行标准。"
        )
    if gold.get("hrt_h") is not None:
        v, qv, hrt = params.get("V"), params.get("Q"), gold["hrt_h"]
        return (
            f"符号约定：V 为池容（m³），Q 为水量（m³/d）。"
            f"池容 V={v} m³，水量 Q={qv} m³/d。"
            f"HRT=V/Q×24={v}/{qv}×24={hrt:.2f} h。"
            f"不可写成 HRT=V/Q 而漏乘 24。数量级合理性检查：停留时间与池容/流量匹配。"
        )
    if gold.get("ratio") is not None:
        bod, cod, ratio = params.get("BOD"), params.get("COD"), gold["ratio"]
        judge = "可生化性较好，宜优先生物处理" if ratio >= 0.3 else "可生化性偏差，需评估预处理或物化强化"
        return (
            f"定义：BOD5/COD 表征废水可生化性。原理：比值越高，好氧微生物越容易利用有机物。"
            f"适用：筛选生物处理可行性。BOD5/COD={bod}/{cod}={ratio:.2f}。{judge}。"
            f"未提供排放标准，需核实现行标准。"
        )
    if gold.get("fm") is not None:
        p, fm = params, gold["fm"]
        level = "高负荷" if fm >= 0.5 else ("中负荷" if fm >= 0.2 else "低负荷")
        return (
            f"定义：F/M 为污泥负荷。原理：进水有机物与曝气池活性污泥量之比。"
            f"适用：判断高/中/低负荷运行。"
            f"F/M=(Q×BOD)/(V×MLSS)=({p.get('Q')}×{p.get('Cin')})/"
            f"({p.get('V')}×{p.get('MLSS')})={fm:.4f} kgBOD/(kgMLSS·d)，属{level}。"
        )
    if "报告摘要" in query and any(k in query for k in ("烟囱", "烟气", "SO2", "NOx")):
        plant = re.search(r"对象为([^，,]+)", query)
        date = re.search(r"采样时间为([^，,]+)", query)
        so2 = re.search(r"SO2\s*=\s*([0-9.]+)", query)
        nox = re.search(r"NOx\s*=\s*([0-9.]+)", query)
        o2 = re.search(r"含氧量\s*=\s*([0-9.]+)", query)
        return _headers(
            ("监测对象", (plant.group(1).strip() if plant else "题给烟囱") + "。"),
            ("采样时间", (date.group(1).strip() if date else "题给时间") + "。"),
            (
                "监测结果",
                f"SO2 {so2.group(1) if so2 else '—'} mg/m³，"
                f"NOx {nox.group(1) if nox else '—'} mg/m³，"
                f"含氧量 {o2.group(1) if o2 else '未测'}%。",
            ),
            ("异常项", "相对常规运行需复核峰值、含氧量或氨逃逸。"),
            ("结论", "未提供排放标准，需核实现行标准，不能判定已经达标。"),
        )
    if "报告摘要" in query or ("监测对象" in query and "采样时间" in query):
        plant = re.search(r"对象为([^，,]+)", query)
        date = re.search(r"采样时间为([^，,]+)", query)
        cod = re.search(r"COD\s*=\s*([0-9.]+)", query)
        nh = re.search(r"NH3-N\s*=\s*([0-9.]+)", query)
        tp = re.search(r"TP\s*=\s*([0-9.]+)", query)
        return _headers(
            ("监测对象", (plant.group(1).strip() if plant else "题给出水") + "。"),
            ("采样时间", (date.group(1).strip() if date else "题给时间") + "。"),
            (
                "监测结果",
                f"COD {cod.group(1) if cod else '—'} mg/L，"
                f"NH3-N {nh.group(1) if nh else '—'} mg/L，"
                f"TP {tp.group(1) if tp else '—'} mg/L。",
            ),
            ("异常项", "相对常规运行需复核峰值或突变项。"),
            ("结论", "未提供排放标准，需核实现行标准，不能判定已经达标。"),
        )
    if category == "air_diagnosis" or (
        fmt.get("exactly_n_causes") and _is_air_query(query, gold, category)
    ):
        if any(k in query for k in ("SO2", "SO₂", "脱硫", "石膏")):
            return (
                "原因1：浆液pH或钙硫比失调。证据：出口SO2回升、脱硫效率下降。复核方法：核对浆液pH与石灰石给料。\n"
                "原因2：除雾器或石膏旋流器结垢堵塞。证据：吸收塔压差升高。复核方法：核对压差曲线与石膏含水率。\n"
                "原因3：烟气量或进口SO2冲击。证据：负荷波动。复核方法：核对Qg与进口浓度。\n"
                "步骤1：核对CEMS。步骤2：查浆液与给料。步骤3：查除雾器压差。"
                "步骤4：排除漏风。步骤5：列出需补测参数。"
                "二次污染与运行安全：石膏外溢与浆液泄漏。需核实现行标准。"
            )
        return (
            "原因1：氨逃逸偏高。证据：出口氨或嗅辨异常。复核方法：核对喷氨量与温度窗口。\n"
            "原因2：催化剂失活或堵塞。证据：脱硝效率下降、床层压差升高。复核方法：核对运行小时与压差曲线。\n"
            "原因3：烟气量或进口NOx冲击。证据：负荷波动。复核方法：核对Qg与进口浓度。\n"
            "步骤1：核对CEMS。步骤2：查喷氨与温度。步骤3：查催化剂压差。"
            "步骤4：排除漏风。步骤5：列出需补测参数。"
            "二次污染与运行安全：氨逃逸与废催化剂。需核实现行标准。"
        )
    if fmt.get("exactly_n_causes") or category == "process_diagnosis":
        return (
            "原因1：丝状菌膨胀。证据：SVI升高、出水SS增加。复核方法：镜检丝状菌与SV测定。\n"
            "原因2：二沉池表面负荷偏高。证据：跑泥或界面上升。复核方法：核算表面负荷与回流比。\n"
            "原因3：营养比或溶解氧失调。证据：污泥结构松散或硝化下降。复核方法：复核C/N/P与DO。\n"
            "步骤1：核对在线与化验数据。步骤2：现场看泥和水色。步骤3：复核曝气与回流。"
            "步骤4：检查进水冲击。步骤5：给出临时调控与需补测参数。"
            "二次污染与运行安全：避免过量加药和污泥上浮外溢。"
        )
    if category in ("air_pollution", "air_process") or (
        _is_air_query(query, gold, category)
        and any(k in query for k in ("比较", "优先", "排序", "推荐方案", "RTO", "吸附", "脱硫", "脱硝", "SNCR", "SCR"))
    ):
        return _air_process_response(query)
    if category == "wastewater_process" or "比较" in query or "表格" in query:
        return (
            "| 项目 | 处理效果 | 占地 | 能耗 | 污泥产量 | 适用条件 |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| 方案A | 对COD/TN较稳 | 中 | 中 | 中 | 城镇污水 |\n"
            "| 方案B | 出水更清 | 小 | 高 | 低 | 用地紧张 |\n"
            "设计输入：按题给水量与水质。关键假设：未给标准时不编造限值。"
            "计算过程：先列已知量。推荐方案：结合占地与出水目标。风险点：膜污染或污泥膨胀。"
            "二次污染与运行安全：关注剩余污泥、膜清洗药剂和恶臭逸散。需核实现行标准。"
        )
    if "→" in query or ("四个" in query and "单元" in query) or "工艺流程" in query:
        return (
            "格栅→沉砂池→A2/O生化池→二沉池→消毒。各单元分别去除漂浮物、无机砂、溶解性有机物与营养盐、残留病原。"
            "设计输入：按题给规模。关键假设：未给标准不编造限值。计算过程：先串流程再核负荷。"
            "推荐方案：按水质选生化主体。风险点：污泥膨胀与消毒副产物。需核实现行标准。"
        )
    if category == "solid_waste" or ("分类" in query and "暂存" in query):
        return (
            "分类：按危险废物/一般固废鉴别后分开存放，禁止混装。"
            "暂存：防雨防渗、标识齐全、贮存不超过规定时限。"
            "运输：交给有资质单位，填写转移联单。"
            "处理处置：优先资源化，不能利用的安全填埋或焚烧。"
            "风险控制：二次污染与运行安全，防止渗滤液、扬尘和非法转移。需核实现行标准。"
        )
    if category == "noise_control" or ("声源" in query and "传播路径" in query):
        return (
            "声源：对风机、泵做减振隔声罩，优先低噪声设备。"
            "传播路径：屏障、包覆风管、绿化降噪。"
            "受声点：敏感点增监测，夜间限制高噪声作业。"
            "二次污染与运行安全：隔声材料防火、检修空间保留。未给标准限值，需核实现行标准。"
        )
    if "什么是" in query or "解释" in query or "含义" in query or "定义" in query:
        return (
            "定义：该术语指环境工程中对应单元/指标的规范含义。"
            "原理：通过物理分离、生物代谢或化学转化实现目标污染物削减。"
            "适用：需结合水质、水量、占地和二次污染风险选择，不能脱离工况套用。"
            "未提供排放标准，需核实现行标准。"
        )
    if "优先" in query and "备选" in query:
        return (
            "优先：结合浓度、风量和防爆要求选择主体工艺。"
            "备选：在负荷波动或维护窗口作为并联/切换方案。"
            "不推荐：与水质明显不匹配或二次污染不可控的方案。"
            "设计输入：按题给风量与浓度。关键假设：不编造设备参数。"
            "二次污染与运行安全：关注冷凝液、废活性炭和高温爆炸风险。需核实现行标准。"
        )
    return (
        "设计输入：按题给条件。关键假设：未给标准时不编造限值。"
        "计算过程：先列已知量再代入公式。推荐方案：结合水质与占地选择。"
        "风险点：二次污染与运行安全。需核实现行标准。需补测参数清单：水量、标准文本、检测方法。"
    )


def typed_negatives(
    query: str,
    gold: Optional[Dict[str, Any]],
    chosen: str,
    category: str = "",
) -> List[Dict[str, str]]:
    gold = gold or {}
    params = gold.get("params") or {}
    cin, cout, q = params.get("Cin"), params.get("Cout"), params.get("Q")
    qg = params.get("Qg")
    v = params.get("V")
    c = params.get("C", cin)
    negs: List[Dict[str, str]] = []
    if qg is not None and cin is not None and cout is not None:
        wrong = round(removal_rate(float(cin), float(cout)) + 12.5, 2)
        rate = round(removal_rate(float(cin), float(cout)), 2)
        delta = float(cin) - float(cout)
        negs.append({
            "error_type": "calc_error",
            "text": (
                f"符号约定：烟气量用 Qg。已知进口{cin} mg/m³、出口{cout} mg/m³、烟气量 Qg={qg} m³/h。"
                f"去除率=({cin}-{cout})/{cin}×100%={wrong:.2f}%。"
                "数量级合理性检查通过。未提供排放标准，需核实现行标准。"
            ),
        })
        if gold.get("load_kg_d") is not None:
            negs.append({
                "error_type": "calc_error",
                "text": (
                    f"已知进口 C_in={cin} mg/m³、出口 C_out={cout} mg/m³、烟气量 Qg={qg} m³/h。"
                    f"去除率=({cin}-{cout})/{cin}×100%={rate:.2f}%。"
                    f"每日去除负荷=Qg×(C_in-C_out)={qg}×({cin}-{cout})={delta * float(qg):.2f} kg/d。"
                    "数量级合理性检查通过。未提供排放标准，需核实现行标准。"
                ),
            })
            negs.append({
                "error_type": "calc_error",
                "text": (
                    f"把烟气量当成水量。负荷=Q×(C_in-C_out)×0.001={qg}×({cin}-{cout})×0.001="
                    f"{delta * float(qg) * 0.001:.2f} kg/d。数量级合理性检查通过。需核实现行标准。"
                ),
            })
            negs.append({
                "error_type": "unit_error",
                "text": (
                    f"符号约定：烟气量用 Qg。去除率={rate:.2f}%。"
                    f"每日去除负荷=Qg×(C_in-C_out)×24×10^{{-6}}={gold['load_kg_d']:.2f} t/d。"
                    "数量级合理性检查通过。未提供排放标准，需核实现行标准。"
                ),
            })
    elif gold.get("o2_corrected") is not None or (params.get("O2m") is not None and params.get("C") is not None):
        cval = float(params.get("C") or 0)
        om = float(params.get("O2m") or 8)
        os = float(params.get("O2s") or 6)
        corr = gold.get("o2_corrected")
        if corr is None:
            corr = round(cval * (21.0 - os) / (21.0 - om), 2)
        negs.append({
            "error_type": "calc_error",
            "text": (
                f"实测浓度 {cval} mg/m³，含氧 {om}%。直接把实测值当作折算浓度 {cval:.2f} mg/m³。"
                "数量级合理性检查通过。需核实现行标准。"
            ),
        })
        inv = round(cval * (21.0 - om) / (21.0 - os), 2)
        negs.append({
            "error_type": "calc_error",
            "text": (
                f"公式写反。C'={cval}×(21−{om})/(21−{os})={inv:.2f} mg/m³。"
                "数量级合理性检查通过。需核实现行标准。"
            ),
        })
    elif qg is not None and gold.get("load_kg_d") is not None:
        cval = float(c or 0)
        negs.append({
            "error_type": "calc_error",
            "text": (
                f"符号约定：烟气量用 Qg。烟气量 Qg={qg} m³/h，浓度 C={cval} mg/m³。"
                f"负荷=Qg×C={qg}×{cval}={cval * float(qg):.2f} kg/d。"
                "数量级合理性检查通过。需核实现行标准。"
            ),
        })
        negs.append({
            "error_type": "calc_error",
            "text": (
                f"套用废水公式。负荷=Q×C×0.001={qg}×{cval}×0.001={cval * float(qg) * 0.001:.2f} kg/d。"
                "数量级合理性检查通过。需核实现行标准。"
            ),
        })
    elif cin is not None and cout is not None:
        wrong = round(removal_rate(float(cin), float(cout)) + 12.5, 2)
        negs.append({
            "error_type": "calc_error",
            "text": (
                f"符号约定：水量用 Q。已知进水{cin} mg/L、出水{cout} mg/L"
                + (f"、水量 Q={q} m³/d" if q is not None else "")
                + f"。去除率=({cin}-{cout})/{cin}×100%={wrong:.2f}%。"
                "数量级合理性检查通过。未提供排放标准，需核实现行标准。"
            ),
        })
        if gold.get("load_kg_d") is not None and q is not None:
            delta = float(cin) - float(cout)
            # 漏乘 0.001：公式像对、单位写成 kg/d，数值错三个数量级
            negs.append({
                "error_type": "calc_error",
                "text": (
                    f"已知进水 C_in={cin} mg/L、出水 C_out={cout} mg/L、水量 Q={q} m³/d。"
                    f"去除率=({cin}-{cout})/{cin}×100%={round(removal_rate(float(cin), float(cout)), 2):.2f}%。"
                    f"每日去除负荷=Q×(C_in-C_out)={q}×({cin}-{cout})={delta * float(q):.2f} kg/d。"
                    f"数量级合理性检查：日去除约{delta * float(q) * 1e-3:.2f} t。未提供排放标准，需核实现行标准。"
                ),
            })
            # 把流量写成 V，并胡乱套 BOD 公式（评测里真实出现过）
            negs.append({
                "error_type": "calc_error",
                "text": (
                    f"已知V={q} m³/d，BOD5/COD=0.36。负荷=V×(BOD5+C)/(F+1)×0.6="
                    f"{q}×(0.36+{c or 0})/(0.34+1)×0.6≈{float(q) * 0.4:.2f} kg/d。"
                    "数量级合理性检查通过。需核实现行标准。"
                ),
            })
            negs.append({
                "error_type": "unit_error",
                "text": (
                    f"符号约定：水量用 Q。去除率={round(removal_rate(float(cin), float(cout)), 2):.2f}%。"
                    f"每日去除负荷=Q×(C_in-C_out)×0.001={gold['load_kg_d']:.2f} t/d。"
                    "数量级合理性检查通过。未提供排放标准，需核实现行标准。"
                ),
            })
    if gold.get("load_kg_d") is not None and q is not None and cin is None:
        cval = float(c or 0)
        negs.append({
            "error_type": "calc_error",
            "text": (
                f"符号约定：水量用 Q。水量 Q={q} m³/d，浓度 C={cval} mg/L。"
                f"负荷=Q×C={q}×{cval}={cval * float(q):.2f} kg/d。"
                "数量级合理性检查通过。需核实现行标准。"
            ),
        })
    if gold.get("hrt_h") is not None and v is not None and q is not None:
        negs.append({
            "error_type": "calc_error",
            "text": (
                f"符号约定：V 为池容，Q 为水量。池容 V={v} m³，水量 Q={q} m³/d。"
                f"HRT=V/Q={v}/{q}={float(v)/float(q):.2f} h。"
                "数量级合理性检查：停留时间与池容/流量匹配。需核实现行标准。"
            ),
        })
    if category in ("wastewater_process", "air_pollution", "air_process") or ("表格" in query and "比较" in query):
        negs.append({
            "error_type": "content_gap",
            "text": "方案A效果更好，方案B占地更小，建议直接选用方案A。二次污染不明显。需核实现行标准。",
        })
    if "监测对象" in query and "采样时间" in query:
        negs.append({
            "error_type": "content_gap",
            "text": "出水 COD 略高，建议加强生化段运行。已经达标。",
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
    negs.append({"error_type": "format_error", "text": "该厂运行正常，建议继续观察，无需按格式作答。"})
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
