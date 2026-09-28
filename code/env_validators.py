"""
环境工程三层验证器。

Layer 1 格式验证: 标题、表格、字段、单位、字数、流程符号
Layer 2 计算验证: 去除率、污染负荷、停留时间、数值误差、小数位
Layer 3 专业依据验证: 是否虚构限值、信息不足是否声明、工艺与污染物是否匹配

本模块可独立用于评测，也可在 AutoIF 步骤 7 / DPO 构建中复用。
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

ALLOWED_CONC_UNITS = ("mg/L", "μg/L", "ug/L", "mg/m³", "mg/m3")
STANDARD_HINT_RE = re.compile(
    r"(GB[\s\-]?\d+|HJ[\s\-]?\d+|CJJ[\s\-]?\d+|CJ[\s\-]?\d+|排放标准|限值\s*[为是]\s*\d)",
    re.IGNORECASE,
)
NEED_VERIFY_MARKERS = ("需核实现行标准", "未提供标准依据", "标准未给出", "不得自行编造")
INSUFFICIENT_MARKERS = ("信息不足", "缺少参数", "无法给出确定结论", "数据不足")
FALSE_CERTAINTY_RE = re.compile(r"(已经达标|已达标|确定达标|确定超标|肯定超标|可以直接排放|判定达标)")
# 长单位优先，避免把 kg/d 拆成无单位数字。
QUANTITY_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*"
    r"(kg\s*/\s*d|千克/天|t\s*/\s*d|吨/天|mg\s*/\s*L|μg\s*/\s*L|ug\s*/\s*L|"
    r"mg\s*/\s*m[³3]|m[³3]\s*/\s*d|m[³3]\s*/\s*h|%|％|小时|\bh\b)?",
    re.IGNORECASE,
)
PROCESS_KEYWORDS = {
    "cod": ("活性污泥", "a2/o", "a2o", "aao", "氧化沟", "sbr", "mbr", "接触氧化", "uasb", "厌氧", "好氧"),
    "nh3": ("硝化", "反硝化", "a2/o", "a2o", "ao", "sbr", "mbr", "曝气"),
    "tp": ("除磷", "化学除磷", "生物除磷", "a2/o", "a2o", "混凝"),
    "vocs": ("吸附", "催化燃烧", "rto", "rco", "冷凝", "生物滤池"),
    "so2": ("脱硫", "石灰石", "石膏", "湿法", "半干法", "双碱"),
    "nox": ("脱硝", "scr", "sncr", "催化剂", "氨逃逸"),
    "noise": ("隔声", "消声", "减振", "声源", "传播路径", "受声点"),
}


def _norm_unit(unit: Optional[str]) -> Optional[str]:
    if not unit:
        return None
    u = unit.lower().replace(" ", "").replace("³", "3").replace("μ", "u").replace("％", "%")
    mapping = {
        "kg/d": "kg/d", "千克/天": "kg/d",
        "t/d": "t/d", "吨/天": "t/d",
        "mg/l": "mg/L",
        "ug/l": "ug/L",
        "mg/m3": "mg/m3",
        "m3/d": "m3/d",
        "m3/h": "m3/h",
        "%": "%",
        "小时": "h", "h": "h",
    }
    return mapping.get(u, u)


def extract_quantities(text: str) -> List[Dict[str, Any]]:
    """抽出 (数值, 单位)。单位匹配是 EnvIF 相对原 AutoIF 正则打分的增量。"""
    if not text:
        return []
    out = []
    for m in QUANTITY_RE.finditer(text.replace(",", "")):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        out.append({"value": val, "unit": _norm_unit(m.group(2)), "raw": m.group(0)})
    return out


def extract_floats(text: str) -> List[float]:
    return [q["value"] for q in extract_quantities(text)]


def values_for_units(quants: List[Dict[str, Any]], *units: str) -> List[float]:
    allow = set(units)
    return [q["value"] for q in quants if q.get("unit") in allow]


def nearly_equal(a: float, b: float, rel_tol: float = 0.02, abs_tol: float = 0.05) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def removal_rate(cin: float, cout: float) -> float:
    if cin == 0:
        return 0.0
    return (cin - cout) / cin * 100.0


def pollution_load_kg_d(q_m3_d: float, c_mg_l: float) -> float:
    """水量 m³/d × 浓度 mg/L × 0.001 = kg/d。"""
    return q_m3_d * c_mg_l * 0.001


def gas_load_kg_d(q_m3_h: float, c_mg_m3: float) -> float:
    """烟气量 m³/h × 浓度 mg/m³ × 24 × 10^{-6} = kg/d。不可套用废水 Q×C×0.001。"""
    return q_m3_h * c_mg_m3 * 24.0 * 1e-6


def dry_o2_correct(c_mg_m3: float, o2_measured_pct: float, o2_std_pct: float) -> float:
    """基准氧折算：C' = C × (21−O2,s)/(21−O2,m)。未给含氧量不得折算。"""
    denom = 21.0 - float(o2_measured_pct)
    if denom == 0:
        return 0.0
    return float(c_mg_m3) * (21.0 - float(o2_std_pct)) / denom


def _formula_norm(text: str) -> str:
    t = text or ""
    for a, b in (("−", "-"), ("–", "-"), ("——", "-"), ("（", "("), ("）", ")"), ("，", ",")):
        t = t.replace(a, b)
    return t


def _float_token_re(raw: str) -> str:
    v = float(raw)
    if abs(v - round(v)) < 1e-9:
        return rf"{int(round(v))}(?:\.0+)?"
    return re.escape(str(v))


def looks_like_o2_query(query: str) -> bool:
    q = query or ""
    return "基准氧" in q or ("含氧" in q and "折算" in q)


def looks_like_gas_load_query(query: str) -> bool:
    q = query or ""
    has_flow = "烟气量" in q or "m³/h" in q or "m3/h" in q
    has_job = any(k in q for k in ("负荷", "去除率", "去除负荷"))
    return has_flow and has_job


def has_o2_correction_formula(text: str, query: str = "") -> bool:
    """必须出现 C' = C×(21−O2,s)/(21−O2,m) 或其正确代入，不能只写任意乘号。"""
    t = _formula_norm(text)
    q = _formula_norm(query)
    if re.search(
        r"21\s*-\s*O2\s*,?\s*s\s*\)?\s*/\s*\(?\s*21\s*-\s*O2\s*,?\s*m",
        t,
        re.IGNORECASE,
    ):
        return True
    m_s = re.search(r"基准氧\s*([0-9]+(?:\.[0-9]+)?)", q)
    m_m = re.search(r"含氧\s*([0-9]+(?:\.[0-9]+)?)", q)
    if m_s and m_m:
        pat = (
            rf"21\s*-\s*{_float_token_re(m_s.group(1))}\s*\)?\s*/\s*"
            rf"\(?\s*21\s*-\s*{_float_token_re(m_m.group(1))}"
        )
        if re.search(pat, t):
            return True
    return False


def has_gas_load_formula(text: str) -> bool:
    """烟气负荷必须写出 24×10^{-6}，不得把废水 ×0.001 当烟气换算。"""
    t = _formula_norm(text).replace(" ", "")
    has_24 = "24" in t
    has_e6 = bool(re.search(r"10\^\{?-6\}?|10\^-6|1[eE]-6|10\*\*-6", t))
    used_ww = bool(re.search(r"[×x*]\s*0\.001", _formula_norm(text))) and not has_e6
    return has_24 and has_e6 and not used_ww


def has_generic_formula_mark(text: str) -> bool:
    return bool(re.search(r"(=|×|x|\*|÷|/|去除率|负荷)", text or ""))


def hrt_hours(volume_m3: float, q_m3_d: float) -> float:
    if q_m3_d == 0:
        return 0.0
    return volume_m3 / q_m3_d * 24.0


def has_table(text: str) -> bool:
    if not text:
        return False
    if "|" in text and text.count("|") >= 6:
        return True
    if re.search(r"<table[\s>]", text, re.IGNORECASE):
        return True
    return False


def count_cjk_chars(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text or ""))


def layer_format(
    response: str,
    constraints: Optional[Dict[str, Any]] = None,
    query: str = "",
    gold: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    constraints = constraints or {}
    checks = {}
    text = response or ""

    required_headers = constraints.get("required_headers") or []
    if required_headers:
        checks["headers"] = all(h in text for h in required_headers)

    if constraints.get("require_table"):
        checks["table"] = has_table(text)

    if constraints.get("require_arrow_process"):
        units = [u.strip() for u in text.replace("->", "→").split("→") if u.strip()]
        checks["process_flow"] = "→" in text.replace("->", "→") and len(units) >= constraints.get("min_units", 4)

    if constraints.get("allowed_units"):
        used = [u for u in ALLOWED_CONC_UNITS if u in text or u.replace("μ", "u") in text]
        checks["allowed_units"] = len(used) > 0 and not bool(
            re.search(r"\d+\s*(ppm(?!\w)|ppt|g/m3(?!\w))", text, re.IGNORECASE)
        )

    if "min_chars" in constraints or "max_chars" in constraints:
        n = count_cjk_chars(text)
        checks["length"] = constraints.get("min_chars", 0) <= n <= constraints.get("max_chars", 10**9)

    if constraints.get("exactly_n_causes"):
        n = constraints["exactly_n_causes"]
        cause_hits = len(re.findall(r"(原因\s*[123一二三]|可能原因\s*[123一二三]|[1-3][\.、])", text))
        evidence = text.count("证据") >= n
        review = ("复核" in text) or ("验证方法" in text)
        checks["three_causes"] = cause_hits >= n and evidence and review

    if constraints.get("must_start_insufficient"):
        checks["insufficient_prefix"] = text.lstrip().startswith("信息不足")

    if constraints.get("require_two_decimals"):
        checks["two_decimals"] = bool(re.search(r"\d+\.\d{2}(?!\d)", text))

    if constraints.get("require_formula"):
        gold = gold or {}
        params = gold.get("params") or {}
        q = query or ""
        if gold.get("o2_corrected") is not None or looks_like_o2_query(q):
            checks["formula"] = has_o2_correction_formula(text, q)
        elif params.get("Qg") is not None or looks_like_gas_load_query(q):
            checks["formula"] = has_gas_load_formula(text)
        else:
            checks["formula"] = has_generic_formula_mark(text)

    passed = all(checks.values()) if checks else True
    return {"layer": "format", "passed": passed, "checks": checks}


def _find_close_value(values: Iterable[float], target: float, rel_tol: float, abs_tol: float) -> bool:
    return any(nearly_equal(v, target, rel_tol=rel_tol, abs_tol=abs_tol) for v in values)


def layer_calculation(response: str, gold: Optional[Dict[str, Any]] = None,
                      rel_tol: float = 0.02, abs_tol: float = 0.05) -> Dict[str, Any]:
    gold = gold or {}
    text = response or ""
    values = extract_floats(text)
    checks = {}

    params = gold.get("params") or {}
    cin = params.get("Cin", gold.get("cin"))
    cout = params.get("Cout", gold.get("cout"))
    q = params.get("Q", gold.get("Q"))
    qg = params.get("Qg", gold.get("Qg"))
    volume = params.get("V", gold.get("V"))

    expected_rate = gold.get("removal_rate_pct")
    if expected_rate is None and cin is not None and cout is not None:
        expected_rate = round(removal_rate(float(cin), float(cout)), 2)

    expected_load = gold.get("load_kg_d")
    if expected_load is None and qg is not None and cin is not None and cout is not None:
        expected_load = round(gas_load_kg_d(float(qg), float(cin) - float(cout)), 2)
    elif expected_load is None and qg is not None:
        c_in = params.get("C", gold.get("C"))
        if c_in is not None:
            expected_load = round(gas_load_kg_d(float(qg), float(c_in)), 2)
    elif expected_load is None and q is not None and cin is not None and cout is not None:
        expected_load = round(pollution_load_kg_d(float(q), float(cin) - float(cout)), 2)
    elif expected_load is None and q is not None:
        c_in = params.get("C", gold.get("C"))
        if c_in is not None:
            expected_load = round(pollution_load_kg_d(float(q), float(c_in)), 2)

    expected_hrt = gold.get("hrt_h")
    if expected_hrt is None and volume is not None and q is not None:
        expected_hrt = round(hrt_hours(float(volume), float(q)), 2)

    quants = extract_quantities(text)

    if expected_rate is not None:
        pct_vals = values_for_units(quants, "%")
        rate_pool = pct_vals or values
        checks["removal_rate"] = _find_close_value(rate_pool, float(expected_rate), rel_tol, max(abs_tol, 0.1))
        if cin is not None:
            checks["mentions_cin"] = str(int(float(cin))) in re.sub(r"\.0+$", "", str(cin)) or any(
                nearly_equal(v, float(cin), 0.01, 0.2) for v in values
            )
        if cout is not None:
            checks["mentions_cout"] = any(nearly_equal(v, float(cout), 0.01, 0.2) for v in values)

    if expected_load is not None:
        kgd = values_for_units(quants, "kg/d")
        td = [v * 1000.0 for v in values_for_units(quants, "t/d")]
        unit_pool = kgd + td
        if unit_pool:
            checks["load_kg_d"] = _find_close_value(unit_pool, float(expected_load), rel_tol, max(abs_tol, 1.0))
            checks["load_unit"] = bool(kgd) or (
                bool(td) and _find_close_value(td, float(expected_load), rel_tol, max(abs_tol, 1.0))
            )
        else:
            # 只有裸数字、或单位写成 t/d 但数值未换算：计为未通过，供 typed DPO 打 unit_error
            checks["load_kg_d"] = False
            checks["load_unit"] = False

    if expected_hrt is not None:
        h_vals = values_for_units(quants, "h") or values
        checks["hrt_h"] = _find_close_value(h_vals, float(expected_hrt), rel_tol, max(abs_tol, 0.1))

    expected_fm = gold.get("fm")
    if expected_fm is not None:
        checks["fm"] = _find_close_value(values, float(expected_fm), rel_tol, max(abs_tol, 1e-3))

    expected_ratio = gold.get("ratio")
    if expected_ratio is not None:
        checks["ratio"] = _find_close_value(values, float(expected_ratio), rel_tol, max(abs_tol, 0.02))

    expected_o2 = gold.get("o2_corrected")
    if expected_o2 is None:
        c_meas = params.get("C", gold.get("C"))
        o2m = params.get("O2m", gold.get("O2m"))
        o2s = params.get("O2s", gold.get("O2s"))
        if c_meas is not None and o2m is not None and o2s is not None:
            expected_o2 = round(dry_o2_correct(float(c_meas), float(o2m), float(o2s)), 2)
    if expected_o2 is not None:
        checks["o2_corrected"] = _find_close_value(values, float(expected_o2), rel_tol, max(abs_tol, 0.1))

    if gold.get("require_magnitude_check"):
        checks["magnitude_check"] = any(k in text for k in ("数量级", "合理性", "量级检查", "是否合理"))

    if gold.get("require_two_decimals") or expected_rate is not None or expected_o2 is not None:
        checks["two_decimals"] = bool(re.search(r"\d+\.\d{2}(?!\d)", text))

    if not checks:
        return {"layer": "calculation", "passed": True, "checks": {"skipped": True}, "skipped": True}

    passed = all(v is True for k, v in checks.items() if k not in ("mentions_cin", "mentions_cout"))
    return {"layer": "calculation", "passed": passed, "checks": checks}


def layer_knowledge(response: str, query: str = "", constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    constraints = constraints or {}
    text = response or ""
    q = query or ""
    checks = {}

    if constraints.get("forbid_fabricated_standard") or True:
        mentions_standard = bool(STANDARD_HINT_RE.search(text))
        query_has_standard = bool(STANDARD_HINT_RE.search(q)) or any(
            k in q for k in ("标准限值", "排放限值", "给出限值")
        )
        if mentions_standard and not query_has_standard:
            checks["no_fabricated_standard"] = any(m in text for m in NEED_VERIFY_MARKERS)
        else:
            checks["no_fabricated_standard"] = True

        if not query_has_standard and FALSE_CERTAINTY_RE.search(text):
            checks["no_false_certainty"] = any(m in text for m in NEED_VERIFY_MARKERS)
        else:
            checks["no_false_certainty"] = True

    missing_data = constraints.get("missing_data") or any(
        k in q for k in ("数据不足", "未给出", "缺少", "未知水量", "未提供进水")
    )
    if missing_data or constraints.get("require_insufficient"):
        checks["uncertainty"] = text.lstrip().startswith("信息不足") or any(m in text[:40] for m in INSUFFICIENT_MARKERS)

    if constraints.get("require_assumptions") or ("假设" in q or "未给出" in q):
        checks["assumptions"] = any(k in text for k in ("假设", "若假设", "在缺少", "建议补测"))

    process_key = constraints.get("process_family")
    if process_key:
        keywords = PROCESS_KEYWORDS.get(process_key, ())
        low = text.lower()
        checks["process_match"] = any(k.lower() in low for k in keywords)

    if constraints.get("require_secondary_pollution"):
        checks["secondary_pollution"] = any(k in text for k in ("二次污染", "安全", "风险", "污泥", "废活性炭", "石膏", "氨逃逸"))

    if constraints.get("forbid_wastewater_formula_on_gas") or ("烟气量" in q and "负荷" in q):
        used_ww = bool(re.search(r"Q×C×0\.001|Q×\(C_in-C_out\)×0\.001", text))
        denied = any(k in text for k in ("不可把烟气量当成污水量", "不得套用废水", "不可套用废水"))
        checks["no_wastewater_formula_on_gas"] = (not used_ww) or denied

    passed = all(checks.values()) if checks else True
    return {"layer": "knowledge", "passed": passed, "checks": checks}


def evaluate_response(
    response: str,
    query: str = "",
    gold: Optional[Dict[str, Any]] = None,
    format_constraints: Optional[Dict[str, Any]] = None,
    knowledge_constraints: Optional[Dict[str, Any]] = None,
    rel_tol: float = 0.02,
) -> Dict[str, Any]:
    gold = gold or {}
    fmt = layer_format(
        response,
        format_constraints or gold.get("format_constraints"),
        query=query,
        gold=gold,
    )
    calc = layer_calculation(response, gold, rel_tol=rel_tol)
    know = layer_knowledge(response, query, knowledge_constraints or gold.get("knowledge_constraints"))
    out = {
        "format": fmt,
        "calculation": calc,
        "knowledge": know,
        "passed": bool(fmt["passed"] and calc["passed"] and know["passed"]),
    }
    out["error_types"] = classify_error_types(out, response=response, query=query, gold=gold)
    return out


def classify_error_types(
    result: Dict[str, Any],
    response: str = "",
    query: str = "",
    gold: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """把三层失败拆成可训练的错误类型，供 DPO 硬负样本标注。"""
    gold = gold or {}
    types: List[str] = []
    fmt = (result.get("format") or {}).get("checks") or {}
    calc = (result.get("calculation") or {}).get("checks") or {}
    know = (result.get("knowledge") or {}).get("checks") or {}
    if fmt.get("headers") is False or fmt.get("table") is False or fmt.get("three_causes") is False:
        types.append("format_error")
    if fmt.get("insufficient_prefix") is False:
        types.append("missing_insufficient")
    if (
        calc.get("removal_rate") is False
        or calc.get("load_kg_d") is False
        or calc.get("hrt_h") is False
        or calc.get("o2_corrected") is False
    ):
        types.append("calc_error")
    if calc.get("load_unit") is False:
        types.append("unit_error")
    if know.get("no_fabricated_standard") is False:
        types.append("fabricated_standard")
    if know.get("no_false_certainty") is False:
        types.append("false_certainty")
    if know.get("uncertainty") is False:
        types.append("false_certainty")
    if not types and not result.get("passed"):
        types.append("other")
    return sorted(set(types))


def score_response(result: Dict[str, Any]) -> float:
    """将三层结果折成 0~1，供 DPO 排序。计算层权重大于纯格式。"""
    weights = {"format": 0.25, "calculation": 0.45, "knowledge": 0.30}
    total = 0.0
    for key, w in weights.items():
        block = result.get(key) or {}
        checks = block.get("checks") or {}
        if block.get("skipped") or not checks:
            total += w
            continue
        vals = [1.0 if v else 0.0 for v in checks.values() if isinstance(v, bool)]
        total += w * (sum(vals) / len(vals) if vals else (1.0 if block.get("passed") else 0.0))
    return round(total, 4)


def instruction_constraints(instruction: str) -> Dict[str, Any]:
    """根据种子指令文本推断格式约束，供步骤7补充验证。"""
    text = instruction or ""
    cons: Dict[str, Any] = {}
    header_map = [
        (("污染来源", "迁移途径", "治理措施", "监测指标"), ["污染来源", "迁移途径", "治理措施", "监测指标"]),
        (("设计输入", "关键假设", "计算过程", "推荐方案", "风险点"), ["设计输入", "关键假设", "计算过程", "推荐方案", "风险点"]),
        (("分类", "暂存", "运输", "处理处置", "风险控制"), ["分类", "暂存", "运输", "处理处置", "风险控制"]),
        (("监测对象", "采样时间", "监测结果", "异常项", "结论"), ["监测对象", "采样时间", "监测结果", "异常项", "结论"]),
        (("作用对象", "基本原理", "预期效果", "局限性"), ["作用对象", "基本原理", "预期效果", "局限性"]),
        (("已知条件", "使用公式", "计算", "结论", "假设"), ["已知条件", "公式", "过程", "结论", "假设"]),
    ]
    for keys, headers in header_map:
        if all(k in text for k in keys[:2]) and any(k in text for k in keys[2:]):
            cons["required_headers"] = headers
            break
    if "表格" in text:
        cons["require_table"] = True
    if "→" in text or "箭头" in text:
        cons["require_arrow_process"] = True
        cons["min_units"] = 4
    if "三个可能原因" in text or "恰好三个" in text or "恰好 3" in text:
        cons["exactly_n_causes"] = 3
    if "信息不足" in text:
        cons["must_start_insufficient"] = True
    if "两位小数" in text:
        cons["require_two_decimals"] = True
    if "300" in text and "500" in text:
        cons["min_chars"] = 300
        cons["max_chars"] = 500
        cons["require_formula"] = True
        cons["require_table"] = True
    if "mg/L" in text or "μg/L" in text or "mg/m³" in text:
        cons["allowed_units"] = True
    return cons
