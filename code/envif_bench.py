"""EnvIF-Bench：环境工程领域指令遵循评测。

约束按 IFEval/FollowBench 思路分成五类：
  content / format / numerical / domain / multi-constraint

每条样本含若干原子约束；`auto_verifiable=true` 的约束有程序 verifier。
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from env_validators import (
    FALSE_CERTAINTY_RE,
    NEED_VERIFY_MARKERS,
    STANDARD_HINT_RE,
    count_cjk_chars,
    extract_floats,
    extract_quantities,
    has_table,
    nearly_equal,
    values_for_units,
)

CONSTRAINT_FAMILIES = (
    "content",
    "format",
    "numerical",
    "domain",
    "multi",
)

TASKS = (
    "wastewater_treatment",
    "environmental_monitoring",
    "pollution_control",
    "environmental_calculation",
    "safety_constraint",
    "format_constraint",
    "multi_step_reasoning",
    "terminology_explanation",
)

Verifier = Callable[[str, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


def _ok(passed: bool, **extra) -> Dict[str, Any]:
    return {"passed": bool(passed), **extra}


def v_content_must_include(response: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    terms = args.get("terms") or []
    missing = [t for t in terms if t not in (response or "")]
    return _ok(not missing, missing=missing)


def v_content_explain_term(response: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    text = response or ""
    term = args.get("term") or ""
    facets = args.get("facets") or ["定义", "原理", "适用"]
    min_facets = int(args.get("min_facets", 2))
    has_term = term in text
    hits = [f for f in facets if f in text]
    aliases = args.get("aliases") or []
    if aliases:
        has_term = has_term or any(a in text for a in aliases)
    return _ok(has_term and len(hits) >= min_facets, term=has_term, facets=hits)


def v_content_missing_params(response: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    need = args.get("params") or []
    text = response or ""
    missing = [p for p in need if p not in text]
    return _ok(not missing, missing=missing)


def v_format_headers(response: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    headers = args.get("headers") or []
    missing = [h for h in headers if h not in (response or "")]
    return _ok(not missing, missing=missing)


def v_format_table(response: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    return _ok(has_table(response or ""))


def v_format_arrow_process(response: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    text = (response or "").replace("->", "→")
    min_units = int(args.get("min_units", 4))
    units = [u.strip() for u in text.split("→") if u.strip()]
    return _ok("→" in text and len(units) >= min_units, n_units=len(units))


def v_format_insufficient_prefix(response: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    return _ok((response or "").lstrip().startswith("信息不足"))


def v_format_length(response: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    n = count_cjk_chars(response or "")
    lo, hi = int(args.get("min_chars", 0)), int(args.get("max_chars", 10**9))
    return _ok(lo <= n <= hi, n_chars=n)


def v_format_n_causes(response: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    n = int(args.get("n", 3))
    text = response or ""
    hits = len(re.findall(r"(原因\s*[123一二三]|可能原因\s*[123一二三]|[1-3][\.、])", text))
    evidence = text.count("证据") >= n
    review = ("复核" in text) or ("验证方法" in text)
    return _ok(hits >= n and evidence and review, hits=hits)


def v_format_n_steps(response: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    n = int(args.get("n", 5))
    text = response or ""
    hits = len(re.findall(rf"(?:^|\n)\s*(?:步骤\s*)?[1-{n}][\.、\)）]", text))
    if hits < n:
        hits = len(re.findall(r"(?:步骤\s*[一二三四五1-5])", text))
    return _ok(hits >= n, hits=hits)


def _close(values: List[float], target: float, rel=0.02, abs_tol=0.05) -> bool:
    return any(nearly_equal(v, target, rel_tol=rel, abs_tol=abs_tol) for v in values)


def v_num_value(response: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    target = float(args["value"])
    unit = args.get("unit")
    rel = float(args.get("rel_tol", 0.02))
    abs_tol = float(args.get("abs_tol", 0.05))
    quants = extract_quantities(response or "")
    if unit == "%":
        pool = values_for_units(quants, "%") or extract_floats(response or "")
        abs_tol = max(abs_tol, 0.1)
    elif unit == "kg/d":
        kgd = values_for_units(quants, "kg/d")
        td = [v * 1000.0 for v in values_for_units(quants, "t/d")]
        pool = kgd + td
        abs_tol = max(abs_tol, 1.0)
        if not pool:
            return _ok(False, reason="missing_mass_time_unit")
    elif unit == "h":
        pool = values_for_units(quants, "h") or extract_floats(response or "")
        abs_tol = max(abs_tol, 0.1)
    else:
        pool = extract_floats(response or "")
    return _ok(_close(pool, target, rel, abs_tol), target=target)


def v_num_two_decimals(response: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    return _ok(bool(re.search(r"\d+\.\d{2}(?!\d)", response or "")))


def v_num_formula(response: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    return _ok(bool(re.search(r"(=|×|x|\*|÷|/)", response or "")))


def v_num_magnitude(response: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    text = response or ""
    return _ok(any(k in text for k in ("数量级", "合理性", "量级检查", "是否合理")))


def v_domain_no_fabricated_standard(response: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    text, query = response or "", ctx.get("query") or ""
    mentions = bool(STANDARD_HINT_RE.search(text))
    query_has = bool(STANDARD_HINT_RE.search(query)) or any(k in query for k in ("标准限值", "排放限值", "给出限值"))
    if mentions and not query_has:
        return _ok(any(m in text for m in NEED_VERIFY_MARKERS))
    return _ok(True)


def v_domain_no_false_certainty(response: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    text, query = response or "", ctx.get("query") or ""
    query_has = bool(STANDARD_HINT_RE.search(query)) or any(k in query for k in ("标准限值", "排放限值", "给出限值"))
    if not query_has and FALSE_CERTAINTY_RE.search(text):
        return _ok(any(m in text for m in NEED_VERIFY_MARKERS))
    return _ok(True)


def v_domain_allowed_units(response: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    text = response or ""
    banned = bool(re.search(r"\d+\s*(ppm(?!\w)|ppt)", text, re.IGNORECASE))
    need = args.get("must_use") or []
    has_need = all(u in text or u.replace("μ", "u") in text for u in need) if need else True
    return _ok((not banned) and has_need)


def v_domain_safety(response: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    text = response or ""
    keys = args.get("terms") or ("安全", "防护", "二次污染", "风险")
    hits = [k for k in keys if k in text]
    return _ok(len(hits) >= int(args.get("min_hits", 2)), hits=hits)


def v_domain_ph_dimensionless(response: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    text = response or ""
    bad = bool(re.search(r"pH\s*[:=为]?\s*\d+(?:\.\d+)?\s*mg\s*/\s*L", text, re.IGNORECASE))
    return _ok((not bad) and ("pH" in text or "酸碱" in text))


def v_domain_process_match(response: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    keys = args.get("keywords") or []
    low = (response or "").lower()
    return _ok(any(k.lower() in low for k in keys))


VERIFIERS: Dict[str, Verifier] = {
    "content.must_include": v_content_must_include,
    "content.explain_term": v_content_explain_term,
    "content.missing_params": v_content_missing_params,
    "format.headers": v_format_headers,
    "format.table": v_format_table,
    "format.arrow_process": v_format_arrow_process,
    "format.insufficient_prefix": v_format_insufficient_prefix,
    "format.length": v_format_length,
    "format.n_causes": v_format_n_causes,
    "format.n_steps": v_format_n_steps,
    "numerical.value": v_num_value,
    "numerical.two_decimals": v_num_two_decimals,
    "numerical.formula": v_num_formula,
    "numerical.magnitude": v_num_magnitude,
    "domain.no_fabricated_standard": v_domain_no_fabricated_standard,
    "domain.no_false_certainty": v_domain_no_false_certainty,
    "domain.allowed_units": v_domain_allowed_units,
    "domain.safety": v_domain_safety,
    "domain.ph_dimensionless": v_domain_ph_dimensionless,
    "domain.process_match": v_domain_process_match,
}


def family_of(constraints: List[Dict[str, Any]]) -> str:
    types = sorted({c.get("type") for c in constraints if c.get("type")})
    if len(types) == 0:
        return "content"
    if len(types) == 1:
        return types[0]
    return "multi"


def verify_constraint(response: str, constraint: Dict[str, Any], query: str = "") -> Dict[str, Any]:
    code = constraint.get("code") or ""
    fn = VERIFIERS.get(code)
    if fn is None:
        return {"code": code, "type": constraint.get("type"), "passed": None, "auto": False, "reason": "no_verifier"}
    out = fn(response, constraint.get("args") or {}, {"query": query})
    return {
        "code": code,
        "type": constraint.get("type"),
        "passed": bool(out.get("passed")),
        "auto": True,
        "detail": {k: v for k, v in out.items() if k != "passed"},
    }


def evaluate_item(item: Dict[str, Any], response: str) -> Dict[str, Any]:
    query = item.get("query") or ""
    auto_cons = [c for c in item.get("constraints") or [] if c.get("code") in VERIFIERS]
    results = [verify_constraint(response, c, query) for c in auto_cons]
    auto_pass = [r["passed"] for r in results if r["passed"] is not None]
    prompt_ok = bool(auto_pass) and all(auto_pass)
    return {
        "id": item.get("id"),
        "task": item.get("task"),
        "constraint_family": item.get("constraint_family"),
        "auto_verifiable": bool(item.get("auto_verifiable", True)),
        "prompt_passed": prompt_ok if item.get("auto_verifiable", True) else None,
        "n_auto_constraints": len(auto_cons),
        "n_auto_passed": sum(1 for x in auto_pass if x),
        "constraint_results": results,
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    auto_rows = [r for r in rows if r.get("auto_verifiable") and r.get("prompt_passed") is not None]
    by_fam = defaultdict(list)
    by_task = defaultdict(list)
    by_code = defaultdict(list)
    cons_all = []
    for r in auto_rows:
        by_fam[r.get("constraint_family") or "?"].append(1.0 if r["prompt_passed"] else 0.0)
        by_task[r.get("task") or "?"].append(1.0 if r["prompt_passed"] else 0.0)
        for cr in r.get("constraint_results") or []:
            if cr.get("passed") is None:
                continue
            val = 1.0 if cr["passed"] else 0.0
            by_code[cr["code"]].append(val)
            cons_all.append(val)

    def avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "n_total": len(rows),
        "n_auto": len(auto_rows),
        "prompt_accuracy": avg([1.0 if r["prompt_passed"] else 0.0 for r in auto_rows]),
        "constraint_accuracy": avg(cons_all),
        "by_family": {k: avg(v) for k, v in sorted(by_fam.items())},
        "by_task": {k: avg(v) for k, v in sorted(by_task.items())},
        "by_verifier": {k: avg(v) for k, v in sorted(by_code.items())},
    }
