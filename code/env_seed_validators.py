"""
为环境工程种子指令提供可执行的确定性验证函数。

AutoIF 步骤2仍会让教师模型生成验证函数；本模块把高确定性规则
（去除率字段、单位白名单、信息不足前缀等）注入交叉验证候选，
避免只检查“是否出现 COD 三个字母”。
"""
from __future__ import annotations

import textwrap
from typing import Dict, List


SEED_EVAL_FUNCS: Dict[str, str] = {
    "去除率": textwrap.dedent('''
        def evaluate(response):
            import re
            if not response:
                return False
            need = ["进水", "出水", "去除率"]
            if not all(k in response for k in need):
                return False
            if not re.search(r"[=（(].*%", response) and "%" not in response:
                return False
            return bool(re.search(r"\\d+(?:\\.\\d+)?\\s*%", response))
    ''').strip(),
    "浓度单位": textwrap.dedent('''
        def evaluate(response):
            import re
            if not response:
                return False
            allowed = ["mg/L", "μg/L", "ug/L", "mg/m³", "mg/m3"]
            if not any(u in response for u in allowed):
                return False
            if re.search(r"pH\\s*[:=为]?\\s*\\d+\\s*mg/L", response, re.I):
                return False
            return True
    ''').strip(),
    "污染负荷": textwrap.dedent('''
        def evaluate(response):
            import re
            if not response:
                return False
            if not any(k in response for k in ["水量", "流量", "m³/d", "m3/d"]):
                return False
            if "浓度" not in response and "mg/L" not in response:
                return False
            if not re.search(r"0\\.001|×\\s*10\\^-3|换算", response):
                return False
            return bool(re.search(r"kg\\s*/\\s*d|kg/d|千克/天", response, re.I))
    ''').strip(),
    "三个原因": textwrap.dedent('''
        def evaluate(response):
            if not response:
                return False
            if response.count("证据") < 3:
                return False
            if response.count("复核") < 1 and response.count("验证") < 1:
                return False
            markers = ["原因1", "原因2", "原因3", "1.", "1、", "（1）", "(1)"]
            return sum(m in response for m in markers) >= 1
    ''').strip(),
    "信息不足": textwrap.dedent('''
        def evaluate(response):
            if not response:
                return False
            return response.lstrip().startswith("信息不足")
    ''').strip(),
    "两位小数": textwrap.dedent('''
        def evaluate(response):
            import re
            if not response:
                return False
            if not re.search(r"\\d+\\.\\d{2}(?!\\d)", response):
                return False
            return any(k in response for k in ["数量级", "合理性", "量级", "是否合理"])
    ''').strip(),
    "四个标题": textwrap.dedent('''
        def evaluate(response):
            keys = ["污染来源", "迁移途径", "治理措施", "监测指标"]
            return all(k in (response or "") for k in keys)
    ''').strip(),
    "工艺表格": textwrap.dedent('''
        def evaluate(response):
            text = response or ""
            cols = ["处理效果", "占地", "能耗", "污泥产量", "适用条件"]
            has_table = "|" in text and text.count("|") >= 6
            return has_table and all(c in text for c in cols)
    ''').strip(),
    "箭头流程": textwrap.dedent('''
        def evaluate(response):
            text = (response or "").replace("->", "→")
            if "→" not in text:
                return False
            return len([x for x in text.split("→") if x.strip()]) >= 4
    ''').strip(),
    "五部分方案": textwrap.dedent('''
        def evaluate(response):
            keys = ["设计输入", "关键假设", "计算过程", "推荐方案", "风险点"]
            return all(k in (response or "") for k in keys)
    ''').strip(),
    "标准引用": textwrap.dedent('''
        def evaluate(response):
            text = response or ""
            if "标准" not in text and "GB" not in text and "HJ" not in text:
                return True
            return "需核实现行标准" in text or ("适用" in text and "限值" in text)
    ''').strip(),
    "监测报告": textwrap.dedent('''
        def evaluate(response):
            keys = ["监测对象", "采样时间", "监测结果", "异常项", "结论"]
            return all(k in (response or "") for k in keys)
    ''').strip(),
    "固废五段": textwrap.dedent('''
        def evaluate(response):
            keys = ["分类", "暂存", "运输", "处理处置", "风险控制"]
            return all(k in (response or "") for k in keys)
    ''').strip(),
    "噪声三层面": textwrap.dedent('''
        def evaluate(response):
            keys = ["声源", "传播路径", "受声点"]
            return all(k in (response or "") for k in keys)
    ''').strip(),
    "废气三级": textwrap.dedent('''
        def evaluate(response):
            keys = ["优先", "备选", "不推荐"]
            return all(k in (response or "") for k in keys)
    ''').strip(),
}


KEYWORD_TO_FUNC = [
    ("计算污染物去除率", "去除率"),
    ("进水浓度、出水浓度", "去除率"),
    ("单位只能从", "浓度单位"),
    ("污染负荷计算", "污染负荷"),
    ("恰好三个可能原因", "三个原因"),
    ("必须以“信息不足”开头", "信息不足"),
    ("保留两位小数", "两位小数"),
    ("污染来源、迁移途径", "四个标题"),
    ("处理效果、占地、能耗", "工艺表格"),
    ("“→”连接至少四个", "箭头流程"),
    ("设计输入、关键假设", "五部分方案"),
    ("需核实现行标准", "标准引用"),
    ("环境监测报告摘要", "监测报告"),
    ("分类、暂存、运输", "固废五段"),
    ("声源、传播路径和受声点", "噪声三层面"),
    ("优先、备选、不推荐", "废气三级"),
]


def funcs_for_instruction(instruction: str) -> List[str]:
    text = instruction or ""
    found = []
    for keyword, key in KEYWORD_TO_FUNC:
        if keyword in text and key in SEED_EVAL_FUNCS:
            found.append(SEED_EVAL_FUNCS[key])
    return found


def all_seed_eval_funcs() -> List[str]:
    return list(SEED_EVAL_FUNCS.values())
