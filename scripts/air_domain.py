"""大气污染主线查询：计算、工艺、诊断、监测、烟气运行摘录。

与污水公式分离：负荷用 Qg×C×24×10^{-6}，另有基准氧折算。
"""
from __future__ import annotations

import random


TRAIN_QG = [6500, 8200, 11000, 14500, 19000, 24000, 32000]
TEST_QG = [7800, 9800, 13200, 20500, 28000]
AIR_SITES = ["临港燃煤热电", "河西烧结机头", "南山水泥窑", "东部垃圾焚烧", "园区工业锅炉", "焦化干熄焦"]
AIR_CASE_SITES = ["滨江热电烟气岛", "北郊烧结机尾", "西山水泥窑尾", "港区焚烧线"]


def o2_gold(c, o2m, o2s):
    corr = round(c * (21.0 - o2s) / (21.0 - o2m), 2)
    return {
        "params": {"C": c, "O2m": o2m, "O2s": o2s},
        "o2_corrected": corr,
        "require_two_decimals": True,
        "format_constraints": {"require_formula": True},
        "knowledge_constraints": {"forbid_fabricated_standard": True},
    }


def build_all_air(split: str, item, idx, gas_gold, process_gold) -> list:
    rows = []
    rows += _calculation(split, item, idx, gas_gold)
    rows += _process(split, item, idx, process_gold)
    rows += _diagnosis(split, item, idx)
    rows += _monitoring(split, item, idx)
    rows += _plant_cases(split, item, idx, gas_gold)
    return rows


def _qgs(split):
    return TRAIN_QG if split == "train" else TEST_QG


def _sites(split):
    return AIR_SITES if split == "train" else AIR_CASE_SITES


def _calculation(split, item, idx, gas_gold) -> list:
    qgs, sites = _qgs(split), _sites(split)
    n = 96 if split == "train" else 32
    rows = []
    for raw_i in range(n):
        i = idx(split, raw_i)
        site = sites[i % len(sites)]
        qg = qgs[i % len(qgs)]
        so2_in = 360 + (i * 17) % 520
        so2_out = 28 + (i % 22)
        if so2_out >= so2_in:
            so2_out = max(20, so2_in // 9)
        nox_in = 220 + (i * 11) % 360
        nox_out = 32 + (i % 18)
        if nox_out >= nox_in:
            nox_out = max(18, nox_in // 7)
        voc_in = 80 + (i * 23) % 400
        voc_out = max(10, int(voc_in * 0.12))
        o2m = round(7.2 + (i % 8) * 0.4, 1)
        o2s = 6.0 if i % 2 == 0 else 3.0
        c_meas = 180 + (i * 13) % 420
        styles = [
            (
                f"{site}烟气量{qg} m³/h，SO2进口{so2_in} mg/m³、出口{so2_out} mg/m³。"
                f"请计算含硫烟气去除率和每日去除负荷，公式用 Qg×ΔC×24×10^{-6}，不得套用废水公式。",
                gas_gold(qg, so2_in, so2_out, "so2"),
                "hard",
                ["大气污染", "SO2", "脱硫"],
            ),
            (
                f"{site}烟气量{qg} m³/h，NOx进口{nox_in} mg/m³、出口{nox_out} mg/m³。"
                f"请计算含氮烟气去除率和每日去除负荷，保留两位小数并做数量级检查。",
                gas_gold(qg, nox_in, nox_out, "nox"),
                "hard",
                ["大气污染", "NOx", "脱硝"],
            ),
            (
                f"{site}有组织废气VOCs进口{voc_in} mg/m³、出口{voc_out} mg/m³、风量{qg} m³/h。"
                f"请估算去除率和每日去除负荷，浓度单位只能用 mg/m³。",
                gas_gold(qg, voc_in, voc_out, "vocs"),
                "medium",
                ["大气污染", "VOCs"],
            ),
            (
                f"{site}实测SO2={c_meas} mg/m³，烟气含氧{o2m}%。请按基准氧{o2s:g}%折算，"
                f"写出 C'=C×(21−O2,s)/(21−O2,m)、代入和两位小数结果。未给排放标准不得判定达标。",
                o2_gold(c_meas, o2m, o2s),
                "hard",
                ["大气污染", "基准氧", "SO2"],
            ),
            (
                f"{site}烟气量{qg} m³/h，进口SO2 {so2_in} mg/m³。请计算日均进气负荷（kg/d），"
                f"必须使用 Qg×C×24×10^{-6}。",
                {
                    "params": {"Qg": qg, "C": so2_in},
                    "load_kg_d": round(qg * so2_in * 24 * 1e-6, 2),
                    "require_two_decimals": True,
                    "format_constraints": {"require_formula": True},
                    "knowledge_constraints": {
                        "forbid_fabricated_standard": True,
                        "forbid_wastewater_formula_on_gas": True,
                    },
                },
                "easy",
                ["大气污染", "SO2"],
            ),
            (
                f"【烟气周报】{site}本周均值：Qg={qg} m³/h，NOx 进/出={nox_in}/{nox_out} mg/m³。"
                f"周报未附排放标准。请核算去除率与 kg/d 负荷，不得把烟气量当成污水量。",
                gas_gold(qg, nox_in, nox_out, "nox"),
                "hard",
                ["大气污染", "NOx", "运行摘录"],
            ),
        ]
        query, gold, diff, knowledge = styles[i % len(styles)]
        query = query + f" 气账 {i:04d}。"
        src = "synthetic_plant_case" if "周报" in query else "template"
        rows.append(item(query, "air_calculation", diff, knowledge, gold, split, src))
    return rows


def _process(split, item, idx, process_gold) -> list:
    qgs, sites = _qgs(split), _sites(split)
    n = 72 if split == "train" else 24
    shops = ["喷漆车间", "印刷车间", "储罐区", "制药发酵"]
    rows = []
    for raw_i in range(n):
        i = idx(split, raw_i)
        site = sites[i % len(sites)]
        shop = shops[i % len(shops)]
        qg = qgs[i % len(qgs)]
        so2 = 400 + (i * 19) % 500
        nox = 250 + (i * 13) % 320
        voc = 90 + (i * 21) % 380
        variants = [
            (
                f"{site}拟对含硫烟气做脱硫，SO2约{so2} mg/m³、烟气量{qg} m³/h。"
                f"请比较石灰石-石膏湿法与半干法，按优先、备选、不推荐输出，并说明石膏副产物与运行安全。",
                process_gold("so2"),
                ["大气污染", "脱硫"],
            ),
            (
                f"{site}拟对含氮烟气做脱硝，NOx约{nox} mg/m³、烟气量{qg} m³/h。"
                f"请比较SCR与SNCR，按优先、备选、不推荐输出，必须说明氨逃逸与废催化剂。",
                process_gold("nox"),
                ["大气污染", "脱硝"],
            ),
            (
                f"{site}烟气同时含SO2约{so2} mg/m³和NOx约{nox} mg/m³，烟气量{qg} m³/h。"
                f"请分别给出脱硫与脱硝的优先、备选、不推荐方案，并单列二次污染与运行安全。",
                {"knowledge_constraints": {"forbid_fabricated_standard": True, "require_secondary_pollution": True}},
                ["大气污染", "SO2", "NOx"],
            ),
            (
                f"{shop}风量{qg} m³/h、VOCs约{voc} mg/m³。比较活性炭吸附与催化燃烧的适用条件、能耗和二次污染风险，按优先、备选、不推荐三级输出。",
                process_gold("vocs"),
                ["大气污染", "VOCs"],
            ),
            (
                f"拟用RTO处理{shop} {voc} mg/m³ VOCs、风量{qg} m³/h。请给出关键假设、推荐方案和风险点，不得虚构设备参数。",
                process_gold("vocs"),
                ["大气污染", "VOCs"],
            ),
            (
                f"{shop}恶臭处理：进气H2S约{3 + (i % 12)} mg/m³、风量{qg} m³/h。请从吸收、生物滤池和离子氧化中排序，按优先、备选、不推荐输出。",
                None,
                ["大气污染", "恶臭"],
            ),
        ]
        query, gold, knowledge = variants[i % len(variants)]
        query = query + f" 工艺账 {i:04d}。"
        rows.append(item(query, "air_process", "medium", knowledge, gold, split))
    return rows


def _diagnosis(split, item, idx) -> list:
    events = [
        "脱硫效率下降、出口SO2回升",
        "湿法吸收塔除雾器压差升高",
        "氨逃逸升高、出口嗅辨异常",
        "SCR床层压差快速上升",
        "SNCR温度窗口偏离、脱硝效率不稳",
        "石灰石给料中断后SO2短时峰值",
    ]
    qgs, sites = _qgs(split), _sites(split)
    n = 40 if split == "train" else 16
    rows = []
    for raw_i in range(n):
        i = idx(split, raw_i)
        ev = events[i % len(events)]
        site = sites[i % len(sites)]
        qg = qgs[i % len(qgs)]
        query = (
            f"{site}烟气治理系统出现{ev}。当前烟气量约{qg} m³/h。"
            f"请分析可能原因并提出排查步骤。必须列出恰好三个可能原因，每个原因包含“证据”和“复核方法”。"
        )
        rows.append(item(
            query, "air_diagnosis", "medium", ["大气污染", "异常诊断", ev],
            gold={"format_constraints": {"exactly_n_causes": 3}},
            split=split,
        ))
    return rows


def _monitoring(split, item, idx) -> list:
    rng = random.Random(21 if split == "train" else 121)
    sites = _sites(split)
    n = 40 if split == "train" else 16
    rows = []
    templates = [
        "请根据以下烟气监测数据编写报告摘要：对象为{site}烟囱，采样时间为{date}，SO2={so2} mg/m³，NOx={nox} mg/m³，含氧量={o2}%。摘要必须包含监测对象、采样时间、监测结果、异常项和结论。未提供排放标准时不得编造限值。",
        "{site}仅测得烟气SO2={so2} mg/m³，未给出烟气量、含氧量和排放标准。现有数据不足判断是否达标，请以“信息不足”开头并列出缺失参数。",
        "{site}除尘器后颗粒物为{dust} mg/m³。现有数据不足判断是否达标，请以“信息不足”开头并列出缺失参数。",
        "{site}CEMS显示SO2由{a} mg/m³骤升至{b} mg/m³。请诊断异常，必须列出恰好三个可能原因，每个原因包含证据和复核方法。",
    ]
    for raw_i in range(n):
        i = idx(split, raw_i)
        site = sites[i % len(sites)]
        so2 = 80 + (i * 17) % 400
        nox = 60 + (i * 11) % 280
        o2 = round(6.5 + (i % 7) * 0.3, 1)
        dust = 8 + (i % 15)
        a = so2
        b = so2 + 40 + (i % 30)
        query = templates[i % len(templates)].format(
            site=site,
            date=rng.choice(["2024-11-03 09:00", "2025-03-18 14:20", "2025-07-09 08:40"]),
            so2=so2, nox=nox, o2=o2, dust=dust, a=a, b=b,
        )
        gold = None
        cat = "air_pollution"
        if "报告摘要" in query:
            cat = "monitoring_report"
        elif "信息不足" in query:
            gold = {"format_constraints": {"must_start_insufficient": True}}
            cat = "air_calculation"
        elif "三个可能原因" in query:
            gold = {"format_constraints": {"exactly_n_causes": 3}}
            cat = "air_diagnosis"
        rows.append(item(query, cat, "medium", ["大气污染", "烟气监测"], gold, split))
    return rows


def _plant_cases(split, item, idx, gas_gold) -> list:
    qgs, sites = _qgs(split), _sites(split)
    n = 64 if split == "train" else 24
    rows = []
    for raw_i in range(n):
        i = idx(split, raw_i)
        site = sites[i % len(sites)]
        qg = qgs[i % len(qgs)]
        cin = 410 + (i * 19) % 480
        cout = 35 + (i % 24)
        if cout >= cin:
            cout = max(22, cin // 10)
        o2m = round(7.0 + (i % 6) * 0.3, 1)
        week = 8 + (i % 36)
        gold = gas_gold(qg, cin, cout, "so2" if i % 2 == 0 else "nox")
        pol = "SO2" if i % 2 == 0 else "NOx"
        variants = [
            (
                f"【{site}第{week}周烟气摘录】Qg={qg} m³/h；{pol} 进 {cin} mg/m³、出 {cout} mg/m³；"
                f"含氧 {o2m}%（未要求折算）。未附排放标准文本。请计算去除率与日去除负荷，保留两位小数。"
            ),
            (
                f"{site}CEMS导出：Qg={qg} m³/h | {pol}_in={cin} | {pol}_out={cout}（浓度单位 mg/m³）。"
                f"请用 Qg×ΔC×24×10^{-6} 完成 kg/d 核算，不得编造限值、不得套用废水公式。"
            ),
        ]
        query = variants[i % 2]
        rows.append(item(
            query, "air_calculation", "hard",
            ["运行摘录", "大气污染", pol], gold, split, source="synthetic_plant_case",
        ))
    return rows
