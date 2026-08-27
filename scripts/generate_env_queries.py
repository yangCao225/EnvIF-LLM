#!/usr/bin/env python3
"""生成环境工程领域查询集与独立测试集。"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TRAIN_Q = [5000, 8000, 10000, 12000, 15000, 20000, 25000, 30000, 40000, 50000]
TEST_Q = [6000, 9000, 11000, 18000, 22000, 35000, 45000, 80000]
TRAIN_C = [180, 220, 250, 280, 300, 320, 350, 400, 450, 500, 600]
TEST_C = [190, 240, 270, 330, 380, 420, 480, 550, 700]
PROCESSES = [
    ("A2/O", "氧化沟"),
    ("SBR", "MBR"),
    ("A/O", "接触氧化"),
    ("UASB+好氧", "A2/O"),
    ("高负荷活性污泥", "延时曝气"),
]
PLANTS = [
    "城东污水厂", "工业园区污水厂", "县城第二污水厂", "开发区再生水厂", "乡镇污水处理站",
    "印染园区综合污水厂", "食品加工废水站", "化工园预处理站", "空港新城水质净化厂", "湖滨再生水厂",
]
CASE_PLANTS = [
    "江北水质净化厂", "滨海工业园污水厂", "老城区合流制调蓄厂", "高新区再生水厂",
]


def idx(split: str, i: int) -> int:
    return i if split == "train" else i + 1000


def dumps(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def item(query, category, difficulty, knowledge, gold=None, split="train", source="template"):
    row = {
        "query": query,
        "category": category,
        "difficulty": difficulty,
        "required_knowledge": knowledge,
        "split": split,
        "source": source,
    }
    if gold:
        row["gold"] = gold
    return row


def calc_cod(q, cin, cout, plant, split, style=0):
    rate = round((cin - cout) / cin * 100, 2)
    load = round(q * (cin - cout) * 0.001, 2)
    gold = {
        "params": {"Q": q, "Cin": cin, "Cout": cout},
        "removal_rate_pct": rate,
        "load_kg_d": load,
        "require_two_decimals": True,
        "require_magnitude_check": True,
        "format_constraints": {"require_formula": True},
        "knowledge_constraints": {"forbid_fabricated_standard": True},
    }
    styles = [
        (
            f"{plant}设计水量为{q} m³/d，进水COD为{cin} mg/L，出水COD为{cout} mg/L，"
            f"请计算COD去除率和每日去除负荷，保留两位小数，并做数量级合理性检查。"
        ),
        (
            f"【运行周报】{plant}本周均值：水量 {q} m³/d，进水 COD {cin} mg/L，出水 COD {cout} mg/L。"
            f"请核算去除率和日去除负荷（kg/d），保留两位小数并做数量级检查。周报未附排放标准文本。"
        ),
        (
            f"设计校核任务书：{plant}规模 {q} m³/d。进水 COD={cin} mg/L，出水 COD={cout} mg/L。"
            f"要求写出公式、代入过程和 kg/d 负荷，不得编造排放限值。"
        ),
        (
            f"{plant}值班记录摘录：Q={q} m³/d；COD 进/出 = {cin}/{cout} mg/L；DO 与 MLSS 未同步给出。"
            f"请只根据已给水量和浓度计算 COD 去除率与每日去除负荷。"
        ),
    ]
    src = "synthetic_plant_case" if style % 4 else "template"
    return item(styles[style % 4], "wastewater_calculation", "medium", ["COD去除率", "污染负荷"], gold, split, src)


def calc_load(q, c, pollutant, split):
    load = round(q * c * 0.001, 2)
    gold = {"params": {"Q": q, "C": c}, "load_kg_d": load, "require_two_decimals": True}
    qtext = (
        f"某污水厂水量 {q} m³/d，进水{pollutant}浓度为 {c} mg/L。"
        f"请计算该污染物日均进水负荷（kg/d），给出水量、浓度、单位换算和结果。"
    )
    return item(qtext, "wastewater_calculation", "easy", ["污染负荷", pollutant], gold, split)


def calc_hrt(v, q, split):
    hrt = round(v / q * 24, 2)
    gold = {"params": {"V": v, "Q": q}, "hrt_h": hrt, "require_two_decimals": True}
    qtext = (
        f"曝气池有效容积为 {v} m³，设计水量为 {q} m³/d。"
        f"请计算水力停留时间（h），写出公式、代入过程和两位小数结果。"
    )
    return item(qtext, "wastewater_calculation", "easy", ["HRT"], gold, split)


def calc_fm(bod, q, v, mlss, split):
    fm = round((q * bod) / (v * mlss), 4)
    gold = {"params": {"Q": q, "Cin": bod, "V": v, "MLSS": mlss}, "fm": fm}
    qtext = (
        f"进水BOD5为 {bod} mg/L，水量 {q} m³/d，曝气池容积 {v} m³，MLSS为 {mlss} mg/L。"
        f"请计算污泥负荷 F/M（kgBOD/(kgMLSS·d)），并说明该负荷属于高、中还是低负荷。"
    )
    return item(qtext, "wastewater_calculation", "hard", ["F/M", "污泥负荷"], gold, split)


def calc_bod_cod(bod, cod, split):
    ratio = round(bod / cod, 2)
    gold = {"params": {"BOD": bod, "COD": cod}, "ratio": ratio}
    qtext = (
        f"进水BOD5为 {bod} mg/L，COD为 {cod} mg/L。"
        f"请计算BOD5/COD，判断可生化性，并说明对工艺选择的含义。"
    )
    return item(qtext, "wastewater_calculation", "medium", ["BOD/COD", "可生化性"], gold, split)


def build_calculation(split: str) -> list:
    qs = TRAIN_Q if split == "train" else TEST_Q
    cs = TRAIN_C if split == "train" else TEST_C
    rng = random.Random(42 if split == "train" else 2026)
    rows = []
    for i, q in enumerate(qs):
        cin = cs[i % len(cs)]
        cout = 30 if cin > 80 else 10
        if cin - cout < 50:
            cout = max(20, cin // 6)
        plant = PLANTS[i % len(PLANTS)]
        rows.append(calc_cod(q, cin, cout, plant, split, style=i))
        rows.append(calc_load(q, cs[(i + 3) % len(cs)], rng.choice(["TN", "NH3-N", "TP", "SS"]), split))
        v = int(q * rng.choice([0.25, 0.3, 0.35, 0.4]))
        rows.append(calc_hrt(v, q, split))
        if i % 2 == 0:
            rows.append(calc_bod_cod(cs[(i + 1) % len(cs)] // 2, cs[(i + 1) % len(cs)], split))
        if i % 3 == 0:
            rows.append(calc_fm(rng.choice([120, 150, 180, 200]), q, v, rng.choice([2500, 3000, 3500, 4000]), split))
    extra_pairs = []
    for qi, q in enumerate(qs):
        for cj, cin in enumerate(cs[:6]):
            cout = 40 + (qi + cj) * 3
            if cout >= cin:
                cout = max(15, cin // 8)
            extra_pairs.append((q, cin, cout, PLANTS[(qi + cj) % len(PLANTS)]))
    for q, cin, cout, plant in extra_pairs[: (96 if split == "train" else 28)]:
        rows.append(calc_cod(q, cin, cout, plant, split, style=q + cin))
    return rows


def build_process(split: str) -> list:
    rows = []
    n = 140 if split == "train" else 48
    q_list = TRAIN_Q if split == "train" else TEST_Q
    for raw_i in range(n):
        i = idx(split, raw_i)
        a, b = PROCESSES[i % len(PROCESSES)]
        plant = PLANTS[i % len(PLANTS)]
        q = q_list[i % len(q_list)]
        c = 180 + (i * 13) % 420
        tn = 18 + (i * 3) % 28
        tp = round(1.2 + (i % 12) * 0.35, 2)
        ratio = round(3.2 + (i % 15) * 0.4, 2)
        pct = 15 + (i * 5) % 50
        variants = [
            f"{plant}规模{q} m³/d。请比较{a}与{b}处理城镇污水的适用条件，必须使用表格，并包含处理效果、占地、能耗、污泥产量和适用条件五列。",
            f"{plant}拟将{a}改造为{b}，现状水量{q} m³/d，进水COD约{c} mg/L。请给出推荐方案，必须包含设计输入、关键假设、计算过程、推荐方案、风险点。",
            f"设计水量{q} m³/d、进水COD {c} mg/L。用“→”写出至少四个单元的{a}污水处理流程，并说明各单元主要去除对象。",
            f"{plant}进水TN={tn} mg/L、TP={tp} mg/L、水量{q} m³/d。请比较生物除磷与化学除磷，并分析污泥产量差异。",
            f"MBR用于{plant}尾水回用，设计水量{q} m³/d。请给出膜通量单位、污染控制措施和化学清洗条件。",
            f"{plant}硝化反硝化系统进水COD/TN约为{ratio}，水量{q} m³/d。请同时给出碳氮比判断和溶解氧控制要点。",
            f"{plant}深度处理段水量{q} m³/d。请比较砂滤、反硝化滤池和臭氧-活性炭，必须使用表格，包含去除对象、药剂、产泥和吨水电耗。",
            f"{plant}曝气池负荷冲击，进水COD短时升高约{pct}%（原COD {c} mg/L）。请给出允许波动判断和至少两条运行调整措施。",
            f"请为{plant}编写{a}运行手册摘要，列出恰好四个运行控制参数及其建议范围，水量按{q} m³/d计。",
            f"{plant}药剂除磷，PAC投加情景水量{q} m³/d、TP={tp} mg/L。请给出投加量单位和至少两项安全注意事项。",
        ]
        query = variants[i % len(variants)] + f" 台账编号 {i:04d}。"
        rows.append(item(query, "wastewater_process", "medium", ["工艺比较", a, b], split=split))
    return rows


def build_diagnosis(split: str) -> list:
    events = [
        "污泥膨胀、SVI升高和出水悬浮物增加",
        "出水氨氮升高、硝化效果下降",
        "二沉池大面积浮泥并伴有细碎气泡",
        "生物除磷失效、出水TP持续超标",
        "MBR跨膜压差快速上升",
        "氧化沟溶解氧长期偏低",
        "好氧池泡沫异常增多",
        "反硝化碳源不足导致TN升高",
        "进水pH突然下降",
        "低温导致COD去除率下降",
        "二沉池出水跑泥",
        "厌氧区ORP异常升高",
    ]
    rows = []
    n = 80 if split == "train" else 32
    for raw_i in range(n):
        i = idx(split, raw_i)
        ev = events[i % len(events)]
        plant = PLANTS[i % len(PLANTS)]
        svi = 120 + i * 7
        mlss = 1800 + i * 80
        t = 8 + (i % 18)
        ph = round(5.2 + (i % 10) * 0.2, 1)
        q = (
            f"{plant}活性污泥系统出现{ev}。当前SVI约{svi} mL/g，MLSS约{mlss} mg/L，水温{t}℃，pH={ph}。"
            f"请分析可能原因并提出排查步骤。必须列出恰好三个可能原因，每个原因包含“证据”和“复核方法”。"
        )
        rows.append(item(
            q, "process_diagnosis", "medium", ["异常诊断", ev],
            gold={"format_constraints": {"exactly_n_causes": 3}},
            split=split,
        ))
    return rows


def build_monitoring(split: str) -> list:
    rng = random.Random(11 if split == "train" else 111)
    indicators = ["COD", "NH3-N", "TN", "TP", "SS", "pH", "DO", "粪大肠菌群"]
    rows = []
    n = 120 if split == "train" else 44
    templates = [
        "请根据以下监测数据编写报告摘要：对象为{plant}出水，采样时间为{date}，COD={cod} mg/L，NH3-N={nh} mg/L，TP={tp} mg/L。摘要必须包含监测对象、采样时间、监测结果、异常项和结论。未提供排放标准时不得编造限值。",
        "{plant}在线COD由{a} mg/L骤升至{b} mg/L。请诊断异常，必须列出恰好三个可能原因，每个原因包含证据和复核方法。",
        "某河流断面{ind}浓度为{c} {unit}。请说明点位布设建议，至少列出三个点位名称及设置依据。",
        "实验室质控：空白样、平行样相对偏差{dev}%。请说明采样质控是否可接受，并包含空白样、平行样或加标回收中的至少两项。",
        "监测数据仅给出COD={cod} mg/L，未给出水量、采样时间和检测方法。请在信息不足的前提下分析，不得直接给出确定结论。",
        "将{ind}从 {a} mg/L 降至 {b} mg/L 的去除率是多少？保留两位小数并做数量级检查。",
        "某地下水监测井pH为{ph}。请按无量纲报出pH，不得添加 mg/L 单位，并说明是否需要复测。",
        "请比较自动在线监测与实验室标准方法在{ind}上的适用差异（对象{plant}），回答控制在300～500字，并包含一个公式和一个表格。",
    ]
    for raw_i in range(n):
        i = idx(split, raw_i)
        plant = rng.choice(PLANTS)
        ind = rng.choice(indicators)
        a = rng.choice(TRAIN_C if split == "train" else TEST_C)
        b = max(20, a // 6)
        query = templates[i % len(templates)].format(
            plant=plant, date=rng.choice(["2024-07-12 10:00", "2025-01-08 08:30", "2025-08-03 16:20"]),
            cod=a, nh=rng.choice([1.2, 4.8, 8.5, 12.0, 18.0]), tp=rng.choice([0.2, 0.6, 1.1, 2.4]),
            a=a, b=b, ind=ind, c=a, unit="mg/L", dev=rng.choice([2.1, 8.5, 15.0, 22.0]), ph=rng.choice([6.4, 7.1, 8.3, 9.2]),
        )
        gold = None
        if "去除率" in query:
            gold = {
                "params": {"Cin": a, "Cout": b},
                "removal_rate_pct": round((a - b) / a * 100, 2),
                "require_two_decimals": True,
                "require_magnitude_check": True,
            }
        if "未给出水量" in query:
            gold = {
                "format_constraints": {"must_start_insufficient": True},
                "knowledge_constraints": {"require_insufficient": True},
            }
        cat = "monitoring_report" if "报告摘要" in query else "monitoring_analysis"
        rows.append(item(query, cat, "medium", ["环境监测", ind], gold, split))
    return rows


def build_air(split: str) -> list:
    rows = []
    n = 90 if split == "train" else 32
    shops = ["喷漆车间", "印刷车间", "制药发酵", "橡胶硫化", "储罐区", "污水站除臭"]
    for raw_i in range(n):
        i = idx(split, raw_i)
        c = 70 + i * 45
        q = 6000 + i * 850
        nox = 60 + i * 12
        h2s = 3 + i
        dust = 6 + i * 2
        shop = shops[i % len(shops)]
        variants = [
            f"{shop}风量{q} m³/h、VOCs约{c} mg/m³。比较活性炭吸附与催化燃烧的适用条件、能耗和二次污染风险，按优先、备选、不推荐三级输出。",
            f"{shop}喷漆废气VOCs进口浓度{c} mg/m³、风量{q} m³/h。请估算去除负荷并评估爆炸下限风险，浓度单位只能用 mg/m³。",
            f"{shop}锅炉烟气NOx={nox} mg/m³、风量{q} m³/h。请比较SNCR与SCR，必须说明二次污染与运行安全；未给出排放标准时标注需核实现行标准。",
            f"{shop}恶臭处理：进气H2S约{h2s} mg/m³、风量{q} m³/h。请从吸收、生物滤池和离子氧化中排序，按优先、备选、不推荐输出。",
            f"拟用RTO处理{shop} {c} mg/m³ VOCs、风量{q} m³/h。请给出关键假设、推荐方案和风险点，不得虚构设备参数。",
            f"{shop}除尘器后颗粒物为{dust} mg/m³。现有数据不足判断是否达标，请以“信息不足”开头并列出缺失参数。",
        ]
        query = variants[i % len(variants)] + f" 台账编号 {i:04d}。"
        gold = None
        if "信息不足" in query:
            gold = {"format_constraints": {"must_start_insufficient": True}}
        if "去除负荷" in query:
            gold = {
                "params": {"Q": q * 24, "C": c},
                "load_kg_d": round(q * 24 * c * 1e-9 * 1000, 2),
            }
        rows.append(item(query, "air_pollution", "medium", ["大气污染", shop], gold, split))
    return rows


def build_solid(split: str) -> list:
    rng = random.Random(17 if split == "train" else 171)
    wastes = ["脱水污泥", "飞灰", "废活性炭", "实验室废液", "生活垃圾", "餐厨垃圾", "危险废物包装物", "隔油池浮渣"]
    n = 60 if split == "train" else 22
    rows = []
    for raw_i in range(n):
        i = idx(split, raw_i)
        w = wastes[i % len(wastes)]
        q = (
            f"某{rng.choice(PLANTS)}产生{w}约{rng.choice([2, 5, 12, 30])} t/d。"
            f"请分别说明分类、暂存、运输、处理处置和风险控制。"
        )
        rows.append(item(q, "solid_waste", "medium", ["固废", w], split=split))
    return rows


def build_noise(split: str) -> list:
    sources = ["鼓风机房", "冷却塔", "泵站", "空压机", "柴油发电机", "厂界道路"]
    n = 30 if split == "train" else 16
    rows = []
    for raw_i in range(n):
        i = idx(split, raw_i)
        s = sources[i % len(sources)]
        q = (
            f"{s}昼间测点声级为{55 + (i % 8) * 3} dB(A)，邻近居民区。"
            f"请从声源、传播路径和受声点三个层面提出噪声治理措施。"
        )
        rows.append(item(q, "noise_control", "easy", ["噪声", s], split=split))
    return rows


def build_eia(split: str) -> list:
    n = 60 if split == "train" else 24
    parks = ["精细化工园", "电镀园", "食品加工园", "印染园", "机械加工园", "综合产业园"]
    rows = []
    for raw_i in range(n):
        i = idx(split, raw_i)
        plant = PLANTS[i % len(PLANTS)]
        park = parks[i % len(parks)]
        water = 800 + i * 35
        reuse = 10 + (i * 3) % 60
        variants = [
            f"{park}拟扩建{plant}至{water} m³/d，位于敏感水体上游。请给出环评阶段需补充的监测与工艺论证要点；未提供标准依据时必须标注需核实现行标准。",
            f"请从清洁生产角度评估{plant}（规模{water} m³/d，回用率{reuse}%）的药剂投加、回用率和污泥减量潜力；涉及药剂时给出投加量单位和至少两项安全注意事项。",
            f"综合整治{park}水、气、固废问题，废水规模约{water} m³/d。每项治理措施必须包含作用对象、基本原理、预期效果、局限性。",
            f"{park}某企业仅提供出水COD数值，缺少当地执行标准年份和行业类别。请评估能否直接判定排放是否达标。",
            f"{plant}论证总量控制、浓度控制和生态环境质量目标的关系，设计规模{water} m³/d，并列出需补测参数清单。",
            f"{park}清洁生产审核发现单位产品新鲜水耗偏高（现状回用率{reuse}%）。请给出可执行的节水与分质回用建议，必须单列二次污染与运行安全。",
        ]
        query = variants[i % len(variants)] + f" 台账编号 {i:04d}。"
        gold = None
        if "缺少当地执行标准" in query:
            gold = {
                "format_constraints": {"must_start_insufficient": True},
                "knowledge_constraints": {"require_insufficient": True, "forbid_fabricated_standard": True},
            }
        rows.append(item(query, "eia_cleaner_production", "hard", ["环评", park], gold, split))
    return rows


def build_plant_cases(split: str) -> list:
    """多字段运行摘录，模拟厂站周报，而不是单句填空。"""
    qs = TRAIN_Q if split == "train" else TEST_Q
    cs = TRAIN_C if split == "train" else TEST_C
    plants = CASE_PLANTS if split == "test" else PLANTS
    n = 160 if split == "train" else 64
    rows = []
    rng = random.Random(9 if split == "train" else 99)
    for raw_i in range(n):
        i = idx(split, raw_i)
        plant = plants[i % len(plants)]
        q = qs[i % len(qs)]
        cin = cs[i % len(cs)]
        cout = max(15, cs[(i + 4) % len(cs)] // 8)
        if cout >= cin:
            cout = max(12, cin // 7)
        do = round(1.2 + (i % 8) * 0.4, 1)
        mlss = 2200 + (i * 60) % 1800
        svi = 80 + (i * 11) % 140
        week = 10 + (i % 40)
        gold = {
            "params": {"Q": q, "Cin": cin, "Cout": cout},
            "removal_rate_pct": round((cin - cout) / cin * 100, 2),
            "load_kg_d": round(q * (cin - cout) * 0.001, 2),
            "require_two_decimals": True,
            "require_magnitude_check": True,
            "knowledge_constraints": {"forbid_fabricated_standard": True},
        }
        variants = [
            (
                f"【{plant}第{week}周运行摘录】设计规模 {q} m³/d；进水COD {cin} mg/L（7日均值）；"
                f"出水COD {cout} mg/L；曝气池 DO {do} mg/L，MLSS {mlss} mg/L，SVI {svi} mL/g。"
                f"未附地方排放标准文本。请计算 COD 去除率与日去除负荷，保留两位小数并做数量级检查。"
            ),
            (
                f"{plant}中控导出：Q={q} m³/d | COD_in={cin} | COD_out={cout}（浓度单位 mg/L）。"
                f"同期 NH3-N、TN 传感器离线。请仅用 COD 与水量完成去除率和 kg/d 负荷核算，不得编造限值。"
            ),
            (
                f"现场交接班：{plant}水量按 {q} m³/d 计。化验室报进水 COD {cin} mg/L、出水 {cout} mg/L。"
                f"值班员认为“看起来已经达标”。请纠正该判断所需的计算，并说明未给标准时不能判定达标。"
            ),
        ]
        query = variants[i % 3]
        rows.append(item(
            query, "wastewater_calculation", "hard",
            ["运行摘录", "COD去除率"], gold, split, source="synthetic_plant_case",
        ))
    return rows


def dedupe(rows: list) -> list:
    seen = set()
    out = []
    for row in rows:
        key = row["query"]
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main():
    random.seed(42)
    train = []
    train += build_calculation("train")
    train += build_process("train")
    train += build_diagnosis("train")
    train += build_monitoring("train")
    train += build_air("train")
    train += build_solid("train")
    train += build_noise("train")
    train += build_eia("train")
    train += build_plant_cases("train")
    train = dedupe(train)

    test = []
    test += build_calculation("test")
    test += build_process("test")
    test += build_diagnosis("test")
    test += build_monitoring("test")
    test += build_air("test")
    test += build_solid("test")
    test += build_noise("test")
    test += build_eia("test")
    test += build_plant_cases("test")
    test = dedupe(test)

    train_queries = {r["query"] for r in train}
    test = [r for r in test if r["query"] not in train_queries]

    dumps(DATA_DIR / "environmental_engineering_queries.jsonl", train)
    dumps(DATA_DIR / "environmental_engineering_test.jsonl", test)

    from collections import Counter
    print(f"train={len(train)} test={len(test)}")
    print("train categories", dict(Counter(r["category"] for r in train)))
    print("test categories", dict(Counter(r["category"] for r in test)))
    print("train gold", sum(1 for r in train if r.get("gold")))
    print("test gold", sum(1 for r in test if r.get("gold")))
    print("train source", dict(Counter(r.get("source", "template") for r in train)))
    print("test source", dict(Counter(r.get("source", "template") for r in test)))


if __name__ == "__main__":
    main()
