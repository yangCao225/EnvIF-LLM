#!/usr/bin/env python3
"""生成 EnvIF-Bench。查询与训练/测试 JSONL 去重。"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from env_validators import count_cjk_chars, dry_o2_correct, gas_load_kg_d, hrt_hours, pollution_load_kg_d, removal_rate  # noqa: E402
from envif_bench import TASKS, VERIFIERS, family_of  # noqa: E402

OUT_DIR = ROOT / "data" / "envif_bench"
BENCH_Q = [7000, 13000, 17000, 28000, 36000, 55000, 72000]
BENCH_C = [210, 260, 310, 370, 440, 520, 650]
PLANTS = ["临江水质净化厂", "北郊工业废水站", "高新区再生水厂", "滨湖污水厂", "空港预处理站"]


def load_blocked_queries() -> set:
    blocked = set()
    for name in ("environmental_engineering_queries.jsonl", "environmental_engineering_test.jsonl"):
        p = ROOT / "data" / name
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    blocked.add(json.loads(line)["query"].strip())
    return blocked


def C(code: str, typ: str, **args) -> dict:
    if code not in VERIFIERS:
        raise KeyError(code)
    return {"code": code, "type": typ, "args": args}


def item(task: str, query: str, instruction: str, constraints: list, reference: str,
         auto: bool = True, gold: dict | None = None, note: str = "") -> dict:
    fam = family_of(constraints) if constraints else "content"
    prompt = f"{query.strip()}\n\n【指令约束】{instruction.strip()}"
    row = {
        "task": task,
        "constraint_family": fam if auto else fam,
        "query": query.strip(),
        "instruction": instruction.strip(),
        "prompt": prompt,
        "constraints": constraints,
        "auto_verifiable": auto,
        "reference": reference.strip(),
        "note": note,
    }
    if gold:
        row["gold"] = gold
    if not auto:
        row["constraint_family"] = row["constraint_family"]
        row["note"] = note or "无自动 verifier，留给人工评测语义质量。"
    return row


def numerical_items() -> list:
    rows = []
    for i, q in enumerate(BENCH_Q):
        cin = BENCH_C[i % len(BENCH_C)]
        cout = max(18, cin // 7)
        plant = PLANTS[i % len(PLANTS)]
        rate = round(removal_rate(cin, cout), 2)
        load = round(pollution_load_kg_d(q, cin - cout), 2)
        gold = {"params": {"Q": q, "Cin": cin, "Cout": cout}, "removal_rate_pct": rate, "load_kg_d": load}
        ref = (
            f"已知进水{cin} mg/L、出水{cout} mg/L、水量{q} m³/d。"
            f"去除率=({cin}-{cout})/{cin}×100%={rate:.2f}%。"
            f"每日去除负荷={q}×({cin}-{cout})×0.001={load:.2f} kg/d。"
            f"数量级合理性检查通过。未提供排放标准，需核实现行标准。"
        )
        rows.append(item(
            "environmental_calculation",
            f"{plant}水量 {q} m³/d，进水COD {cin} mg/L，出水COD {cout} mg/L。请计算COD去除率和日去除负荷。",
            "必须写出公式；结果保留两位小数；负荷单位必须是 kg/d；结尾做数量级检查。",
            [
                C("numerical.value", "numerical", value=rate, unit="%"),
                C("numerical.value", "numerical", value=load, unit="kg/d"),
                C("numerical.two_decimals", "numerical"),
                C("numerical.formula", "numerical"),
                C("numerical.magnitude", "numerical"),
            ],
            ref, gold=gold,
        ))
        c = BENCH_C[(i + 2) % len(BENCH_C)]
        lin = round(pollution_load_kg_d(q, c), 2)
        pol = ["TN", "NH3-N", "TP", "SS"][i % 4]
        rows.append(item(
            "environmental_calculation",
            f"{plant}水量 {q} m³/d，进水{pol} {c} mg/L。请计算日均进水负荷。",
            "必须给出水量、浓度、0.001 换算和 kg/d 结果，保留两位小数。",
            [
                C("numerical.value", "numerical", value=lin, unit="kg/d"),
                C("numerical.two_decimals", "numerical"),
                C("numerical.formula", "numerical"),
            ],
            f"水量{q} m³/d，浓度{c} mg/L。负荷={q}×{c}×0.001={lin:.2f} kg/d。",
            gold={"params": {"Q": q, "C": c}, "load_kg_d": lin},
        ))
        v = int(q * (0.22 + (i % 4) * 0.03))
        hrt = round(hrt_hours(v, q), 2)
        rows.append(item(
            "environmental_calculation",
            f"{plant}曝气池有效容积 {v} m³，设计水量 {q} m³/d。请计算水力停留时间（h）。",
            "必须给出 HRT=V/Q×24 的代入过程，结果以小时计并保留两位小数。",
            [
                C("numerical.value", "numerical", value=hrt, unit="h"),
                C("numerical.formula", "numerical"),
                C("numerical.two_decimals", "numerical"),
            ],
            f"V={v} m³，Q={q} m³/d。HRT=V/Q×24={v}/{q}×24={hrt:.2f} h。",
            gold={"params": {"V": v, "Q": q}, "hrt_h": hrt},
        ))
    # 含硫 / 含氮烟气：与训练集台账去重，公式为 Qg×ΔC×24×10^{-6}
    so2_qg, so2_in, so2_out = 9300, 820, 65
    so2_rate = round(removal_rate(so2_in, so2_out), 2)
    so2_load = round(gas_load_kg_d(so2_qg, so2_in - so2_out), 2)
    rows.append(item(
        "air_pollution",
        f"临海热电厂烟气量 {so2_qg} m³/h，进口 SO2 {so2_in} mg/m³，出口 SO2 {so2_out} mg/m³。请计算去除率和每日去除负荷。",
        "必须写出 Qg×ΔC×24×10^{-6}；结果保留两位小数；负荷单位必须是 kg/d；不得套用废水公式。",
        [
            C("numerical.value", "numerical", value=so2_rate, unit="%"),
            C("numerical.value", "numerical", value=so2_load, unit="kg/d"),
            C("numerical.two_decimals", "numerical"),
            C("numerical.formula", "numerical"),
        ],
        f"已知进口{so2_in} mg/m³、出口{so2_out} mg/m³、烟气量Qg={so2_qg} m³/h。"
        f"去除率=({so2_in}-{so2_out})/{so2_in}×100%={so2_rate:.2f}%。"
        f"每日去除负荷={so2_qg}×({so2_in}-{so2_out})×24×10^{-6}={so2_load:.2f} kg/d。"
        f"未提供排放标准，需核实现行标准。",
        gold={"params": {"Qg": so2_qg, "Cin": so2_in, "Cout": so2_out}, "removal_rate_pct": so2_rate, "load_kg_d": so2_load},
    ))
    nox_qg, nox_in, nox_out = 16800, 410, 48
    nox_rate = round(removal_rate(nox_in, nox_out), 2)
    nox_load = round(gas_load_kg_d(nox_qg, nox_in - nox_out), 2)
    rows.append(item(
        "air_pollution",
        f"北港烧结车间烟气量 {nox_qg} m³/h，进口 NOx {nox_in} mg/m³，出口 NOx {nox_out} mg/m³。请计算去除率和每日去除负荷。",
        "必须写出烟气负荷公式；结果保留两位小数；负荷单位 kg/d。",
        [
            C("numerical.value", "numerical", value=nox_rate, unit="%"),
            C("numerical.value", "numerical", value=nox_load, unit="kg/d"),
            C("numerical.two_decimals", "numerical"),
            C("numerical.formula", "numerical"),
        ],
        f"已知进口{nox_in} mg/m³、出口{nox_out} mg/m³、烟气量Qg={nox_qg} m³/h。"
        f"去除率=({nox_in}-{nox_out})/{nox_in}×100%={nox_rate:.2f}%。"
        f"每日去除负荷={nox_qg}×({nox_in}-{nox_out})×24×10^{-6}={nox_load:.2f} kg/d。",
        gold={"params": {"Qg": nox_qg, "Cin": nox_in, "Cout": nox_out}, "removal_rate_pct": nox_rate, "load_kg_d": nox_load},
    ))
    so2_inlet_qg, so2_c = 25500, 640
    so2_inlet = round(gas_load_kg_d(so2_inlet_qg, so2_c), 2)
    rows.append(item(
        "air_pollution",
        f"滨江工业锅炉烟气量 {so2_inlet_qg} m³/h，进口 SO2 {so2_c} mg/m³。请计算日均进气负荷。",
        "必须给出 Qg、浓度、24×10^{-6} 换算和 kg/d 结果，保留两位小数。",
        [
            C("numerical.value", "numerical", value=so2_inlet, unit="kg/d"),
            C("numerical.two_decimals", "numerical"),
            C("numerical.formula", "numerical"),
        ],
        f"烟气量Qg={so2_inlet_qg} m³/h，浓度C={so2_c} mg/m³。"
        f"负荷=Qg×C×24×10^{-6}={so2_inlet_qg}×{so2_c}×24×10^{-6}={so2_inlet:.2f} kg/d。",
        gold={"params": {"Qg": so2_inlet_qg, "C": so2_c}, "load_kg_d": so2_inlet},
    ))
    o2_c, o2m, o2s = 260, 8.5, 6.0
    o2_corr = round(dry_o2_correct(o2_c, o2m, o2s), 2)
    rows.append(item(
        "air_pollution",
        f"望海热电实测 SO2={o2_c} mg/m³，烟气含氧 {o2m}%。请按基准氧 {o2s:g}% 折算。",
        "必须写出 C'=C×(21−O2,s)/(21−O2,m)；结果保留两位小数；同时给出实测值；不得编造排放限值。",
        [
            C("numerical.value", "numerical", value=o2_corr, unit="mg/m3"),
            C("numerical.two_decimals", "numerical"),
            C("numerical.formula", "numerical"),
            C("domain.no_fabricated_standard", "domain"),
        ],
        f"公式 C'=C×(21−O2,s)/(21−O2,m)={o2_c}×(21−{o2s:g})/(21−{o2m})={o2_corr:.2f} mg/m³。"
        f"实测浓度 {o2_c} mg/m³。未提供排放标准，需核实现行标准。",
        gold={"params": {"C": o2_c, "O2m": o2m, "O2s": o2s}, "o2_corrected": o2_corr},
    ))
    # BOD/COD and F/M
    for i, (bod, cod) in enumerate([(90, 260), (140, 370), (110, 440), (160, 520)]):
        ratio = round(bod / cod, 2)
        plant = PLANTS[i]
        rows.append(item(
            "environmental_calculation",
            f"{plant}进水 BOD5={bod} mg/L，COD={cod} mg/L。请计算 BOD5/COD 并判断可生化性。",
            "必须给出比值（两位小数）、可生化性结论，以及该结论对工艺选择的含义。",
            [
                C("numerical.value", "numerical", value=ratio),
                C("numerical.two_decimals", "numerical"),
                C("content.must_include", "content", terms=["可生化"]),
            ],
            f"BOD5/COD={bod}/{cod}={ratio:.2f}。可生化性{'较好，宜优先生物处理' if ratio >= 0.3 else '偏差，需评估预处理'}。",
            gold={"params": {"BOD": bod, "COD": cod}, "ratio": ratio},
        ))
    for i, (bod, q, v, mlss) in enumerate([(150, 13000, 3900, 3000), (180, 28000, 8400, 3500)]):
        fm = round((q * bod) / (v * mlss), 4)
        rows.append(item(
            "environmental_calculation",
            f"进水BOD5 {bod} mg/L，水量 {q} m³/d，曝气池 {v} m³，MLSS {mlss} mg/L。请计算 F/M。",
            "必须给出 kgBOD/(kgMLSS·d) 结果，并判断高/中/低负荷。",
            [
                C("numerical.value", "numerical", value=fm, rel_tol=0.03, abs_tol=0.002),
                C("content.must_include", "content", terms=["负荷"]),
            ],
            f"F/M=(Q×BOD)/(V×MLSS)=({q}×{bod})/({v}×{mlss})={fm:.4f} kgBOD/(kgMLSS·d)，属中负荷。",
            gold={"params": {"Q": q, "Cin": bod, "V": v, "MLSS": mlss}, "fm": fm},
        ))
    return rows


def format_items() -> list:
    rows = []
    rows.append(item(
        "format_constraint",
        "请比较氧化沟与 SBR 处理城镇污水的适用性。",
        "必须使用表格，并包含处理效果、占地、能耗、污泥产量、适用条件五列。",
        [
            C("format.table", "format"),
            C("format.headers", "format", headers=["处理效果", "占地", "能耗", "污泥产量", "适用条件"]),
        ],
        "| 项目 | 处理效果 | 占地 | 能耗 | 污泥产量 | 适用条件 |\n| --- | --- | --- | --- | --- | --- |\n"
        "| 氧化沟 | 较稳 | 大 | 中 | 中 | 中小城镇 |\n| SBR | 灵活 | 中 | 中 | 中 | 水量波动大 |",
    ))
    rows.append(item(
        "wastewater_treatment",
        "请写出城镇污水 A2/O 的处理流程。",
        "必须使用“→”连接至少四个处理单元。",
        [C("format.arrow_process", "format", min_units=4)],
        "格栅→沉砂池→A2/O生化池→二沉池→消毒。",
    ))
    rows.append(item(
        "wastewater_treatment",
        "临江水质净化厂出现污泥膨胀、SVI 升高、出水 SS 增加。请做故障诊断。",
        "必须列出恰好三个可能原因，每个原因包含“证据”和“复核方法”。",
        [C("format.n_causes", "format", n=3)],
        "原因1：丝状菌膨胀。证据：SVI升高。复核方法：镜检。\n"
        "原因2：负荷冲击。证据：出水SS增加。复核方法：核算污泥负荷。\n"
        "原因3：营养比失调。证据：絮体松散。复核方法：复核N、P。",
    ))
    rows.append(item(
        "safety_constraint",
        "加氯间泄漏时请给出应急处置步骤。",
        "必须用有序列表给出恰好五个步骤。",
        [C("format.n_steps", "format", n=5)],
        "1. 疏散下风向人员。\n2. 佩戴防护进入。\n3. 切断氯源。\n4. 用碱液喷淋吸收。\n5. 报告并监测余氯。",
    ))
    rows.append(item(
        "environmental_monitoring",
        "仅给出出水 COD=80 mg/L，未给水量、采样时间和执行标准。请判断是否达标。",
        "回答必须以“信息不足”开头，并列出缺失参数，不得直接给出确定结论。",
        [
            C("format.insufficient_prefix", "format"),
            C("content.missing_params", "content", params=["水量", "采样时间"]),
            C("domain.no_false_certainty", "domain"),
        ],
        "信息不足。缺少水量、采样时间、检测方法及排放标准年份/适用对象，不能给出确定结论，需核实现行标准。",
    ))
    rows.append(item(
        "format_constraint",
        "请用四段结构说明某污水厂提标改造要点：污染来源、迁移途径、治理措施、监测指标。",
        "回答必须包含这四个标题，总长度控制在 300～500 字。",
        [
            C("format.headers", "format", headers=["污染来源", "迁移途径", "治理措施", "监测指标"]),
            C("format.length", "format", min_chars=300, max_chars=500),
        ],
        "污染来源：生活污水与部分工业废水进入管网，负荷随早高峰波动。"
        "迁移途径：污染物经管网进入厂区，在预处理、生化和二沉中迁移转化，部分随污泥排出。"
        "治理措施：强化预处理与碳源调控，稳定硝化反硝化，必要时化学除磷。"
        "监测指标：进出水 COD、NH3-N、TN、TP、SS 及 DO、MLSS、SVI。"
        + "补充运行说明。" * 40,
    ))
    # trim length item reference to actually be 300-500 CJK
    # I'll fix after counting - the *40 might overshoot. Build a proper 350-char block.
    rows[-1]["reference"] = (
        "污染来源：服务范围内以生活污水为主，混有食品加工废水，早高峰 COD 和氨氮明显抬升，雨季存在稀释与溢流风险。"
        "迁移途径：污染物经管网输送进入厂区，依次经过格栅沉砂、生化反应和二沉分离，溶解性物质在微生物代谢中转化，残渣进入污泥系统。"
        "治理措施：保证缺氧区碳源与好氧区溶解氧，稳定硝化反硝化；总磷偏高时辅以化学除磷；污泥及时排放避免上浮。"
        "监测指标：进出水 COD、氨氮、总氮、总磷、SS，以及曝气池 DO、MLSS、SVI 和二沉池泥位，用于判断负荷与分离效果。"
    )
    pad = "结合季节水温变化，需同步调整曝气强度与内回流，避免仅凭单项指标做结论。"
    while count_cjk_chars(rows[-1]["reference"]) < 320:
        rows[-1]["reference"] += pad
    # 若超过 500 字则截到句末附近
    ref = rows[-1]["reference"]
    n = count_cjk_chars(ref)
    if n > 480:
        rows[-1]["reference"] = ref[: int(len(ref) * 460 / n)]
    return rows


def content_items() -> list:
    rows = []
    terms = [
        ("水力停留时间", ["HRT"], ["定义", "公式", "适用"]),
        ("污泥龄", ["SRT"], ["定义", "原理", "适用"]),
        ("污泥容积指数", ["SVI"], ["定义", "适用", "局限"]),
        ("混合液悬浮固体", ["MLSS"], ["定义", "原理", "适用"]),
        ("A2/O", ["AAO", "A²/O"], ["原理", "适用", "局限"]),
        ("膜生物反应器", ["MBR"], ["原理", "适用", "局限"]),
    ]
    for term, aliases, facets in terms:
        rows.append(item(
            "terminology_explanation",
            f"请解释环境工程术语：{term}。",
            f"必须出现术语名称，并至少覆盖「{'、'.join(facets)}」中的两项；不得编造排放限值。",
            [
                C("content.explain_term", "content", term=term, aliases=aliases, facets=facets, min_facets=2),
                C("domain.no_fabricated_standard", "domain"),
            ],
            f"{term}（{'/'.join(aliases)}）的定义是该工艺或参数的基本含义。"
            f"原理上它用于描述处理系统的运行状态。适用时需结合水质与池容，局限性是不能单独替代排放达标判断。"
            f"未提供标准时需核实现行标准。",
        ))
    rows.append(item(
        "multi_step_reasoning",
        "北郊工业废水站拟从高负荷活性污泥改为 A2/O。请给出论证提纲。",
        "必须包含设计输入、关键假设、计算过程、推荐方案、风险点五个标题。",
        [C("content.must_include", "content", terms=["设计输入", "关键假设", "计算过程", "推荐方案", "风险点"])],
        "设计输入：水量与进水 COD/TN/TP。关键假设：未给标准时不编造限值。计算过程：先核算负荷与 HRT。"
        "推荐方案：A2/O 利于脱氮除磷。风险点：碳源不足与污泥膨胀。需核实现行标准。",
    ))
    rows.append(item(
        "pollution_control",
        "请分别说明脱水污泥的分类、暂存、运输、处理处置和风险控制。",
        "五个字段都必须出现。",
        [C("content.must_include", "content", terms=["分类", "暂存", "运输", "处理处置", "风险控制"])],
        "分类：按含水率和是否危废判定。暂存：防渗防雨。运输：密闭车辆。处理处置：消化或干化焚烧。"
        "风险控制：渗滤液与恶臭。",
    ))
    rows.append(item(
        "pollution_control",
        "鼓风机房邻近居民区，请提出噪声治理措施。",
        "必须从声源、传播路径和受声点三个层面作答。",
        [C("content.must_include", "content", terms=["声源", "传播路径", "受声点"])],
        "声源：减振与设备选型。传播路径：隔声罩与屏障。受声点：敏感建筑隔声窗。",
    ))
    return rows


def domain_items() -> list:
    rows = []
    rows.append(item(
        "environmental_monitoring",
        "未提供排放标准名称、适用对象和年份。请判断出水 COD=40 mg/L 是否达标。",
        "不得编造 GB/HJ 限值，也不得使用“已经达标”等确定性表述。",
        [
            C("domain.no_fabricated_standard", "domain"),
            C("domain.no_false_certainty", "domain"),
        ],
        "未提供标准依据，需核实现行标准，不能判定是否超标。",
    ))
    rows.append(item(
        "environmental_calculation",
        "请汇报某监测井 pH=7.4。",
        "pH 必须按无量纲报出，不得添加 mg/L。",
        [C("domain.ph_dimensionless", "domain")],
        "该监测井 pH 为 7.4（无量纲），不得写作 mg/L。建议按质控要求复测。",
    ))
    rows.append(item(
        "pollution_control",
        "喷漆车间风量 12000 m³/h，VOCs 约 380 mg/m³。请比较活性炭吸附与催化燃烧。",
        "浓度单位只能用 mg/m³；必须评估二次污染与运行安全；按优先、备选、不推荐输出。",
        [
            C("domain.allowed_units", "domain", must_use=["mg/m³"]),
            C("domain.safety", "domain", terms=["安全", "二次污染", "风险"], min_hits=2),
            C("content.must_include", "content", terms=["优先", "备选", "不推荐"]),
        ],
        "进口浓度 380 mg/m³。优先：催化燃烧（连续高浓度）。备选：活性炭吸附（波动负荷）。"
        "不推荐：无预处理直接高浓度吸附。风险包括爆炸下限与废活性炭二次污染，需核实现行标准。",
    ))
    rows.append(item(
        "air_pollution",
        "临海热电厂含硫烟气拟做脱硫改造。请比较石灰石-石膏湿法与半干法。",
        "必须按优先、备选、不推荐输出；必须涉及脱硫或石膏；必须评估二次污染与运行安全；不得编造排放限值。",
        [
            C("content.must_include", "content", terms=["优先", "备选", "不推荐"]),
            C("domain.process_match", "domain", keywords=["脱硫", "石灰石", "石膏", "湿法"]),
            C("domain.safety", "domain", terms=["安全", "二次污染", "石膏"], min_hits=2),
            C("domain.no_fabricated_standard", "domain"),
        ],
        "优先：石灰石-石膏湿法脱硫。备选：半干法脱硫。不推荐：仅靠高烟囱稀释。"
        "二次污染与运行安全：石膏副产物与浆液泄漏。未提供排放标准，需核实现行标准。",
    ))
    rows.append(item(
        "air_pollution",
        "北港烧结车间含氮烟气拟做脱硝。请比较 SCR 与 SNCR。",
        "必须按优先、备选、不推荐输出；必须涉及脱硝或氨逃逸；不得编造排放限值。",
        [
            C("content.must_include", "content", terms=["优先", "备选", "不推荐"]),
            C("domain.process_match", "domain", keywords=["脱硝", "SCR", "SNCR", "氨逃逸"]),
            C("domain.safety", "domain", terms=["安全", "氨逃逸", "催化剂", "二次污染"], min_hits=2),
            C("domain.no_fabricated_standard", "domain"),
        ],
        "优先：SCR脱硝。备选：SNCR。不推荐：无温度窗口盲目喷氨。"
        "二次污染与运行安全：氨逃逸与废催化剂。未提供排放标准，需核实现行标准。",
    ))
    rows.append(item(
        "air_pollution",
        "望海热电SCR床层压差快速上升，出口NOx回升。请诊断。",
        "必须列出恰好三个可能原因，每个原因包含证据和复核方法。",
        [C("format.n_causes", "format", n=3)],
        "原因1：催化剂堵塞。证据：压差升高。复核方法：核对压差曲线与运行小时。\n"
        "原因2：氨逃逸或喷氨不均。证据：出口NOx与嗅辨异常。复核方法：核对喷氨量。\n"
        "原因3：烟气量或进口NOx冲击。证据：负荷波动。复核方法：核对Qg与进口浓度。",
    ))
    rows.append(item(
        "wastewater_treatment",
        "进水氨氮偏高，请给出生物处理要点。",
        "必须涉及硝化或反硝化，并说明溶解氧控制，不得编造排放限值。",
        [
            C("domain.process_match", "domain", keywords=["硝化", "反硝化"]),
            C("content.must_include", "content", terms=["溶解氧"]),
            C("domain.no_fabricated_standard", "domain"),
        ],
        "氨氮去除依赖硝化，TN 还需要反硝化。好氧区保持适宜溶解氧，缺氧区控制碳氮比。未提供标准，需核实现行标准。",
    ))
    rows.append(item(
        "safety_constraint",
        "PAC 化学除磷的操作注意事项。",
        "必须给出投加量单位，并至少两项安全注意事项。",
        [
            C("domain.safety", "domain", terms=["安全", "防护", "腐蚀", "刺激"], min_hits=2),
            C("content.must_include", "content", terms=["投加"]),
        ],
        "投加量以 mg/L 或 kg/d 计。安全注意事项：防止药剂腐蚀皮肤；配制时佩戴防护眼镜；避免与碱液混放。",
    ))
    rows.append(item(
        "environmental_monitoring",
        "请说明地表水监测点布设。",
        "至少列出三个点位名称及每个点位的设置依据。",
        [C("content.must_include", "content", terms=["对照", "控制", "削减"])],
        "对照断面设于排污口上游，依据是背景。控制断面靠近排污影响区。削减断面设于下游充分混合后，依据是衰减。",
    ))
    return rows


def multi_items() -> list:
    rows = []
    q, cin, cout = 17000, 310, 45
    rate = round(removal_rate(cin, cout), 2)
    load = round(pollution_load_kg_d(q, cin - cout), 2)
    rows.append(item(
        "multi_step_reasoning",
        f"高新区再生水厂规模 {q} m³/d，进水 COD {cin} mg/L，出水 {cout} mg/L。请完成计算并给出工艺建议。",
        "必须：写出去除率与 kg/d 负荷（两位小数、有公式）；用→给出至少四单元流程；未给标准不得编造限值或宣称达标。",
        [
            C("numerical.value", "numerical", value=rate, unit="%"),
            C("numerical.value", "numerical", value=load, unit="kg/d"),
            C("numerical.formula", "numerical"),
            C("format.arrow_process", "format", min_units=4),
            C("domain.no_fabricated_standard", "domain"),
            C("domain.no_false_certainty", "domain"),
        ],
        f"去除率=({cin}-{cout})/{cin}×100%={rate:.2f}%。负荷={q}×({cin}-{cout})×0.001={load:.2f} kg/d。"
        f"流程：格栅→沉砂池→A2/O→二沉池→消毒。未提供排放标准，需核实现行标准。",
        gold={"params": {"Q": q, "Cin": cin, "Cout": cout}, "removal_rate_pct": rate, "load_kg_d": load},
    ))
    rows.append(item(
        "wastewater_treatment",
        "请比较 MBR 与常规活性污泥用于尾水回用。",
        "必须用表格（含处理效果、占地、能耗、污泥产量、适用条件），并单列二次污染与运行安全。",
        [
            C("format.table", "format"),
            C("format.headers", "format", headers=["处理效果", "占地", "能耗", "污泥产量", "适用条件"]),
            C("domain.safety", "domain", min_hits=2),
        ],
        "| 项目 | 处理效果 | 占地 | 能耗 | 污泥产量 | 适用条件 |\n| --- | --- | --- | --- | --- | --- |\n"
        "| MBR | 出水更清 | 小 | 高 | 较低 | 回用、用地紧 |\n| 活性污泥 | 常规 | 中 | 中 | 中 | 普通二级 |\n"
        "二次污染与运行安全：关注膜清洗药剂和剩余污泥。",
    ))
    rows.append(item(
        "environmental_monitoring",
        "请根据对象为滨湖污水厂出水、采样时间 2025-03-12 09:00、COD=55 mg/L、NH3-N=8.2 mg/L 编写摘要。",
        "摘要必须包含监测对象、采样时间、监测结果、异常项和结论；未给标准不得编造限值。",
        [
            C("format.headers", "format", headers=["监测对象", "采样时间", "监测结果", "异常项", "结论"]),
            C("domain.no_fabricated_standard", "domain"),
            C("domain.no_false_certainty", "domain"),
        ],
        "监测对象：滨湖污水厂出水。采样时间：2025-03-12 09:00。监测结果：COD 55 mg/L，NH3-N 8.2 mg/L。"
        "异常项：相对该厂历史均值氨氮偏高，需复核。结论：未提供排放标准，需核实现行标准，不能判定达标。",
    ))
    rows.append(item(
        "safety_constraint",
        "RTO 处理印刷废气，进口 VOCs 420 mg/m³。请给出方案排序并说明安全约束。",
        "按优先、备选、不推荐输出；浓度单位 mg/m³；必须讨论爆炸下限或火灾风险；不得编造排放限值。",
        [
            C("content.must_include", "content", terms=["优先", "备选", "不推荐"]),
            C("domain.allowed_units", "domain", must_use=["mg/m³"]),
            C("domain.safety", "domain", terms=["爆炸", "火灾", "安全", "风险"], min_hits=2),
            C("domain.no_fabricated_standard", "domain"),
        ],
        "进口 420 mg/m³。优先：RTO（连续高浓度）。备选：催化燃烧。不推荐：仅用小容量炭罐。"
        "需评估爆炸下限与火灾风险。未提供排放标准，需核实现行标准。",
    ))
    rows.append(item(
        "multi_step_reasoning",
        "数据不足：只有 COD=300 mg/L，无水量、无标准。请给出下一步。",
        "必须以信息不足开头；列出缺失的水量与标准；不得说已经达标。",
        [
            C("format.insufficient_prefix", "format"),
            C("content.missing_params", "content", params=["水量"]),
            C("domain.no_false_certainty", "domain"),
        ],
        "信息不足。缺少水量、采样时间和排放标准名称/年份，不能给出确定结论，需核实现行标准。",
    ))
    return rows


def human_items() -> list:
    """无自动 verifier：开放论证、文笔与专业深度。"""
    notes = "无自动 verifier，评测语义完整性与专业性。"
    return [
        item(
            "wastewater_treatment",
            "请评价将氧化沟改为 MBR 对一座老旧县城污水厂的可行性，结合占地、电耗和运维能力。",
            "要求论证充分，给出适用边界，而不是只给结论。",
            [],
            "（人工参考）应讨论土建约束、电耗、膜更换与人员技能，避免一刀切推荐。",
            auto=False, note=notes,
        ),
        item(
            "environmental_monitoring",
            "河流断面氨氮短期升高，请设计一周加密监测方案。",
            "方案需可执行，说明点位、频次和质控，但本条不提供程序打分。",
            [],
            "（人工参考）应含上下游对照、昼夜频次、平行样与空白样。",
            auto=False, note=notes,
        ),
        item(
            "pollution_control",
            "化工园恶臭投诉，现有生物滤池效果不稳。请提出分阶段治理路径。",
            "需考虑气象、源强波动和居民敏感点，本条人工评测。",
            [],
            "（人工参考）源强核查→工艺诊断→应急除臭→长期改造。",
            auto=False, note=notes,
        ),
        item(
            "terminology_explanation",
            "向非专业管理人员解释“总量控制”和“浓度控制”的差别。",
            "要求通俗且不出现事实性错误，本条人工评测表述质量。",
            [],
            "（人工参考）浓度是瞬时/断面水平，总量是许可排放质量。",
            auto=False, note=notes,
        ),
    ]


def extra_items() -> list:
    """补覆盖：更多诊断、监测、安全、污染控制，尽量保持单类型。"""
    rows = []
    events = [
        ("出水氨氮升高", "硝化变差"),
        ("二沉池大面积浮泥", "细碎气泡"),
        ("MBR 跨膜压差快速上升", "膜污染"),
        ("低温导致 COD 去除下降", "水温偏低"),
        ("反硝化碳源不足", "出水 TN 升高"),
    ]
    for i, (ev, clue) in enumerate(events):
        plant = PLANTS[i % len(PLANTS)]
        rows.append(item(
            "wastewater_treatment",
            f"{plant}出现{ev}，现场记录：{clue}。请诊断。",
            "必须列出恰好三个可能原因，每个原因包含证据和复核方法。",
            [C("format.n_causes", "format", n=3)],
            "原因1：负荷或供氧异常。证据：与现场记录一致。复核方法：核算曝气与污泥负荷。\n"
            "原因2：污泥性状变化。证据：沉降或压差异常。复核方法：SVI/镜检或跨膜压差曲线。\n"
            "原因3：进水冲击或碳氮比失调。证据：水质波动。复核方法：加密进出水监测。",
        ))
    headers_sets = [
        (["作用对象", "基本原理", "预期效果", "局限性"], "请说明化学除磷药剂的作用机制。"),
        (["分类", "暂存", "运输", "处理处置", "风险控制"], "请说明废活性炭的全过程管理。"),
        (["监测对象", "采样时间", "监测结果", "异常项", "结论"], "请把一次厂界恶臭巡测写成摘要（无具体数字也可）。"),
    ]
    for hs, q in headers_sets:
        body = "。".join(f"{h}：按题意说明对应内容" for h in hs) + "。"
        rows.append(item(
            "format_constraint",
            q,
            "回答必须包含下列标题：" + "、".join(hs) + "。",
            [C("format.headers", "format", headers=hs)],
            body,
        ))
    rows.append(item(
        "pollution_control",
        "储罐区低浓度 VOCs，风量较大。请排序治理技术。",
        "必须按优先、备选、不推荐三级输出。",
        [C("content.must_include", "content", terms=["优先", "备选", "不推荐"])],
        "优先：转轮浓缩+氧化。备选：吸附。不推荐：无组织直排。",
    ))
    rows.append(item(
        "safety_constraint",
        "次氯酸钠消毒车间的日常管理要点。",
        "至少涉及安全和风险两方面。",
        [C("domain.safety", "domain", terms=["安全", "风险", "防护", "泄漏"], min_hits=2)],
        "储存避免暴晒与酸混放，泄漏时疏散并冲洗，操作佩戴防护，安全与风险必须写入巡检。",
    ))
    rows.append(item(
        "environmental_monitoring",
        "实验室平行样相对偏差 18%。请说明质控要点。",
        "必须提到平行样，并至少再涉及空白样或加标回收之一。",
        [C("content.must_include", "content", terms=["平行样"]), C("content.must_include", "content", terms=["空白样"])],
        "平行样偏差 18% 偏高，应复查空白样、平行样和加标回收，必要时复测。",
    ))
    rows.append(item(
        "wastewater_treatment",
        "请用箭头写出 UASB+好氧的至少四段流程。",
        "必须使用 → 连接至少四个单元。",
        [C("format.arrow_process", "format", min_units=4)],
        "格栅→调节池→UASB→好氧池→二沉池。",
    ))
    rows.append(item(
        "environmental_monitoring",
        "请判断能否根据单独一个 COD 数字认定超标。题目未给标准。",
        "不得编造限值，不得宣称确定超标。",
        [C("domain.no_fabricated_standard", "domain"), C("domain.no_false_certainty", "domain")],
        "仅有单一 COD 且未提供标准名称与年份，需核实现行标准，不能确定超标。",
    ))
    rows.append(item(
        "environmental_calculation",
        "请计算将 COD 从 440 mg/L 降至 55 mg/L 的去除率。",
        "保留两位小数，必须写出公式并做数量级检查。",
        [
            C("numerical.value", "numerical", value=round((440 - 55) / 440 * 100, 2), unit="%"),
            C("numerical.two_decimals", "numerical"),
            C("numerical.formula", "numerical"),
            C("numerical.magnitude", "numerical"),
        ],
        f"去除率=(440-55)/440×100%={round((440-55)/440*100, 2):.2f}%。数量级合理性检查：去除率介于 0～100%。",
    ))
    rows.append(item(
        "multi_step_reasoning",
        "空港预处理站水量 36000 m³/d，进水 TN 44 mg/L。请计算进水 TN 负荷，并说明硝化反硝化控制要点。",
        "负荷必须用 kg/d；必须同时提到硝化与溶解氧；不得编造排放限值。",
        [
            C("numerical.value", "numerical", value=round(36000 * 44 * 0.001, 2), unit="kg/d"),
            C("domain.process_match", "domain", keywords=["硝化"]),
            C("content.must_include", "content", terms=["溶解氧"]),
            C("domain.no_fabricated_standard", "domain"),
        ],
        f"负荷=36000×44×0.001={round(36000*44*0.001, 2):.2f} kg/d。氨氮去除靠硝化，好氧区控制溶解氧。未提供标准，需核实现行标准。",
    ))
    return rows


def assign_ids(rows: list) -> list:
    out = []
    for i, row in enumerate(rows, 1):
        row = dict(row)
        row["id"] = f"envifbench-{i:04d}"
        row["split"] = "test"
        out.append(row)
    return out


def main():
    blocked = load_blocked_queries()
    raw = numerical_items() + format_items() + content_items() + domain_items() + extra_items() + multi_items() + human_items()
    kept, skipped = [], 0
    seen = set()
    for row in raw:
        q = row["query"]
        if q in blocked or q in seen:
            skipped += 1
            continue
        seen.add(q)
        kept.append(row)
    kept = assign_ids(kept)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bench_path = OUT_DIR / "envif_bench.jsonl"
    with bench_path.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fam = Counter(r["constraint_family"] for r in kept)
    task = Counter(r["task"] for r in kept)
    auto_n = sum(1 for r in kept if r["auto_verifiable"])
    n_cons = sum(len(r["constraints"]) for r in kept)
    meta = {
        "name": "EnvIF-Bench",
        "title": "环境工程领域指令遵循评测集",
        "n": len(kept),
        "n_auto_verifiable": auto_n,
        "n_human_only": len(kept) - auto_n,
        "n_atomic_constraints": n_cons,
        "skipped_overlap": skipped,
        "by_family": dict(fam),
        "by_task": dict(task),
        "verifiers": sorted(VERIFIERS),
        "tasks": list(TASKS),
        "families": ["content", "format", "numerical", "domain", "multi"],
        "metrics": ["prompt_accuracy", "constraint_accuracy", "by_family", "by_task", "by_verifier"],
        "note": "自动 verifier 覆盖 format/numerical/domain/content 可执行约束；开放论证题标记 auto_verifiable=false。",
    }
    (OUT_DIR / "envif_bench_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"wrote {bench_path}")


if __name__ == "__main__":
    main()
