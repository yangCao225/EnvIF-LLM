#!/usr/bin/env python3
"""按用户提供的竖版学术流程图风格绘制 EnvIF 技术路线，并全屏演示。"""
from __future__ import annotations

import json
from pathlib import Path

import win32com.client

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "EnvIF技术路线_竖版.pptx"
SUMMARY = ROOT / "output" / "experiment_summary.json"

MSO_RECT = 1
MSO_ROUND = 5
MSO_DOWN = 36
PP_CENTER = 2
PP_LEFT = 1
MSO_MID = 3
PP_BLANK = 12
MSO_F = 0
MSO_T = -1
MSO_DASH = 4
MSO_SOLID = 1
PP_SHOW = 1

W, H = 720, 1280


def rgb(r, g, b):
    return int(r) + (int(g) << 8) + (int(b) << 16)


WHITE = rgb(255, 255, 255)
INK = rgb(45, 52, 54)
NAVY = rgb(47, 64, 80)
CREAM = rgb(255, 249, 225)
CREAM_LINE = rgb(196, 176, 132)
BLUE = rgb(214, 234, 248)
BLUE_H = rgb(174, 214, 241)
PINK = rgb(253, 236, 234)
ORANGE_LINE = rgb(192, 132, 72)
MUTED = rgb(92, 96, 100)
ARROW = rgb(70, 78, 86)


def load_counts():
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    c = data.get("counts") or {}
    return {
        "seed": c.get("seed_instructions", 52),
        "train": c.get("domain_queries", 1020),
        "test": c.get("heldout_test", 389),
        "sft": c.get("sft", 1020),
        "dpo": c.get("dpo_pairs", 2667),
        "bench": c.get("envif_bench", 79),
        "rft": c.get("query_rft", 0),
        "when": data.get("generated_at", ""),
    }


def fill_line(sh, fill, line, weight=1.15, dash=False):
    sh.Fill.Visible = MSO_T
    sh.Fill.Solid()
    sh.Fill.ForeColor.RGB = fill
    sh.Line.Visible = MSO_T
    sh.Line.ForeColor.RGB = line
    sh.Line.Weight = weight
    sh.Line.DashStyle = MSO_DASH if dash else MSO_SOLID
    sh.Shadow.Visible = MSO_F


def txt(sh, text, size, color=INK, bold=False, align=PP_CENTER, m=5):
    tf = sh.TextFrame
    tf.WordWrap = MSO_T
    tf.MarginLeft = m
    tf.MarginRight = m
    tf.MarginTop = 3
    tf.MarginBottom = 3
    tf.AutoSize = 0
    try:
        tf.VerticalAnchor = MSO_MID
    except Exception:
        pass
    tr = tf.TextRange
    tr.Text = text
    tr.Font.Name = "微软雅黑"
    tr.Font.NameFarEast = "微软雅黑"
    tr.Font.Size = size
    tr.Font.Bold = MSO_T if bold else MSO_F
    tr.Font.Color.RGB = color
    tr.ParagraphFormat.Alignment = align
    tr.ParagraphFormat.SpaceAfter = 0
    tr.ParagraphFormat.SpaceBefore = 0


def shape(slide, kind, l, t, w, h, fill, line, text="", size=11, bold=False, weight=1.15, dash=False, color=INK, align=PP_CENTER):
    sh = slide.Shapes.AddShape(kind, l, t, w, h)
    fill_line(sh, fill, line, weight, dash)
    if text:
        txt(sh, text, size, color, bold, align)
    return sh


def round_box(slide, l, t, w, h, fill, line, text="", size=11, bold=False, weight=1.15, dash=False, color=INK):
    return shape(slide, MSO_ROUND, l, t, w, h, fill, line, text, size, bold, weight, dash, color)


def rect(slide, l, t, w, h, fill, line, text="", size=11, bold=False, weight=1.0, dash=False, color=INK, align=PP_CENTER):
    return shape(slide, MSO_RECT, l, t, w, h, fill, line, text, size, bold, weight, dash, color)


def down(slide, x, y, w=11, h=14):
    sh = slide.Shapes.AddShape(MSO_DOWN, x, y, w, h)
    sh.Fill.Visible = MSO_T
    sh.Fill.Solid()
    sh.Fill.ForeColor.RGB = ARROW
    sh.Line.Visible = MSO_F
    sh.Shadow.Visible = MSO_F
    return sh


def lane_label(slide, l, t, w, h, text):
    return round_box(slide, l, t, w, h, NAVY, NAVY, text, 13, True, 0, False, WHITE)


def draw(slide, cnt):
    # page
    bg = slide.Shapes.AddShape(MSO_RECT, 0, 0, W, H)
    fill_line(bg, WHITE, WHITE, 0)
    bg.Line.Visible = MSO_F

    LX, LW = 14, 86
    CX, CW = 112, 590

    # ----- 1 研究问题 -----
    y = 18
    round_box(slide, CX, y, CW, 188, CREAM, CREAM_LINE, weight=1.35)
    lane_label(slide, LX, y + 52, LW, 84, "研究\n问题")
    round_box(
        slide, CX + 16, y + 12, CW - 32, 44,
        WHITE, rgb(80, 140, 180),
        "基于 AutoIF 的环境工程指令遵循优化（EnvIF-TLR）",
        13, True, 1.1, True, NAVY,
    )
    down(slide, CX + CW / 2 - 5, y + 58, 11, 12)
    round_box(slide, CX + 16, y + 72, CW - 32, 44, PINK, rgb(210, 170, 170),
              "任务：污水处理 · 环境监测 · 大气污染（含硫/含氮烟气）的结构化回答\n不是环境工程专家大模型", 11, False, 1.0)
    down(slide, CX + CW / 2 - 5, y + 118, 11, 12)
    round_box(slide, CX + 16, y + 132, CW - 32, 44, PINK, rgb(210, 170, 170),
              "方法增量：Typed Layered Reward（带单位计算 + 知识边界 + 分类型难负样本）", 11, False, 1.0)

    down(slide, CX + CW / 2 - 5, y + 188, 11, 14)

    # ----- 2 数据 -----
    y = 224
    round_box(slide, CX, y, CW, 292, CREAM, CREAM_LINE, weight=1.35)
    lane_label(slide, LX, y + 96, LW, 100, "数据收集\n与处理")
    # three pillars
    cols = [
        "污水处理\n负荷 / 去除率 / HRT",
        "环境监测\n报告格式 / 单位 / 信息不足",
        "大气污染\nSO₂ NOx 烟气 · 脱硫脱硝 · 基准氧",
    ]
    bw = 178
    for i, t in enumerate(cols):
        round_box(slide, CX + 16 + i * (bw + 10), y + 14, bw, 52, BLUE, rgb(140, 180, 200), t, 11, True)
    # sources
    src = [
        "种子指令  %s 条\n可程序验证约束" % cnt["seed"],
        "模板 + 厂区案例\ntrain %s / test %s" % (cnt["train"], cnt["test"]),
        "gold 公式与单位\n禁止套用污水公式到烟气",
    ]
    for i, t in enumerate(src):
        round_box(slide, CX + 16 + i * (bw + 10), y + 76, bw, 52, WHITE, rgb(150, 160, 170), t, 10, False)
    down(slide, CX + CW / 2 - 5, y + 130, 11, 12)
    round_box(
        slide, CX + 16, y + 146, CW - 32, 56, BLUE_H, rgb(120, 160, 190),
        "EnvIF-TLR 离线构造（Python 验证器）\n不是 7B 教师蒸馏，query_rft = %s" % cnt["rft"],
        12, True,
    )
    down(slide, CX + CW / 2 - 5, y + 204, 11, 12)
    round_box(
        slide, CX + 16, y + 220, CW - 32, 56, WHITE, rgb(150, 160, 170),
        "得到多主线联合语料：离线 SFT  %s 条    ·    离线 DPO  %s 对" % (cnt["sft"], cnt["dpo"]),
        12, True,
    )

    down(slide, CX + CW / 2 - 5, y + 292, 11, 14)

    # ----- 3 模型 -----
    y = 534
    round_box(slide, CX, y, CW, 486, CREAM, CREAM_LINE, weight=1.35)
    lane_label(slide, LX, y + 180, LW, 120, "模型构建\n与优化\nEnvIF-TLR")

    # group A dashed
    rect(slide, CX + 12, y + 10, CW - 24, 210, CREAM, ORANGE_LINE, dash=True, weight=1.35)
    round_box(slide, CX + 22, y + 18, 168, 26, BLUE_H, rgb(120, 160, 190), "输入 + 三层奖励", 11, True)
    round_box(
        slide, CX + 200, y + 18, 370, 26, WHITE, rgb(150, 160, 170),
        "基座 Qwen2.5-1.5B-Instruct  +  领域查询 / gold", 11, False,
    )
    mods = [
        "格式层 L1\n标题 · 单位 · 两位小数\n信息不足前缀",
        "计算层 L2（带单位）\n污水 Q×C×0.001\n烟气 Qg×C×24×10⁻⁶\n基准氧 (21−O₂,s)/(21−O₂,m)",
        "知识边界 L3\n不编造排放限值\n不虚假达标\n工艺匹配 湿法/SCR",
    ]
    mw = 178
    for i, t in enumerate(mods):
        round_box(slide, CX + 22 + i * (mw + 10), y + 52, mw, 100, BLUE, rgb(140, 180, 200), t, 10, False)
    down(slide, CX + CW / 2 - 5, y + 154, 11, 12)
    split_w = 270
    round_box(slide, CX + 22, y + 170, split_w, 40, WHITE, rgb(150, 160, 170), "Chosen  →  离线 SFT", 11, True)
    round_box(
        slide, CX + 22 + split_w + 16, y + 170, split_w, 40, WHITE, rgb(150, 160, 170),
        "Typed rejected  →  DPO\ncalc / unit / 编造标准 / 虚假确定性", 10, True,
    )

    # group B dashed 调参
    rect(slide, CX + 12, y + 228, CW - 24, 70, CREAM, ORANGE_LINE, dash=True, weight=1.35)
    round_box(slide, CX + 22, y + 236, 168, 26, BLUE_H, rgb(120, 160, 190), "学生模型 LoRA", 11, True)
    round_box(
        slide, CX + 200, y + 236, 370, 54, WHITE, rgb(150, 160, 170),
        "SFT v2：1020 条 · 3 epoch · max_len 768 · 不覆盖 v1\nDPO v1：400 对（未超过 SFT，不默认更强）", 11, False,
    )

    # group C dashed 评估
    rect(slide, CX + 12, y + 308, CW - 24, 166, CREAM, ORANGE_LINE, dash=True, weight=1.35)
    round_box(slide, CX + 22, y + 316, 168, 26, BLUE_H, rgb(120, 160, 190), "评测模块", 11, True)
    round_box(
        slide, CX + 22, y + 350, 178, 110, BLUE, rgb(140, 180, 200),
        "EnvIF-Bench\n%s 题 / 75 可自动打分\ncontent / format\nnumerical / domain / multi" % cnt["bench"],
        11, True,
    )
    # baseline grid
    bases = ["Base 1.5B", "SFT v1\n72 题 污水+监测", "SFT v2\n79 题 含大气", "DPO v1\nnumerical 下降", "Oracle\n打分器自洽 1.0", "7B 教师\n本机未跑"]
    for i, t in enumerate(bases):
        r, c = divmod(i, 3)
        round_box(slide, CX + 214 + c * 118, y + 350 + r * 56, 110, 50, WHITE, rgb(150, 160, 170), t, 9, False)

    down(slide, CX + CW / 2 - 5, y + 486, 11, 14)

    # ----- 4 结论 -----
    y = 1038
    round_box(slide, CX, y, CW, 176, CREAM, CREAM_LINE, weight=1.35)
    lane_label(slide, LX, y + 46, LW, 84, "研究结论\n与展望")
    round_box(
        slide, CX + 16, y + 16, CW - 32, 68, PINK, rgb(210, 170, 170),
        "结论：v2 已能排出湿法脱硫 / SCR 脱硝优先序；烟气负荷与基准氧公式仍常算错。\n"
        "1.5B 学会的是指令骨架，不是排放标准专家知识。",
        11, False,
    )
    down(slide, CX + CW / 2 - 5, y + 86, 11, 12)
    round_box(
        slide, CX + 16, y + 100, CW - 32, 60, BLUE, rgb(140, 180, 200),
        "展望：本机路径可复现（1.5B LoRA）。AutoDL 7B 教师九步需 40GB+ GPU，query_rft 保持 0。\n"
        "数字只引用 output/experiment_summary.json（%s）" % cnt["when"],
        11, False,
    )

    cap = slide.Shapes.AddShape(MSO_RECT, CX, 1222, CW, 42)
    cap.Fill.Visible = MSO_F
    cap.Line.Visible = MSO_F
    txt(
        cap,
        "图 1  EnvIF-TLR 技术路线（本机可复现主路径）。仓库 github.com/yangCao225/EnvIF-LLM",
        10, MUTED, False, PP_LEFT, 2,
    )


def main():
    cnt = load_counts()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ppt = win32com.client.Dispatch("PowerPoint.Application")
    ppt.Visible = True
    pres = ppt.Presentations.Add()
    pres.PageSetup.SlideWidth = W
    pres.PageSetup.SlideHeight = H
    if pres.Slides.Count == 0:
        slide = pres.Slides.Add(1, PP_BLANK)
    else:
        slide = pres.Slides(1)
        try:
            slide.Layout = PP_BLANK
        except Exception:
            pass
        for i in range(slide.Shapes.Count, 0, -1):
            slide.Shapes(i).Delete()
    draw(slide, cnt)
    while pres.Slides.Count > 1:
        pres.Slides(pres.Slides.Count).Delete()
    if OUT.exists():
        try:
            OUT.unlink()
        except Exception:
            pass
    pres.SaveAs(str(OUT))
    pres.Slides(1).Select()
    ss = pres.SlideShowSettings
    ss.ShowType = PP_SHOW
    ss.LoopUntilStopped = MSO_F
    ss.Run()
    print("slideshow", OUT)


if __name__ == "__main__":
    main()
