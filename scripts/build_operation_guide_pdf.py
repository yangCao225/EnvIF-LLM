#!/usr/bin/env python3
"""将 docs/AutoIF环境工程操作指南.md 转为中文 PDF。"""
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "docs" / "AutoIF环境工程操作指南.md"
OUT = ROOT / "AutoIF环境工程操作指南.pdf"

SONG_CANDIDATES = [
    ("C:/Windows/Fonts/simsun.ttc", 0),
    ("C:/Windows/Fonts/STSONG.TTF", None),
    ("C:/Windows/Fonts/simsun.ttf", None),
]
TIMES_REG = "C:/Windows/Fonts/times.ttf"
TIMES_BOLD = "C:/Windows/Fonts/timesbd.ttf"
TIMES_ITALIC = "C:/Windows/Fonts/timesi.ttf"
TIMES_BI = "C:/Windows/Fonts/timesbi.ttf"


def _register_song():
    last_err = None
    for path, idx in SONG_CANDIDATES:
        if not Path(path).exists():
            continue
        try:
            if idx is None:
                pdfmetrics.registerFont(TTFont("Song", path))
            else:
                pdfmetrics.registerFont(TTFont("Song", path, subfontIndex=idx))
            return
        except Exception as e:
            last_err = e
    raise RuntimeError(f"未找到宋体 SimSun/华文宋体: {last_err}")


def register_fonts():
    _register_song()
    pdfmetrics.registerFont(TTFont("TimesNR", TIMES_REG))
    if Path(TIMES_BOLD).exists():
        pdfmetrics.registerFont(TTFont("TimesNR-Bold", TIMES_BOLD))
    else:
        pdfmetrics.registerFont(TTFont("TimesNR-Bold", TIMES_REG))
    if Path(TIMES_ITALIC).exists():
        pdfmetrics.registerFont(TTFont("TimesNR-Italic", TIMES_ITALIC))
    else:
        pdfmetrics.registerFont(TTFont("TimesNR-Italic", TIMES_REG))
    if Path(TIMES_BI).exists():
        pdfmetrics.registerFont(TTFont("TimesNR-BoldItalic", TIMES_BI))
    else:
        pdfmetrics.registerFont(TTFont("TimesNR-BoldItalic", TIMES_REG))
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    registerFontFamily(
        "TimesNR",
        normal="TimesNR",
        bold="TimesNR-Bold",
        italic="TimesNR-Italic",
        boldItalic="TimesNR-BoldItalic",
    )
    registerFontFamily("Song", normal="Song", bold="Song", italic="Song", boldItalic="Song")


def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def mix_fonts(escaped: str) -> str:
    """正文默认宋体；连续西文/数字用 Times New Roman。"""
    parts = re.split(r"(<[^>]+>)", escaped)
    out = []
    for part in parts:
        if part.startswith("<"):
            out.append(part)
            continue
        out.append(re.sub(
            r"[\x21-\x7E]+",
            lambda m: f'<font name="TimesNR">{m.group(0)}</font>',
            part,
        ))
    return "".join(out)


def inline(text: str) -> str:
    text = escape(text)
    # 行内代码只缩小字号，中文仍走宋体；西文由 mix_fonts 套 Times
    text = re.sub(r"`([^`]+)`", lambda m: f'<font size="8">{m.group(1)}</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return mix_fonts(text)


def mixed_width(text: str, size: float) -> float:
    total = 0.0
    for ch in text:
        font = "TimesNR" if ord(ch) < 128 else "Song"
        total += pdfmetrics.stringWidth(ch, font, size)
    return total


def draw_mixed(canvas, x, y, text, size=8, center=False):
    if center:
        x = x - mixed_width(text, size) / 2
    for ch in text:
        font = "TimesNR" if ord(ch) < 128 else "Song"
        canvas.setFont(font, size)
        canvas.drawString(x, y, ch)
        x += pdfmetrics.stringWidth(ch, font, size)


def styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(
        "CoverTitle", fontName="Song", fontSize=18, leading=28, alignment=TA_CENTER,
        textColor=colors.HexColor("#1e3a5f"), spaceAfter=12,
    ))
    ss.add(ParagraphStyle(
        "CoverSub", fontName="Song", fontSize=11, leading=18, alignment=TA_CENTER,
        textColor=colors.HexColor("#334155"), spaceAfter=6,
    ))
    ss.add(ParagraphStyle(
        "H1", fontName="Song", fontSize=14, leading=22, textColor=colors.HexColor("#1e3a5f"),
        spaceBefore=14, spaceAfter=8, borderPadding=3,
    ))
    ss.add(ParagraphStyle(
        "H2", fontName="Song", fontSize=12, leading=18, textColor=colors.HexColor("#1d4ed8"),
        spaceBefore=10, spaceAfter=6,
    ))
    ss.add(ParagraphStyle(
        "H3", fontName="Song", fontSize=10.5, leading=16, textColor=colors.HexColor("#0f172a"),
        spaceBefore=8, spaceAfter=4,
    ))
    ss.add(ParagraphStyle(
        "Body", fontName="Song", fontSize=9.5, leading=16, alignment=TA_JUSTIFY, spaceAfter=6,
    ))
    ss.add(ParagraphStyle(
        "BulletCN", fontName="Song", fontSize=9.5, leading=15, leftIndent=14, spaceAfter=2,
    ))
    ss.add(ParagraphStyle(
        "CodeBlock", fontName="Song", fontSize=7.4, leading=10.5,
        backColor=colors.HexColor("#0f172a"), textColor=colors.HexColor("#e2e8f0"),
        leftIndent=4, rightIndent=4, spaceBefore=4, spaceAfter=8,
    ))
    ss.add(ParagraphStyle(
        "Th", fontName="Song", fontSize=8, leading=12, textColor=colors.white, alignment=TA_CENTER,
    ))
    ss.add(ParagraphStyle(
        "Td", fontName="Song", fontSize=8, leading=12,
    ))
    ss.add(ParagraphStyle(
        "Footer", fontName="Song", fontSize=8, textColor=colors.HexColor("#64748b"), alignment=TA_CENTER,
    ))
    ss.add(ParagraphStyle(
        "Caption", fontName="Song", fontSize=9, leading=14, textColor=colors.HexColor("#475569"),
        alignment=TA_CENTER, spaceAfter=10,
    ))
    return ss


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#1e3a5f"))
    canvas.rect(0, A4[1] - 12 * mm, A4[0], 12 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    draw_mixed(canvas, 18 * mm, A4[1] - 8 * mm, "AutoIF 环境工程指令遵循优化系统 · 操作指南", size=8)
    canvas.setFillColor(colors.HexColor("#e2e8f0"))
    canvas.rect(0, 0, A4[0], 12 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#334155"))
    draw_mixed(
        canvas, A4[0] / 2, 5 * mm,
        f"第 {doc.page} 页  |  实验数字只引用 output/experiment_summary.json",
        size=8, center=True,
    )
    canvas.restoreState()


def parse_table(rows: list[str], st) -> Table:
    data = []
    for i, row in enumerate(rows):
        cols = [c.strip() for c in row.strip().strip("|").split("|")]
        if i == 1 and all(re.match(r"^:?-+:?$", c.replace(" ", "")) for c in cols):
            continue
        style_name = "Th" if i == 0 else "Td"
        data.append([Paragraph(inline(c), st[style_name]) for c in cols])
    col_n = max(len(r) for r in data)
    for r in data:
        while len(r) < col_n:
            r.append(Paragraph("", st["Td"]))
    usable = A4[0] - 36 * mm
    widths = [usable / col_n] * col_n
    tbl = Table(data, colWidths=widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f1f5f9")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return tbl


def build():
    register_fonts()
    st = styles()
    text = MD.read_text(encoding="utf-8")
    lines = text.splitlines()
    story = []

    # cover
    story.append(Spacer(1, 38 * mm))
    story.append(Paragraph(inline("面向污水处理与环境监测的"), st["CoverSub"]))
    story.append(Paragraph(inline("AutoIF 环境工程大模型"), st["CoverTitle"]))
    story.append(Paragraph(inline("指令遵循优化系统"), st["CoverTitle"]))
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(inline("完 整 操 作 指 南"), st["CoverSub"]))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(inline("从 AutoDL 租机、环境安装、一键训练，到评测、排错与关机"), st["Caption"]))
    story.append(Paragraph(inline("主线：污水处理　　第二场景：环境监测　　扩展：废气 / 固废 / 噪声 / 环评"), st["Caption"]))
    story.append(Paragraph(inline("请勿沿用旧材料中互相矛盾的 37/406 条 SFT 或两套 DPO 学习率"), st["Caption"]))
    story.append(PageBreak())

    i = 0
    n = len(lines)
    first_h1 = True
    while i < n:
        line = lines[i]
        raw = line.rstrip()
        if not raw:
            i += 1
            continue
        if raw.startswith("```"):
            buf = []
            i += 1
            while i < n and not lines[i].startswith("```"):
                buf.append(lines[i].replace("\t", "    "))
                i += 1
            i += 1
            # Preformatted does not wrap well; keep as Paragraph with <br/>
            safe = mix_fonts(escape("\n".join(buf) if buf else " ").replace("\n", "<br/>"))
            story.append(Paragraph(safe or "&nbsp;", st["CodeBlock"]))
            continue
        if raw.startswith("|") and i + 1 < n and lines[i + 1].strip().startswith("|"):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            story.append(parse_table(rows, st))
            story.append(Spacer(1, 4 * mm))
            continue
        if raw.startswith("# "):
            if not first_h1:
                story.append(PageBreak())
            first_h1 = False
            title = raw[2:].strip()
            if title.startswith("面向污水处理"):
                i += 1
                continue
            story.append(Paragraph(inline(title), st["H1"]))
            i += 1
            continue
        if raw.startswith("## "):
            story.append(Paragraph(inline(raw[3:].strip()), st["H1"]))
            i += 1
            continue
        if raw.startswith("### "):
            story.append(Paragraph(inline(raw[4:].strip()), st["H2"]))
            i += 1
            continue
        if raw.startswith("---"):
            i += 1
            continue
        if raw.startswith("- "):
            story.append(Paragraph("• " + inline(raw[2:].strip()), st["BulletCN"]))
            i += 1
            continue
        if re.match(r"^\d+\.\s", raw):
            story.append(Paragraph(inline(raw), st["BulletCN"]))
            i += 1
            continue
        if raw.startswith("> "):
            story.append(Paragraph(inline(raw[2:].strip()), st["Caption"]))
            i += 1
            continue
        story.append(Paragraph(inline(raw), st["Body"]))
        i += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title="AutoIF 环境工程操作指南",
        author="AutoIF-EnvEng",
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"PDF: {OUT}  pages~{doc.page}")


if __name__ == "__main__":
    build()
