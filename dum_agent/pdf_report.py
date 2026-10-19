"""Minimal Markdown -> PDF converter (headings, bullets, numbered lists, tables, bold/italic)."""
import re
from pathlib import Path
from typing import List
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

_REPLACEMENTS = {
    "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2022": "-", "\u2192": "->", "\u2026": "...", "\u00a0": " ", "\u2713": "ok", "\u2705": "[x]",
}


def _latin1(text: str) -> str:
    for k, v in _REPLACEMENTS.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")  # built-in PDF fonts are Latin-1


def _inline(text: str) -> str:
    text = escape(_latin1(text))
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)
    return text


def _para(text: str, style: ParagraphStyle, **kw) -> Paragraph:
    try:
        return Paragraph(_inline(text), style, **kw)
    except Exception:  # malformed inline markup -> plain text
        return Paragraph(escape(_latin1(text)), style, **kw)


def markdown_to_pdf(md: str, path, title: str = "Report") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, leading=14)
    cell = ParagraphStyle("cell", parent=body, fontSize=9, leading=12)
    cell_head = ParagraphStyle("cellh", parent=cell, fontName="Helvetica-Bold")
    doc = SimpleDocTemplate(
        str(path), pagesize=A4, title=title,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
    )
    avail = A4[0] - 40 * mm

    story: List = []
    para_buf: List[str] = []

    def flush():
        if para_buf:
            story.append(_para(" ".join(para_buf), body))
            story.append(Spacer(1, 4))
            para_buf.clear()

    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            flush(); i += 1; continue

        if stripped.startswith("|"):  # table
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                n = max(len(r) for r in rows)
                data = [
                    [_para(c, cell_head if r == 0 else cell) for c in row + [""] * (n - len(row))]
                    for r, row in enumerate(rows)
                ]
                t = Table(data, colWidths=[avail / n] * n, repeatRows=1, hAlign="LEFT")
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]))
                story.extend([t, Spacer(1, 8)])
            continue

        m = re.match(r"^(#{1,4})\s+(.*)", stripped)
        if m:
            flush()
            level = min(len(m.group(1)), 3)
            story.append(_para(m.group(2), styles[f"Heading{level}"]))
            i += 1; continue

        if re.fullmatch(r"-{3,}|\*{3,}", stripped):
            flush(); story.append(HRFlowable(width="100%", color=colors.lightgrey)); i += 1; continue

        b = re.match(r"^(\s*)[-*+]\s+(.*)", line)
        n = re.match(r"^(\s*)(\d+)[.)]\s+(.*)", line)
        if b or n:
            flush()
            indent = len((b or n).group(1)) // 2
            style = ParagraphStyle("li", parent=body, leftIndent=14 * (indent + 1), bulletIndent=14 * indent)
            if b:
                story.append(_para(b.group(2), style, bulletText="\u2022"))
            else:
                story.append(_para(n.group(3), style, bulletText=f"{n.group(2)}."))
            i += 1; continue

        para_buf.append(stripped)
        i += 1

    flush()
    doc.build(story)
    return path
