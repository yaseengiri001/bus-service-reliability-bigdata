"""
Build an editable Microsoft Word version of the report: docs/report.docx.

Produces a POPULATED (literal) Table of Contents and Table of Figures with real
page numbers, a centred/styled cover page, numbered "Figure N:" captions, and
all figures/tables. Two-pass: build with blank page numbers -> render with
LibreOffice -> read each heading/figure's page -> rebuild with real numbers.

Requires: pandoc, LibreOffice (soffice), pdftotext (poppler).
Run:  .venv/bin/python src/build_report_docx.py
"""
from __future__ import annotations

import html
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "docs" / "report.md"
OUT = ROOT / "docs" / "report.docx"
TMP = Path("/tmp/report_pandoc.md")
RENDER_DIR = Path("/tmp/docx_render")
SOFFICE = "/Applications/LibreOffice.app/Contents/MacOS/soffice"

PAGEBREAK = '\n```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n'
TABPOS = 9200


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def number_figures(md_lines):
    out, n, i = [], 0, 0
    figs = []
    img_re = re.compile(r'^!\[\]\((.+?)\)\s*$')
    cap_re = re.compile(r'^\*(.+?)\*\s*$')
    while i < len(md_lines):
        m = img_re.match(md_lines[i])
        j = i + 1
        while j < len(md_lines) and md_lines[j].strip() == "":
            j += 1
        if m and j < len(md_lines) and cap_re.match(md_lines[j]):
            n += 1
            cap = cap_re.match(md_lines[j]).group(1)
            figs.append((n, cap))
            out.append(f'![Figure {n}: {cap}]({m.group(1)}){{width=76%}}')
            out.append("")
            i = j + 1
            continue
        out.append(md_lines[i]); i += 1
    return "\n".join(out), figs


def parse_headings(md_text):
    hs = []
    for line in md_text.splitlines():
        if line.startswith("### "):
            hs.append((3, line[4:].strip()))
        elif line.startswith("## "):
            hs.append((2, line[3:].strip()))
    return hs


def cover_openxml():
    def p(text, size, color, bold=True, italic=False, before=0, after=60):
        b = "<w:b/>" if bold else ""
        it = "<w:i/>" if italic else ""
        return (f'<w:p><w:pPr><w:jc w:val="center"/>'
                f'<w:spacing w:before="{before}" w:after="{after}"/></w:pPr>'
                f'<w:r><w:rPr>{b}{it}<w:color w:val="{color}"/><w:sz w:val="{size}"/></w:rPr>'
                f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>')
    parts = [
        p("Softwarica", 60, "1F4E79", before=1400, after=0),
        p("COLLEGE OF IT & E-COMMERCE", 18, "2E75B6", after=200),
        p("in collaboration with", 20, "555555", bold=False, italic=True, after=0),
        p("Coventry University", 40, "0D3B66", after=200),
        p("Assignment Title:", 30, "6B1F1F", before=1400),
        p("Individual Coursework of Big Data Programming Project", 26, "222222", bold=False, after=180),
        p("Module Code:", 30, "6B1F1F"),
        p("ST5011CEM", 26, "222222", bold=False, after=180),
        p("Date of Submission:", 30, "6B1F1F"),
        p("[Date of Submission]", 26, "222222", bold=False),
        p("Submitted by:", 28, "6B1F1F", before=1200),
        p("[Student ID] — [Student Name]", 24, "222222", bold=False, after=180),
        p("Submitted to:", 28, "6B1F1F"),
        p("Siddhartha Neupane", 24, "222222", bold=False),
    ]
    return '\n```{=openxml}\n' + "".join(parts) + '\n```\n'


def _entry(text, page, bold=False, indent=0):
    b = "<w:b/>" if bold else ""
    ind = f'<w:ind w:left="{indent}"/>' if indent else ""
    return (f'<w:p><w:pPr>{ind}'
            f'<w:tabs><w:tab w:val="right" w:leader="dot" w:pos="{TABPOS}"/></w:tabs>'
            f'<w:spacing w:after="20"/><w:rPr>{b}</w:rPr></w:pPr>'
            f'<w:r><w:rPr>{b}</w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r>'
            f'<w:r><w:tab/></w:r><w:r><w:t xml:space="preserve">{page}</w:t></w:r></w:p>')


def toc_openxml(headings, hpage):
    parts = ['<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
             '<w:r><w:t>Table of Contents</w:t></w:r></w:p>']
    for lvl, text in headings:
        parts.append(_entry(text, hpage.get(text, ""), bold=(lvl == 2),
                            indent=0 if lvl == 2 else 400))
    return '\n```{=openxml}\n' + "".join(parts) + '\n```\n'


def lof_openxml(figs, fpage):
    parts = ['<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
             '<w:r><w:t>Table of Figures</w:t></w:r></w:p>']
    for n, cap in figs:
        parts.append(_entry(f"Figure {n}: {cap}", fpage.get(n, "")))
    return '\n```{=openxml}\n' + "".join(parts) + '\n```\n'


def assemble(body, headings, figs, hpage, fpage):
    doc = (cover_openxml() + PAGEBREAK
           + toc_openxml(headings, hpage) + PAGEBREAK
           + lof_openxml(figs, fpage) + PAGEBREAK + body)
    TMP.write_text(doc)
    subprocess.run(["pandoc", str(TMP), "-o", str(OUT),
                    "--resource-path", str(ROOT / "docs"),
                    "--from=markdown+raw_attribute"], check=True)


def render_pdf():
    RENDER_DIR.mkdir(exist_ok=True)
    for f in RENDER_DIR.glob("*.pdf"):
        f.unlink()
    proc = subprocess.Popen(
        [SOFFICE, "--headless", "-env:UserInstallation=file:///tmp/lo_profile",
         "--convert-to", "pdf", "--outdir", str(RENDER_DIR), str(OUT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        proc.wait(timeout=180)
    except subprocess.TimeoutExpired:
        proc.kill()
    return RENDER_DIR / "report.pdf"


def extract_pages(pdf, headings, figs):
    """Exact-line match, searching only from the Table-of-Figures page onward
    (i.e. the body), so headings that are substrings of other headings — e.g.
    'Methodology' inside 'Validation Methodology' — resolve to the right page."""
    txt = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"]).decode("utf-8", "ignore")
    pages = txt.split("\f")
    # body begins at the first EXACT "Executive Summary" heading line — this is
    # past the cover, Table of Contents and Table of Figures (whose entries wrap
    # and would otherwise false-match on the ToF page itself).
    body_start = 0
    for i, pg in enumerate(pages):
        if any(ln.strip() == "Executive Summary" for ln in pg.splitlines()):
            body_start = i
            break

    def hpage(h):
        for i in range(body_start, len(pages)):
            if any(ln.strip() == h for ln in pages[i].splitlines()):
                return str(i + 1)
        return ""

    def fpage(n):
        pat = f"Figure {n}:"
        for i in range(body_start, len(pages)):
            for ln in pages[i].splitlines():
                s = ln.strip()
                if s.startswith(pat) and not re.search(r"\.{4,}", s):
                    return str(i + 1)
        return ""
    return {text: hpage(text) for _, text in headings}, {n: fpage(n) for n, _ in figs}


def main():
    body, figs = number_figures(MD.read_text().splitlines())
    headings = parse_headings(MD.read_text())

    # pass 1 — blank page numbers (fixes layout size)
    assemble(body, headings, figs, {}, {})
    print("[docx] pass 1 built; rendering to read page numbers ...")
    pdf = render_pdf()
    hpage, fpage = extract_pages(pdf, headings, figs)
    found = sum(1 for v in {**hpage, **{str(k): v for k, v in fpage.items()}}.values() if v)
    print(f"[docx] resolved page numbers for {found} entries")

    # pass 2 — real page numbers baked in
    assemble(body, headings, figs, hpage, fpage)
    print(f"[docx] wrote {OUT.name} ({OUT.stat().st_size/1024:.0f} KB, {len(figs)} figures, "
          f"{len(headings)} TOC entries) — populated Contents + Figures.")


if __name__ == "__main__":
    main()
