"""
Build the ST5011CEM-style report PDF with WeasyPrint:
cover page -> acknowledgement -> Table of Contents -> Table of Figures -> body,
with auto-numbered figures, real page numbers, and page borders.

Run:  .venv/bin/python src/build_report_pdf.py
"""
from __future__ import annotations

import os
os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
    "/opt/homebrew/lib:" + os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", ""))

import re
from pathlib import Path

import markdown
import weasyprint

DOCS = Path(__file__).resolve().parent.parent / "docs"
MD = DOCS / "report.md"
PDF = DOCS / "report.pdf"

# ----- editable cover fields (placeholders kept for the student) -------------
ASSIGNMENT_TITLE = "Individual Coursework of Big Data Programming Project"
MODULE_CODE = "ST5011CEM"
SUBMISSION_DATE = "[Date of Submission]"
STUDENT_ID = "[Student ID]"
STUDENT_NAME = "[Student Name]"
SUPERVISOR = "Siddhartha Neupane"

CSS = """
@page {
  size: A4; margin: 20mm 18mm 18mm 18mm;
  @bottom-right { content: counter(page); font-family: Georgia, serif;
                  font-size: 9pt; color: #444; }
}
@page cover { margin: 0; @bottom-right { content: ""; } }
@page nonum { @bottom-right { content: ""; } }

.page-border { position: fixed; top: -12mm; left: -10mm; right: -10mm; bottom: -10mm;
               border: 1px solid #555; z-index: -1; }

body { font-family: Georgia, 'Times New Roman', serif; font-size: 11pt;
       line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 17pt; font-weight: bold; color: #111; margin: 4px 0 10px;
     page-break-after: avoid; }
h2 { font-size: 15pt; font-weight: bold; color: #111; margin: 20px 0 8px;
     page-break-after: avoid; }
h3 { font-size: 12.5pt; font-weight: bold; color: #222; margin: 14px 0 6px;
     page-break-after: avoid; }
p { margin: 6px 0; text-align: justify; }
strong { color: #000; }
.content { counter-reset: figure; }

figure { margin: 14px auto; text-align: center; page-break-inside: avoid; }
figure img { max-width: 84%; max-height: 165mm; height: auto; }
figcaption { font-size: 9.5pt; font-style: italic; color: #333; margin-top: 5px; }

table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9.6pt;
        page-break-inside: avoid; }
th { border: 1px solid #999; padding: 6px 8px; background: #f2f2f2; text-align: left; }
td { border: 1px solid #bbb; padding: 5px 8px; }
ul, ol { margin: 6px 0 6px 4px; }
li { margin: 3px 0; text-align: justify; }
a { color: #1a4a7a; text-decoration: none; }

/* ---------- cover ---------- */
.cover { page: cover; position: relative; width: 210mm; height: 297mm;
         background: #fff; overflow: hidden; break-after: page; }
.tri { position: absolute; }
.cover .brand { position: absolute; top: 42mm; left: 0; right: 0; text-align: center; }
.cover .soft { font-family: Arial, sans-serif; font-size: 26pt; font-weight: bold;
               color: #1f4e79; letter-spacing: 1px; }
.cover .soft small { display: block; font-size: 10pt; letter-spacing: 3px; color: #2e75b6; }
.cover .collab { font-family: Arial, sans-serif; font-size: 10pt; color: #555;
                 margin-top: 10px; }
.cover .cov { font-family: Arial, sans-serif; font-size: 20pt; font-weight: bold;
              color: #0d3b66; }
.cover .mid { position: absolute; top: 108mm; left: 0; right: 0; text-align: center;
              font-family: Georgia, serif; }
.cover .lbl { color: #6b1f1f; font-weight: bold; font-size: 15pt; margin-top: 14px; }
.cover .val { font-size: 13pt; color: #222; margin-top: 2px; }
.cover .foot { position: absolute; top: 200mm; left: 24mm; right: 24mm;
               display: flex; justify-content: space-between;
               font-family: Georgia, serif; }
.cover .foot .lbl2 { color: #6b1f1f; font-weight: bold; font-size: 14pt; }
.cover .foot .v { font-size: 12pt; margin-top: 8px; color: #222; }

/* ---------- front matter ---------- */
.frontpage { page: nonum; break-after: page; }
.frontpage h2 { border-bottom: 2px solid #ccc; padding-bottom: 4px; }
.ack p { text-indent: 24px; }

ul.toc, ul.lof { list-style: none; margin: 0; padding: 0; font-size: 11pt; }
ul.toc li, ul.lof li { margin: 5px 0; }
ul.toc li.l3 { margin-left: 22px; font-size: 10.5pt; }
ul.toc a, ul.lof a { color: #111; }
ul.toc a::after, ul.lof a::after {
  content: leader('. ') target-counter(attr(href), page);
  color: #333; }
ul.toc li.l2 > a { font-weight: bold; font-style: italic; }
"""


def _slug(text, i):
    s = re.sub(r"<[^>]+>", "", text)
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return f"{s}-{i}" or f"sec-{i}"


def process(html):
    """Anchor headings (TOC) and wrap+number figures (LoF)."""
    toc, figs = [], []

    # headings -> add ids
    def h_repl(m):
        lvl, text = m.group(1), m.group(2)
        hid = _slug(text, len(toc))
        toc.append((lvl, re.sub(r"<[^>]+>", "", text), hid))
        return f'<h{lvl} id="{hid}">{text}</h{lvl}>'
    html = re.sub(r"<h([23])>(.*?)</h\1>", h_repl, html, flags=re.DOTALL)

    # <p><img><em>cap</em></p> -> numbered figure
    counter = {"n": 0}
    def f_repl(m):
        counter["n"] += 1
        n = counter["n"]
        img, cap = m.group(1), m.group(2).strip()
        fid = f"fig{n}"
        figs.append((n, cap, fid))
        return (f'<figure id="{fid}">{img}'
                f'<figcaption>Figure {n}: {cap}</figcaption></figure>')
    html = re.sub(r"<p>\s*(<img[^>]*>)\s*<em>(.*?)</em>\s*</p>", f_repl,
                  html, flags=re.DOTALL)
    return html, toc, figs


def cover_html():
    tris = (
        '<div class="tri" style="top:0;left:0;width:70mm;height:55mm;'
        'background:#2e75b6;clip-path:polygon(0 0,100% 0,0 100%)"></div>'
        '<div class="tri" style="top:0;left:0;width:45mm;height:80mm;'
        'background:#9dc3e6;clip-path:polygon(0 0,100% 0,0 100%);opacity:.6"></div>'
        '<div class="tri" style="top:0;right:0;width:60mm;height:40mm;'
        'background:#bdd7ee;clip-path:polygon(100% 0,100% 100%,0 0);opacity:.7"></div>'
        '<div class="tri" style="bottom:0;right:0;width:80mm;height:70mm;'
        'background:#1f4e79;clip-path:polygon(100% 0,100% 100%,0 100%)"></div>'
        '<div class="tri" style="bottom:0;right:0;width:55mm;height:100mm;'
        'background:#2e75b6;clip-path:polygon(100% 0,100% 100%,0 100%);opacity:.5"></div>'
        '<div class="tri" style="bottom:0;left:0;width:45mm;height:35mm;'
        'background:#9dc3e6;clip-path:polygon(0 100%,100% 100%,0 0);opacity:.6"></div>')
    return f"""
    <div class="cover">{tris}
      <div class="brand">
        <div class="soft">Softwarica<small>COLLEGE OF IT &amp; E-COMMERCE</small></div>
        <div class="collab">in collaboration with</div>
        <div class="cov">Coventry University</div>
      </div>
      <div class="mid">
        <div class="lbl">Assignment Title:</div>
        <div class="val">{ASSIGNMENT_TITLE}</div>
        <div class="lbl">Module Code:</div>
        <div class="val">{MODULE_CODE}</div>
        <div class="lbl">Date of Submission:</div>
        <div class="val">{SUBMISSION_DATE}</div>
      </div>
      <div class="foot">
        <div><div class="lbl2">Submitted by:</div>
             <div class="v">{STUDENT_ID}<br>{STUDENT_NAME}</div></div>
        <div><div class="lbl2">Submitted to:</div>
             <div class="v">{SUPERVISOR}</div></div>
      </div>
    </div>"""


def front_matter(toc, figs):
    ack = ("""<div class="frontpage ack"><h2>Acknowledgement</h2>
      <p>I want to extend my heartfelt thanks to <strong>Siddhartha Neupane Sir</strong>
      for his invaluable support and expert guidance throughout this project. His
      encouragement, knowledge and continuous motivation played a crucial role in
      helping me successfully complete this work.</p>
      <p>I am also deeply grateful to the contributors of the open-data platforms,
      research communities and development tools that made this project possible.
      Their shared knowledge and resources greatly supported the learning and
      implementation process.</p></div>""")
    toc_items = "".join(
        f'<li class="l{lvl}"><a href="#{hid}">{text}</a></li>'
        for lvl, text, hid in toc)
    toc_html = (f'<div class="frontpage"><h2>Table of Contents</h2>'
                f'<ul class="toc">{toc_items}</ul></div>')
    lof_items = "".join(
        f'<li><a href="#{fid}">Figure {n}: {cap}</a></li>'
        for n, cap, fid in figs)
    lof_html = (f'<div class="frontpage"><h2>Table of Figures</h2>'
                f'<ul class="lof">{lof_items}</ul></div>')
    return ack + toc_html + lof_html


def main():
    body = markdown.markdown(MD.read_text(),
                             extensions=["tables", "attr_list", "sane_lists"])
    body, toc, figs = process(body)
    html = (f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style>"
            f"</head><body><div class='page-border'></div>"
            f"{cover_html()}{front_matter(toc, figs)}"
            f"<div class='content'>{body}</div></body></html>")
    (DOCS / "report_build.html").write_text(html)
    weasyprint.HTML(string=html, base_url=str(DOCS)).write_pdf(str(PDF))
    print(f"[pdf] wrote {PDF.name} ({PDF.stat().st_size/1024:.0f} KB) — "
          f"{len(toc)} TOC entries, {len(figs)} figures")


if __name__ == "__main__":
    main()
