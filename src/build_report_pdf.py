"""
Convert docs/report.md -> a polished, self-contained docs/report.pdf.

Pipeline: Markdown -> HTML (tables) -> embed images as base64 -> academic CSS
-> headless Chrome --print-to-pdf.  No LaTeX/pandoc required.

Run:  .venv/bin/python src/build_report_pdf.py
"""
from __future__ import annotations

import base64
import re
import subprocess
from pathlib import Path

import markdown

DOCS = Path(__file__).resolve().parent.parent / "docs"
MD = DOCS / "report.md"
HTML = DOCS / "report.html"
PDF = DOCS / "report.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 20mm 18mm; }
* { box-sizing: border-box; }
body { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 10.6pt;
       line-height: 1.5; color: #1a1a1a; max-width: 100%; }
h1 { font-size: 20pt; color: #1a5276; margin: 0 0 2px; line-height: 1.2; }
h1 + h3 { color: #555; font-weight: 500; margin-top: 0; font-size: 12pt; }
h2 { font-size: 13.5pt; color: #1a5276; border-bottom: 2px solid #d5dbdb;
     padding-bottom: 3px; margin-top: 22px; page-break-after: avoid; }
h3 { font-size: 11.5pt; color: #34495e; }
p { margin: 7px 0; text-align: justify; }
strong { color: #212f3d; }
hr { border: none; border-top: 1px solid #ccc; margin: 14px 0; }
figure { margin: 14px auto; text-align: center; page-break-inside: avoid; }
figure img { max-width: 86%; height: auto; border: 1px solid #e5e7eb;
             border-radius: 4px; }
figcaption { font-size: 8.6pt; color: #566; font-style: italic; margin-top: 5px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9.4pt;
        page-break-inside: avoid; }
th { background: #1a5276; color: #fff; padding: 6px 8px; text-align: left; }
td { border: 1px solid #d5dbdb; padding: 5px 8px; }
tr:nth-child(even) td { background: #f4f6f7; }
em { color: #34495e; }
code { background: #f0f2f4; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
.cover-meta p { margin: 2px 0; text-align: left; }
"""


def embed_images(html: str) -> str:
    def repl(m):
        src = m.group(1)
        path = (DOCS / src).resolve()
        if not path.exists():
            print(f"  [warn] missing image {src}")
            return m.group(0)
        data = base64.b64encode(path.read_bytes()).decode()
        ext = path.suffix.lstrip(".") or "png"
        return f'src="data:image/{ext};base64,{data}"'
    return re.sub(r'src="([^"]+)"', repl, html)


def wrap_figures(html: str) -> str:
    # <p><img ...><em>caption</em></p>  ->  <figure><img><figcaption>..</figcaption></figure>
    pat = re.compile(r'<p>\s*(<img[^>]*>)\s*<em>(.*?)</em>\s*</p>', re.DOTALL)
    return pat.sub(r'<figure>\1<figcaption>\2</figcaption></figure>', html)


def main():
    body = markdown.markdown(MD.read_text(),
                             extensions=["tables", "attr_list", "sane_lists"])
    body = wrap_figures(body)
    body = embed_images(body)
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    HTML.write_text(html)
    print(f"[html] wrote {HTML.name} ({len(html)//1024} KB)")

    # headless Chrome -> PDF
    proc = subprocess.Popen([
        CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        "--no-first-run", "--user-data-dir=/tmp/chrome_pdf_profile",
        f"--print-to-pdf={PDF}", HTML.as_uri(),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        proc.wait(timeout=45)
    except subprocess.TimeoutExpired:
        proc.kill()
    if PDF.exists():
        print(f"[pdf] wrote {PDF.name} ({PDF.stat().st_size/1024:.0f} KB)")
    else:
        print("[pdf] FAILED to generate PDF")


if __name__ == "__main__":
    main()
