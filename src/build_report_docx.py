"""
Build an editable Microsoft Word version of the report: docs/report.docx.

Transforms docs/report.md into a pandoc-ready document (numbered "Figure N:"
captions, a cover page, and Word TOC + Table-of-Figures fields), then converts
it with pandoc. Headings use built-in Word Heading styles, so the TOC/ToF
fields populate when you open the file and choose "Update Field".

Requires: pandoc  (brew install pandoc).
Run:  .venv/bin/python src/build_report_docx.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "docs" / "report.md"
OUT = ROOT / "docs" / "report.docx"
TMP = Path("/tmp/report_pandoc.md")

PAGEBREAK = '\n```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n'
TOC_FIELD = ('\n```{=openxml}\n'
  '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Table of Contents</w:t></w:r></w:p>'
  '<w:sdt><w:sdtPr><w:docPartObj><w:docPartGallery w:val="Table of Contents"/>'
  '<w:docPartUnique/></w:docPartObj></w:sdtPr><w:sdtContent>'
  '<w:p><w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>'
  '<w:r><w:instrText xml:space="preserve"> TOC \\o "1-3" \\h \\z \\u </w:instrText></w:r>'
  '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
  '<w:r><w:t>Right-click and choose "Update Field" to build the Table of Contents.</w:t></w:r>'
  '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:sdtContent></w:sdt>\n```\n')
LOF_FIELD = ('\n```{=openxml}\n'
  '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Table of Figures</w:t></w:r></w:p>'
  '<w:p><w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>'
  '<w:r><w:instrText xml:space="preserve"> TOC \\h \\z \\c "Figure" </w:instrText></w:r>'
  '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
  '<w:r><w:t>Right-click and choose "Update Field" to build the Table of Figures.</w:t></w:r>'
  '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>\n```\n')
COVER = (
 "**Softwarica College of IT & E-commerce**\n\n"
 "*in collaboration with*\n\n"
 "**Coventry University**\n\n\n\n"
 "**Assignment Title:** Individual Coursework of Big Data Programming Project\n\n"
 "**Module Code:** ST5011CEM\n\n"
 "**Date of Submission:** [Date of Submission]\n\n\n\n"
 "**Submitted by:** [Student ID] — [Student Name]\n\n"
 "**Submitted to:** Siddhartha Neupane\n")


def number_figures(md_lines):
    out, n, i = [], 0, 0
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
            out.append(f'![Figure {n}: {cap}]({m.group(1)}){{width=78%}}')
            out.append("")
            i = j + 1
            continue
        out.append(md_lines[i]); i += 1
    return "\n".join(out), n


def main():
    body, n = number_figures(MD.read_text().splitlines())
    TMP.write_text(COVER + PAGEBREAK + TOC_FIELD + LOF_FIELD + PAGEBREAK + body)
    subprocess.run(
        ["pandoc", str(TMP), "-o", str(OUT), "--resource-path", str(ROOT / "docs"),
         "--from=markdown+raw_attribute"], check=True)
    print(f"[docx] wrote {OUT.name} ({OUT.stat().st_size/1024:.0f} KB, {n} figures). "
          f"Open in Word and Update Fields to build the TOC / Table of Figures.")


if __name__ == "__main__":
    main()
