"""Keep the Springer (LNCS) version of Paper 1 in sync with the working version.

The article-class file negoplay_paper1.tex is the single source of the text.
This script transplants its abstract text and its whole body (from the
Introduction marker to \\end{document}) into submission/negoplay_paper1_lncs.tex,
whose preamble, title block and \\keywords line stay as they are.

Usage (from this folder):  python sync_lncs.py
Then compile both files with xelatex (three passes each).
"""
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ART = HERE / "negoplay_paper1.tex"
LNCS = HERE / "submission" / "negoplay_paper1_lncs.tex"

MARK = "% ==================================================================\n\\section{Introduction}\\label{sec:intro}\n"
END = "\\end{document}\n"


def main() -> None:
    art = ART.read_text(encoding="utf-8")
    lncs = LNCS.read_text(encoding="utf-8")

    m_art = re.search(r"\\begin\{abstract\}\n\\noindent\n(.*?)\n\\end\{abstract\}", art, re.S)
    m_lncs = re.search(r"(\\begin\{abstract\}\n\\noindent\n)(.*?)(\n\n\\keywords\{.*?\}\n\\end\{abstract\})",
                       lncs, re.S)
    if not (m_art and m_lncs):
        sys.exit("abstract anchors not found")
    lncs = lncs[:m_lncs.start(2)] + m_art.group(1) + lncs[m_lncs.end(2):]

    for name, s in (("article", art), ("lncs", lncs)):
        if s.count(MARK) != 1 or s.count(END) != 1:
            sys.exit(f"{name}: body anchors not unique (MARK={s.count(MARK)}, END={s.count(END)})")
    body = art[art.index(MARK):art.index(END)]
    lncs = lncs[:lncs.index(MARK)] + body + lncs[lncs.index(END):]

    LNCS.write_text(lncs, encoding="utf-8")
    print("ok: LNCS synced;", len(lncs.splitlines()), "lines")


if __name__ == "__main__":
    main()
