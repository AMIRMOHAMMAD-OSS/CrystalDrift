"""Render the evidence-generated morning memo as one independently deliverable page."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent

def escape(s):
    tokens={"\\":r"\textbackslash{}", "&":r"\&", "%":r"\%", "_":r"\_", "#":r"\#",
            "{":r"\{","}":r"\}", "²":r"\textsuperscript{2}","Å":r"\AA{}"}
    return "".join(tokens.get(c,c) for c in s)

def main():
    source=(HERE/"MORNING.md").read_text(encoding="utf8")
    paragraphs=[p.strip() for p in source.split("\n\n") if p.strip()]
    title=paragraphs.pop(0).removeprefix("# ")
    body=[]
    for line in paragraphs:
        match=re.fullmatch(r"\*\*(.*?)\*\*\s*(.*)",line,re.S)
        if not match:raise ValueError(line)
        body.append(r"\noindent\textbf{"+escape(match[1])+"} "+escape(match[2])+r"\par")
    tex=r"""\documentclass[10pt,letterpaper]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\setlength{\textwidth}{7.06in}\setlength{\oddsidemargin}{-.28in}
\setlength{\textheight}{9.56in}\setlength{\topmargin}{-.65in}
\pagestyle{empty}\setlength{\parindent}{0pt}\setlength{\parskip}{7pt}\setlength{\emergencystretch}{3em}
\begin{document}
{\Large\bfseries """+escape(title)+r"""}\hfill 21 November 2026
\par\vspace{-2pt}\hrule\vspace{3pt}
"""+ "\n".join(body) +r"""
\end{document}"""
    with tempfile.TemporaryDirectory(prefix="crytaldrift-morning-") as path:
        work=Path(path);(work/"memo.tex").write_text(tex,encoding="utf8")
        exe=shutil.which("pdflatex")
        cmd=[exe,"-no-shell-escape","-interaction=nonstopmode","-halt-on-error","memo.tex"]
        if "miktex" in exe.lower():cmd.insert(1,"-disable-installer")
        p=subprocess.run(cmd,cwd=work,capture_output=True,timeout=90)
        if p.returncode:raise RuntimeError(p.stdout.decode(errors="replace")[-1500:])
        info=subprocess.check_output(["pdfinfo",str(work/"memo.pdf")]).decode(errors="replace")
        if not re.search(r"^Pages:\s+1\s*$",info,re.M):raise ValueError("Memo is longer than one page.")
        shutil.copyfile(work/"memo.pdf",HERE/"MORNING.pdf")
    print("Verified one-page author brief",flush=True)
if __name__=="__main__":main()
