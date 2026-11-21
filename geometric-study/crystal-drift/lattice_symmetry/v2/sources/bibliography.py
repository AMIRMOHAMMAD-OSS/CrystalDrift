"""Resolve only the references cited by the new manuscript into a self-contained bibliography."""
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
V2=HERE.parent
ROOT=V2.parents[1]
PAPER=V2/"paper"


def safe(s):
    for a,b in (("&",r"\&"),("%",r"\%"),("_",r"\_"),("–","--"),("—","---"),("‐","-"),("‑","-"),("’","'")):
        s=s.replace(a,b)
    return s


def bibliography():
    citations=[]
    visited=set()
    def read(path):
        if path in visited:return
        visited.add(path)
        text=path.read_text(encoding="utf8")
        for token in re.finditer(r"\\(input|cite)\{([^}]+)\}",text):
            cmd,arg=token.groups()
            if cmd=="cite":
                for k in arg.split(","):
                    if k.strip() not in citations:citations.append(k.strip())
            elif arg!="bibliography":
                read(PAPER/(arg if arg.endswith(".tex") else arg+".tex"))
    read(PAPER/"main.tex")
    recs={r["key"]:r for r in json.loads((HERE/"primary-records.json").read_text(encoding="utf8"))}
    old={m[1]:m[0] for m in re.finditer(r"(?m)^\\bibitem\{(.*?)\}.*$",(ROOT/"paper/bibliography.tex").read_text(encoding="utf8"))}
    published=json.loads((HERE/"published.json").read_text(encoding="utf8"))
    lines=[r"\begin{thebibliography}{99}\small\setlength{\itemsep}{1.5pt}"]
    for key in citations:
        if key in published:
            r=published[key]
            a=r["authors"]
            authors=a[0]
            if len(a)>4:
                authors+=" et al"
            elif len(a)>1:
                authors=", ".join(a[:-1])+" and "+a[-1]
            lines.append(r"\bibitem{"+key+"} "+safe(authors)+r". \linkto{"+r["url"]+"}{"+safe(r["title"])+"}. "+safe(r["citation"])+" ("+str(r["year"])+").")
            continue
        if key in old:
            lines.append(old[key]);continue
        r=recs[key]
        assert r["metadata_verified"] and r.get("title") and r.get("year"),r
        author=r["first_author"]
        if "," in author:
            family,given=author.split(",",1)
            author=" ".join(x[0]+"." for x in given.split())+" "+family
        # A reference with only a family name in the Crossref fallback still identifies its original author.
        line=r"\bibitem{"+key+"} "+safe(author)+r" et al. \linkto{"+r["url"]+"}{"+safe(r["title"])+"}. "
        if "arxiv.org/abs/" in r["url"]:
            line+="arXiv:"+r["url"].rsplit("/",1)[-1]+" "
        lines.append(line+"("+str(r["year"])+").")
    lines.append(r"\end{thebibliography}")
    (PAPER/"bibliography.tex").write_text("\n".join(lines)+"\n",encoding="utf8")
    print(len(citations),"cited references")


if __name__=="__main__":bibliography()
