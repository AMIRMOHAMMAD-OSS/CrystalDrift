"""Build and verify the v2 manuscript, per-draw evidence overlay and editable figures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

V2=Path(__file__).resolve().parent
ROOT=V2.parents[1]
PAPER=V2/"paper"
OUT=V2/"release"
SOURCE=OUT/"CrytalDrift-periodic-arxiv.zip"
PDF=OUT/"CrytalDrift-periodic.pdf"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as fp:
        for buf in iter(lambda:fp.read(4<<20),b""):h.update(buf)
    return h.hexdigest()


def tex_files():
    seen=set()
    def collect(path):
        if path in seen:return
        assert path.parent==PAPER and path.is_file(),path
        seen.add(path)
        if path.suffix==".tex":
            s=path.read_text(encoding="utf8")
            for name in re.findall(r"\\input\{([^}]+)\}",s):
                collect(PAPER/(name if name.endswith(".tex") else name+".tex"))
            for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",s):
                collect(PAPER/name)
    collect(PAPER/"main.tex")
    return seen


def zip_checked(path,files):
    OUT.mkdir(exist_ok=True)
    tmp=path.with_suffix(".building")
    names=set()
    with ZipFile(tmp,"w",ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
        for p,name in sorted(files,key=lambda x:x[1]):
            assert name not in names and not p.is_symlink(),name
            names.add(name);z.write(p,name)
    with ZipFile(tmp) as z:
        assert z.testzip() is None and set(z.namelist())==names
    tmp.replace(path)
    print(path.name,len(names),"members",path.stat().st_size,"bytes",flush=True)


def source(stage):
    src=tex_files()
    zip_checked(SOURCE,[(p,p.name) for p in src])
    if stage is not None:
        stage.mkdir(parents=True,exist_ok=False)
        with ZipFile(SOURCE) as z:z.extractall(stage)
        for p in src:assert (stage/p.name).read_bytes()==p.read_bytes()


def verify(stage):
    from pypdf import PdfReader
    src=tex_files()
    text="\n".join(p.read_text(encoding="utf8") for p in src if p.suffix==".tex")
    labels=re.findall(r"\\label\{([^}]+)\}",text)
    refs=re.findall(r"\\(?:eq)?ref\{([^}]+)\}",text)
    assert len(set(labels))==len(labels) and set(refs)<=set(labels)
    cited={v.strip() for c in re.findall(r"\\cite\{([^}]+)\}",text) for v in c.split(",")}
    bib=re.findall(r"\\bibitem\{([^}]+)\}",text)
    assert set(bib)==cited and len(bib)==len(cited)
    with ZipFile(SOURCE) as z:
        assert {p.name for p in src}==set(z.namelist())
        for p in src:assert p.read_bytes()==z.read(p.name)==(stage/p.name).read_bytes(),p.name
    log=(stage/"main.log").read_text(errors="replace")
    for err in ("undefined references","undefined citations","Overfull","LaTeX Warning:","Package hyperref Warning:","Fatal error","! "):
        assert err not in log,err
    pdf=PdfReader(stage/"main.pdf")
    pages=[p.extract_text() for p in pdf.pages]
    assert pdf.metadata.author=="AmirM.MohammadHosseini" and "CrytalDrift" in pdf.metadata.title
    assert all(s.strip() and "??" not in s for s in pages)
    bibpage=next(i for i,s in enumerate(pages,1) if re.search(r"(?m)^References\s*$",s) and "[1]" in s)
    supp=next(i for i,s in enumerate(pages,1) if "Mathematical and experimental supplement" in s)
    assert bibpage<supp
    fonts=set();seenres=set();uris=set()
    def embedded(res):
        if not res:return
        res=res.get_object()
        ref=getattr(res,"indirect_reference",None)
        if ref:
            if ref.idnum in seenres:return
            seenres.add(ref.idnum)
        for face in res.get("/Font",{}):
            f=res["/Font"][face]
            for candidate in f.get("/DescendantFonts",[f]):
                font=candidate.get_object();desc=font.get("/FontDescriptor")
                assert (font.get("/Subtype")=="/Type3" and "/CharProcs" in font) or (desc and any(k in desc.get_object() for k in ("/FontFile","/FontFile2","/FontFile3"))),font
                fonts.add(str(font.get("/BaseFont",face)))
        for x in res.get("/XObject",{}):
            embedded(res["/XObject"][x].get("/Resources"))
    for page in pdf.pages:
        embedded(page.get("/Resources"))
        for ann in page.get("/Annots",[]):
            act=ann.get_object().get("/A",{})
            if act.get("/URI"):uris.add(str(act["/URI"]))
    expected=set(re.findall(r"\\linkto\{([^}]+)\}",(PAPER/"bibliography.tex").read_text(encoding="utf8")))
    assert expected<=uris,expected-uris
    countfig=len(re.findall(r"\\includegraphics",text))
    aux=(stage/"main.aux").read_text(encoding="utf8")
    assert countfig==len(re.findall(r"\\@writefile\{lof\}",aux))
    abstract=(PAPER/"abstract.tex").read_text(encoding="utf8")
    abstract=" ".join(abstract.replace(r"\%","%").replace(r"\,"," ").replace("~"," ").replace("--","-").split())
    assert len(abstract)<=1920 and "\\" not in abstract,abstract
    shutil.copyfile(stage/"main.pdf",PDF)
    (OUT/"submission-metadata.txt").write_text("Title\nCrytalDrift: One-step generation in the geometry of periodic crystals\n\nAuthor\nAmirM.MohammadHosseini\n\nAffiliation\nDepartment of Biotechnology, University of Tehran\n\nSuggested primary category\ncond-mat.mtrl-sci\n\nAbstract\n"+abstract+f"\n\nComments\n{len(pages)} pages, {countfig} figures, including the mathematical and experimental supplement.\n",encoding="utf8")
    check=dict(pages=len(pages),figures=countfig,citations=len(cited),bibliography_page=bibpage,
               supplement_page=supp,unique_labels=len(labels),embedded_fonts=len(fonts),bibliography_links=len(expected),
               source_sha256=sha(SOURCE),pdf_sha256=sha(PDF),members={p.name:sha(p) for p in sorted(src)})
    (OUT/"clean-build.json").write_text(json.dumps(check,indent=2)+"\n",encoding="utf8")
    print({k:v for k,v in check.items() if k!="members"})


def evidence():
    files=[]
    allowed={".py",".md",".json",".csv",".gz",".npz",".npy",".pt",".tex",".svg",".pdf",".txt",".yml",".yaml",".sh",".slurm",".cif",".tsv",".png",".cpp",".pyx"}
    for folder,ds,ns in os.walk(V2):
        ds[:]=[v for v in ds if v not in {"__pycache__","release","build"} and not v.startswith(".")]
        for n in ns:
            p=Path(folder)/n
            if (p.suffix.lower() in allowed or n.upper().startswith("LICENSE")) and "-partial." not in n and not n.startswith("."):
                files.append((p,"crystal-drift/"+p.relative_to(ROOT).as_posix()))
    for p in [*sorted(V2.parent.glob("*.py")),V2.parent/"README.md",V2.parent/"CODEX_GOAL.md"]:
        files.append((p,"crystal-drift/"+p.relative_to(ROOT).as_posix()))
    # The previous evidence ZIP cannot contain itself. Its companion bundles and seal
    # are included here so extracting both archives restores every audit input.
    for p in (ROOT/"release-sha256.txt",ROOT/"revision/editable-figures.zip",ROOT/"generated-crystals.zip",
              ROOT/"expansion/benchmark-tables.zip",PDF,SOURCE,OUT/"clean-build.json",OUT/"submission-metadata.txt"):
        files.append((p,"crystal-drift/"+p.relative_to(ROOT).as_posix()))
    inventory={name:dict(sha256=sha(p),bytes=p.stat().st_size) for p,name in files}
    inv=OUT/"evidence-inventory.json"
    inv.write_text(json.dumps(inventory,indent=1)+"\n",encoding="utf8")
    files.append((inv,"crystal-drift/"+inv.relative_to(ROOT).as_posix()))
    zip_checked(OUT/"CrytalDrift-periodic-evidence.zip",files)
    with ZipFile(OUT/"CrytalDrift-periodic-evidence.zip") as z:
        assert len(z.namelist())==len(inventory)+1
        for name,record in inventory.items():
            h=hashlib.sha256()
            with z.open(name) as stream:
                for block in iter(lambda:stream.read(4<<20),b""):h.update(block)
            assert h.hexdigest()==record["sha256"],name


def extras():
    plots=[]
    for p in tex_files():
        if p.suffix==".pdf":
            matches=[x for x in V2.rglob(p.stem+".svg") if OUT not in x.parents]
            assert matches,p
            plots.append((matches[0],p.stem+".svg"))
    scripts=["figures/method.py","equivalence_figure.py","csp/report.py","csp/training_figure.py",
             "wyckoff/example.py","wyckoff/report.py","denovo/render.py"]
    for name in scripts:plots.append((V2/name,"plotting/"+name))
    zip_checked(OUT/"CrytalDrift-periodic-figures.zip",plots)
    cifs=sorted(V2.rglob("*.cif"))
    assert cifs and any("wyckoff" in p.parts for p in cifs) and any("denovo" in p.parts for p in cifs)
    zip_checked(OUT/"CrytalDrift-periodic-cifs.zip",[(p,p.relative_to(V2).as_posix()) for p in cifs
        if "external" not in p.parts and OUT not in p.parents])
    tables=sorted(list(V2.rglob("*.csv"))+list(V2.rglob("*.csv.gz")))
    tables=[p for p in tables if OUT not in p.parents and "external" not in p.parts]
    for p in (V2/"audit.json",V2/"CLAIM-AUDIT.md",V2/"denovo/verified.json",V2/"csp/selection.json"):
        tables.append(p)
    zip_checked(OUT/"CrytalDrift-periodic-tables.zip",[(p,p.relative_to(V2).as_posix()) for p in tables])


def seal():
    names=[SOURCE,PDF,*[OUT/("CrytalDrift-periodic-"+n+".zip") for n in ("evidence","figures","cifs","tables")]]
    b=json.loads((OUT/"clean-build.json").read_text())
    assert sha(SOURCE)==b["source_sha256"] and sha(PDF)==b["pdf_sha256"]
    (OUT/"SHA256SUMS.txt").write_text("".join(f"{sha(p)}  {p.name}\n" for p in names))
    print("Sealed PDF and five ZIPs.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["source","verify","evidence","extras","seal"])
    p.add_argument("--stage",type=Path)
    a=p.parse_args()
    if a.action in ("source","verify"):globals()[a.action](a.stage)
    else:globals()[a.action]()
