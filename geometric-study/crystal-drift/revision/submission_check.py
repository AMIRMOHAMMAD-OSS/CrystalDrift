"""Inspect a twice-compiled clean extraction and seal the submission PDF and metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from zipfile import ZipFile

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT/"paper"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(stage):
    with ZipFile(PAPER/"crystal-drift-arxiv.zip") as package:
        assert package.testzip() is None
        names = sorted(package.namelist())
        for name in names:
            assert name==Path(name).name
            assert package.read(name)==(stage/name).read_bytes()==(PAPER/name).read_bytes(), name
    sources = "\n".join((stage/n).read_text(encoding="utf8") for n in names if n.endswith(".tex"))
    labels = re.findall(r"\\label\{([^}]+)\}", sources)
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", sources))
    assert len(labels)==len(set(labels)) and refs <= set(labels)
    citations = {x for group in re.findall(r"\\cite\{([^}]+)\}",sources) for x in group.split(",")}
    bib = re.findall(r"\\bibitem\{([^}]+)\}",sources)
    assert len(bib)==len(set(bib)) and set(bib)==citations
    logs = (stage/"main.log").read_text(encoding="utf8",errors="replace")
    for problem in ("undefined","Overfull","LaTeX Warning","Fatal error","! "):
        assert problem not in logs, problem
    reader = PdfReader(stage/"main.pdf")
    fonts = {}
    seen_resources = set()

    def check_fonts(resources):
        if resources is None:
            return
        obj = resources.get_object()
        ref = getattr(obj, "indirect_reference", None)
        if ref and ref.idnum in seen_resources:
            return
        if ref:
            seen_resources.add(ref.idnum)
        for key in obj.get("/Font", {}):
            face = obj["/Font"][key]
            for item in face.get("/DescendantFonts", [face]):
                font = item.get_object()
                desc = font.get("/FontDescriptor")
                embedded = ("/CharProcs" in font if font.get("/Subtype") == "/Type3" else
                            bool(desc and any(x in desc.get_object() for x in ("/FontFile","/FontFile2","/FontFile3"))))
                name = str(font.get("/BaseFont", key))
                assert embedded, f"Unembedded font: {name}"
                fonts[name] = True
        for key in obj.get("/XObject", {}):
            check_fonts(obj["/XObject"][key].get("/Resources"))

    for page in reader.pages:
        check_fonts(page.get("/Resources"))
    n = len(reader.pages)
    assert reader.metadata.author=="AmirM.MohammadHosseini" and "CrytalDrift" in reader.metadata.title
    destinations = [node.title for node in reader.outline if hasattr(node,"title")]
    assert len(destinations)>=4, destinations
    pages = [page.extract_text() for page in reader.pages]
    assert all(s.strip() and "??" not in s for s in pages)
    bib_pages = [i for i,s in enumerate(pages,1) if re.search(r"(?m)^References[ \t]*$",s) and "[1]" in s]
    assert len(bib_pages)==1, bib_pages
    bib_page = bib_pages[0]
    math_page = next(i for i,s in enumerate(pages,1) if "Mathematical supplement" in s and "S1" in s)
    assert bib_page < math_page < n
    uris = set()
    for page in reader.pages:
        for annot in page.get("/Annots", []):
            action = annot.get_object().get("/A", {})
            if action.get("/URI"):
                uris.add(str(action["/URI"]))
    required_uris = set(re.findall(r"\\linkto\{([^}]+)\}", (stage/"bibliography.tex").read_text()))
    assert required_uris <= uris, required_uris - uris
    count_figures = len(re.findall(r"\\@writefile\{lof\}", (stage/"main.aux").read_text()))
    assert count_figures == len(re.findall(r"\\includegraphics",sources))
    metadata = (PAPER/"submission-metadata.txt").read_text(encoding="utf8")
    main = (stage/"main.tex").read_text(encoding="utf8")
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",main,re.S)[1]
    abstract = " ".join(abstract.strip().replace(r"\,"," ").replace(r"\%","%").replace("--","-").split())
    assert len(abstract)<=1920 and "\\" not in abstract
    metadata = re.sub(r"(?<=Comments\n).*?(?=\n\nAbstract)",f"{n} pages, {count_figures} figures. The mathematical supplement and additional experimental results follow the references in the same PDF.",metadata,flags=re.S)
    metadata = re.sub(r"(?<=Abstract\n).*?(?=\n\nSource package)",abstract,metadata,flags=re.S)
    (PAPER/"submission-metadata.txt").write_text(metadata,encoding="utf8")
    for name in ("main.pdf","main.aux","main.log"):
        shutil.copyfile(stage/name,PAPER/name)
    verification = dict(source_zip_sha256=sha(PAPER/"crystal-drift-arxiv.zip"),pdf_sha256=sha(PAPER/"main.pdf"),
        compiler="pdfLaTeX, MiKTeX 25.12, two sequential passes in a fresh extraction with package installation and shell escape disabled",
        pages=n,figures=count_figures,bibliography_entries=len(bib),linked_bibliography_urls=len(required_uris),
        unique_labels=len(labels),embedded_fonts=len(fonts),reference_start_page=bib_page,supplement_start_page=math_page,outlines=destinations,
        members={name:sha(stage/name) for name in names})
    (ROOT/"revision/package-verification.json").write_text(json.dumps(verification,indent=2)+"\n",encoding="utf8")
    print(f"Verified {n} pages, {count_figures} figures, {len(bib)} cited references and {len(names)} source files.")
    print("PDF",verification["pdf_sha256"],"ZIP",verification["source_zip_sha256"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",type=Path)
    check(parser.parse_args().stage)
