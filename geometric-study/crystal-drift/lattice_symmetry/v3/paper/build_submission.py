"""Build the minimal arXiv source and compile twice from a pristine extraction."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile
from xml.etree import ElementTree as ET

HERE=Path(__file__).resolve().parent
RELEASE=HERE.parent/"release"
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest() if hasattr(hashlib,"file_digest") else hashlib.sha256(f.read()).hexdigest()
def inputs():
    todo=["main.tex"];files=set()
    while todo:
        filename=todo.pop()
        if filename in files:continue
        p=HERE/filename
        if not p.is_file():raise FileNotFoundError(p)
        files.add(filename)
        if p.suffix!=".tex":continue
        body=p.read_text(encoding="utf8")
        for child in re.findall(r"\\input\{([^}]+)\}",body):
            todo.append(child if Path(child).suffix else child+".tex")
        todo.extend(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",body))
    return sorted(files)

def command(args,where):
    out=subprocess.run(args,cwd=where,capture_output=True)
    if out.returncode:
        raise RuntimeError(out.stdout.decode(errors="replace")[-2500:]+out.stderr.decode(errors="replace")[-1200:])
    return out.stdout

def audit_layout(source,out):
    from PIL import Image,ImageDraw
    out.mkdir(exist_ok=True,parents=True)
    command(["pdftoppm","-r","96","-png",str(source),"page"],out)
    xml=command(["pdftotext","-enc","UTF-8","-bbox",str(source),"-"],out).decode("utf8")
    # Poppler can export an unassigned math-font glyph as an XML-forbidden
    # control character. Keep its word bounding box for the clipping audit.
    xml=re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "\ufffd", xml)
    root=ET.fromstring(xml)
    pages=[]
    for page in root.findall(".//{*}page"):
        words=page.findall(".//{*}word")
        if not words:raise ValueError("Empty manuscript page")
        bounds=[min(float(w.get("xMin")) for w in words),min(float(w.get("yMin")) for w in words),
                max(float(w.get("xMax")) for w in words),max(float(w.get("yMax")) for w in words)]
        pages.append(dict(words=len(words),bounds=bounds,width=float(page.get("width")),height=float(page.get("height"))))
    files=sorted(out.glob("page-*.png"),key=lambda x:int(x.stem.split("-")[-1]))
    assert len(files)==len(pages)
    for start in range(0,len(files),4):
        sheet=Image.new("RGB",(920,1250),"#cccccc");draw=ImageDraw.Draw(sheet)
        for j,p in enumerate(files[start:start+4]):
            with Image.open(p) as im:
                im.thumbnail((450,593));x=(j%2)*460+5;y=(j//2)*625+5
                sheet.paste(im,(x,y));draw.text((x,y+595),"Page "+str(start+j+1),fill="black")
        sheet.save(out/f"overview-{start+1:02}.png")
    return pages

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--preview",action="store_true");args=parser.parse_args()
    from sync import sync
    sync()
    name="working" if args.preview else "CrytalDrift"
    files=inputs();RELEASE.mkdir(exist_ok=True)
    archive=RELEASE/(name+"-arxiv-source.zip")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for f in files:
            info=zipfile.ZipInfo(f,(2026,11,21,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,(HERE/f).read_bytes())
    identity=sha(archive)
    # A distinct directory for each run prevents stale auxiliary files from satisfying references.
    root=RELEASE/"qa"/(name+"-"+identity[:12]);turn=1;build=root/f"build-{turn}"
    while build.exists():turn+=1;build=root/f"build-{turn}"
    build.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and sorted(z.namelist())==files
        z.extractall(build)
    binary=shutil.which("pdflatex")
    passes=[]
    for i in (1,2):
        cmd=[binary,"-no-shell-escape","-interaction=nonstopmode","-halt-on-error"]
        if "miktex" in binary.lower():cmd.insert(1,"-disable-installer")
        command(cmd+["main.tex"],build)
        report=(build/"main.log").read_text(errors="replace")
        passes.append(dict(pass_number=i,log_sha=sha(build/"main.log"),undefined="There were undefined references" in report))
    unresolved=re.findall(r"LaTeX Warning: (?:Citation|Reference).+?undefined",report)
    if unresolved or passes[-1]["undefined"]:raise RuntimeError(unresolved)
    source=build/"main.pdf";pdf=RELEASE/(name+".pdf");shutil.copyfile(source,pdf)
    view=build/"view";pages=audit_layout(source,view)
    fonts=command(["pdffonts",str(source)],build).decode(errors="replace")
    extracted=command(["pdftotext","-layout",str(source),"-"],build).decode(errors="replace")
    assert "AmirM.MohammadHosseini" in extracted and "CrytalDrift" in extracted
    report_data=dict(pdf=pdf.name,pdf_sha=sha(pdf),source=archive.name,source_sha=identity,members=files,
        passes=passes,layout=pages,fonts=fonts,unresolved=unresolved,
        overfull=re.findall(r"Overfull \\[hv]box \([^)]+\)[^\n]*",report),
        latex_warnings=re.findall(r"(?:LaTeX(?: Font)? Warning|Package [A-Za-z-]+ Warning):[^\n]*",report),review=view.relative_to(RELEASE).as_posix())
    receipt=RELEASE/(name+"-build.json");receipt.write_text(json.dumps(report_data,indent=2))
    print(pdf,len(pages),"pages;",len(files),"source files;",len(report_data["overfull"]),"overfull diagnostics",flush=True)
if __name__=="__main__":main()
