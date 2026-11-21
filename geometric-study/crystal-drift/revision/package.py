"""Build an arXiv source tree or archive the reproducible research release."""
import argparse
from pathlib import Path
import re
import os
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/"paper"


def sources():
    found=set()
    def collect(path):
        if path in found:return
        if not path.is_file() or path.parent!=PAPER:
            raise ValueError(f"Undeliverable TeX dependency {path}")
        found.add(path)
        if path.suffix!=".tex":return
        content=path.read_text(encoding="utf8")
        for arg in re.findall(r"\\input\{([^}]+)\}",content):
            collect(PAPER/(arg if arg.endswith(".tex") else arg+".tex"))
        for arg in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",content):
            collect(PAPER/arg)
    collect(PAPER/"main.tex")
    tex="\n".join(p.read_text(encoding="utf8") for p in found if p.suffix==".tex")
    citations={key for group in re.findall(r"\\cite\{([^}]+)\}",tex) for key in group.split(",")}
    bib=set(re.findall(r"\\bibitem\{([^}]+)\}",tex))
    assert citations==bib,(citations-bib,bib-citations)
    assert tex.count(r"\documentclass")==1
    return sorted(found)


def archive(path,files):
    temporary=path.with_suffix(".building")
    with ZipFile(temporary,"w",ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
        for file,name in sorted(files,key=lambda pair:pair[1]):
            z.write(file,name)
    with ZipFile(temporary) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))==len(files)
    temporary.replace(path)
    print(path.name,len(files),"members,",path.stat().st_size,"bytes",flush=True)


def source(stage):
    name=PAPER/"crystal-drift-arxiv.zip"
    files=sources()
    archive(name,[(p,p.name) for p in files])
    if stage:
        stage.mkdir(parents=True,exist_ok=False)
        with ZipFile(name) as z:z.extractall(stage)
        assert {p.name:p.read_bytes() for p in stage.iterdir()}=={p.name:p.read_bytes() for p in files}


def editable_figures():
    files = (ROOT/"revision/figures/revision-method.svg", PAPER/"revision-fourier-collision.svg",
             ROOT/"revision/binary/revision-binary-accuracy-time.svg", ROOT/"revision/figures/revision-relaxation.svg")
    extra = [ROOT/"expansion/exp-method.svg", *sorted((ROOT/"expansion/training").glob("exp-*.svg"))]
    panels = ("exp-structures","crystal-examples","unseen-examples")
    extra.extend(ROOT/f"revision/figures/{panel}.svg" for panel in panels)
    bundle = [(p,p.name) for p in (*files,*extra)]
    for rel in ("expansion/physical/render.py","expansion/physical/figure-selection.json","figures-v2/crystal_render.py"):
        bundle.append((ROOT/rel,"sources/"+rel))
    archive(ROOT/"revision/editable-figures.zip",bundle)


def crystal_examples():
    generated = ROOT/"de-novo"
    files = [(p,"generated/"+p.name) for p in sorted((generated/"cifs").glob("*.cif"))]
    files += [(p,"relaxed/"+p.name) for p in sorted((generated/"relaxation/cifs").glob("draw-*-relaxed.cif"))]
    assert len(files) == 64
    for file in ("manifest.csv","all-draws.npz","candidates.npz","summary.json","relaxation/results.csv",
                 "relaxation/run.json","relaxation/README.md","relaxation/selection.json",
                 "elemental-screen/candidate-gaps.csv","elemental-screen/selection.json","timing/summary.json",
                 "figures/render-metadata.json","figures/unseen-examples-metadata.json"):
        files.append((generated/file,file))
    for file in ("crystal-examples.pdf","crystal-examples.png","unseen-examples.pdf","unseen-examples.png",
                 "crystal-render-metadata.json","crystal_render.py"):
        files.append((ROOT/"figures-v2"/file,"figures/"+file))
    for suffix in ("svg","pdf"):
        files.append((ROOT/f"revision/figures/revision-relaxation.{suffix}",f"figures/relaxation-outcomes.{suffix}"))
    for file in ("per-proposal.csv","summary.json"):
        files.append((ROOT/"scaled/novelty"/file,"structural-novelty/"+file))
    for file in ("per-cell.csv","summary.json"):
        files.append((ROOT/"revision/denovo"/file,"generation-metrics/"+file))
    physical = ROOT/"expansion/physical"
    cifs = sorted((physical/"cifs").glob("*.cif"))
    assert len(cifs) == 160
    for p in [*cifs, *(physical/name for name in ("by-arm.csv", "per-candidate.csv", "reference-library.json",
                                         "figure-selection.json", "structures.png", "DESIGN.md", "render.py"))]:
        files.append((p,"multielement/"+p.relative_to(physical).as_posix()))
    for p in (ROOT/"paper/exp-structures.pdf",):
        files.append((p,"multielement/"+p.name))
    local = ROOT/"expansion/physical-joint-local"
    local_cifs = sorted((local/"cifs").glob("*.cif"))
    local_files = [*local_cifs, *(local/n for n in ("per-candidate.csv","by-arm.csv","verification.json","DESIGN.md"))]
    for p in local_files:
        files.append((p,"multielement-local-crystalflow/"+p.relative_to(local).as_posix()))
    archive(ROOT/"generated-crystals.zip",files)


def benchmark_tables():
    patterns=(
        "expansion/training/*.csv","expansion/training/*.csv.gz","expansion/training/*selection.json","expansion/training/gpu/*.csv",
        "expansion/external/csp-s5/summary.json","expansion/external/timing-s5/runtime.json",
        "expansion/external/timing-s5/elapsed-recovery.csv","expansion/volume/summary.json",
        "expansion/hybrid/paired-hybrids.json","expansion/hybrid/*per-seed.csv","expansion/hybrid/confirmation.json",
        "expansion/denovo/panel/*.json","expansion/physical/by-arm.csv","expansion/physical/per-candidate.csv",
        "expansion/physical-joint-local/by-arm.csv","expansion/physical-joint-local/per-candidate.csv",
        "expansion/physical-joint-local/verification.json","expansion/denovo/pools/crystalflow-joint-local.json",
        "expansion/external/*sensitivity*.json","expansion/external/*anneal*.json","expansion/denovo/panel/*local*/*.json",
        "expansion/dft/summary.json","revision/binary/headline-prefix.csv","revision/sensitivity/paired.csv",
        "revision/ablation/*summary.json","revision/denovo/summary.json","expansion/verified-claims.json")
    items={p for pattern in patterns for p in ROOT.glob(pattern)}
    items.add(ROOT/"expansion/CLAIM-AUDIT.md")
    archive(ROOT/"expansion/benchmark-tables.zip",[(p,p.relative_to(ROOT).as_posix()) for p in items])


def research():
    allowed={".json",".npz",".pt",".npy",".md",".bz2",".csv",".py",".sh",".slurm",".png",
             ".svg",".pdf",".tex",".txt",".cif",".yaml",".upf",".in",".out",".sha256",".cpp",".pyx",".tsv",".ckpt",".gz"}
    active_tex=set(sources())
    selected=[]
    for directory, children, names in os.walk(ROOT):
        children[:] = [c for c in children if not c.startswith(".") and c != "__pycache__"]
        for name in names:
            p=Path(directory)/name
            rel=p.relative_to(ROOT)
            if name.startswith(".") or ".part." in name or p.is_symlink():continue
            if rel.as_posix()=="release-sha256.txt":continue  # The outer manifest hashes this archive.
            if rel.parts[0]=="examples" or (p.suffix.lower() not in allowed and not p.name.endswith("LICENSE")):continue
            if p.parent==PAPER and p not in active_tex and p.name not in {
                "main.pdf","submission-metadata.txt","revision-fourier-collision.svg"}:continue
            selected.append((p,"crystal-drift/"+rel.as_posix()))
    raw_name="data/mp20_train.csv.bz2"
    if not (ROOT/raw_name).exists():selected.append((ROOT.parent/"crystal"/raw_name,"crystal-drift/"+raw_name))
    selected.append((PAPER/"crystal-drift-arxiv.zip","crystal-drift/paper/crystal-drift-arxiv.zip"))
    name=ROOT/"crystal-drift-research.zip"
    archive(name,selected)
    with ZipFile(name) as z:
        assert z.read("crystal-drift/paper/main.pdf")== (PAPER/"main.pdf").read_bytes()
        required=("revision/binary/headline-prefix.csv","revision/sensitivity/paired.csv",
                  "revision/ablation/test-summary.json","revision/denovo/per-cell.csv",
                  "revision/full/wbm/test-crosswalk.npz","revision/full/mp20/complete-crosswalk-k20.npz",
                  "revision/full/external/mp20/complete-crosswalk-k20.npz","revision/verified-claims.json",
                  "revision/package-verification.json","paper/revision-fourier-collision.svg",
                  "expansion/verified-claims.json","expansion/CLAIM-AUDIT.md","expansion/training/selection.json",
                  "expansion/volume/corrected-scores.npz","expansion/physical/per-candidate.csv",
                  "expansion/denovo/panel/learned-drift.json","expansion/external/csp-s5/audit.json",
                  "expansion/physical-joint-local/verification.json","expansion/denovo/pools/crystalflow-joint-local.npz",
                  "expansion/denovo/pools/crystalflow-joint-local.json")
        for n in required:assert "crystal-drift/"+n in z.namelist(),n


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind",choices=["source","figures","crystals","tables","research"])
    parser.add_argument("--stage",type=Path,help="Extract into a fresh directory for the two clean compiler passes")
    args=parser.parse_args()
    if args.kind=="source":source(args.stage)
    elif args.kind=="figures":editable_figures()
    elif args.kind=="crystals":crystal_examples()
    elif args.kind=="tables":benchmark_tables()
    else:research()


if __name__=="__main__":
    main()
