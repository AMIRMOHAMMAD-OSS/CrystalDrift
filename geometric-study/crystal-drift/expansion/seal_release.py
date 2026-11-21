"""Bind the evidence and editable archives to the verified manuscript version."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("paper/main.pdf","paper/crystal-drift-arxiv.zip","crystal-drift-research.zip",
         "revision/editable-figures.zip","generated-crystals.zip","expansion/benchmark-tables.zip")

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def main():
    check=json.loads((ROOT/"revision/package-verification.json").read_text())
    hashes={name:sha(ROOT/name) for name in NAMES}
    assert hashes[NAMES[0]]==check["pdf_sha256"] and hashes[NAMES[1]]==check["source_zip_sha256"]
    with ZipFile(ROOT/NAMES[2]) as research:
        for key in (NAMES[0],NAMES[1],"revision/package-verification.json","revision/submission_check.py",
                    "expansion/seal_release.py","expansion/verified-claims.json","expansion/denovo/manifest.json",
                    "expansion/denovo/panel/audit.json","expansion/physical-joint-local/verification.json",
                    "expansion/external/step-annealing-factorial.json","expansion/CLAIM-AUDIT.md"):
            body=research.read("crystal-drift/"+key)
            assert hashlib.sha256(body).hexdigest()==sha(ROOT/key), key
    with ZipFile(ROOT/NAMES[3]) as figures:
        assert {"exp-method.svg","exp-gpu-curves.svg","exp-training-curves.svg","revision-fourier-collision.svg"}<=set(figures.namelist())
        for stem in ("exp-structures","crystal-examples","unseen-examples"):
            assert figures.read(stem+".svg")==(ROOT/"revision/figures"/(stem+".svg")).read_bytes()
        for name in ("exp-gpu-curves.svg","exp-training-curves.svg"):
            assert figures.read(name)==(ROOT/"expansion/training"/name).read_bytes()
    with ZipFile(ROOT/NAMES[4]) as crystals:
        local=ROOT/"expansion/physical-joint-local"
        local_cifs={p.name for p in (local/"cifs").glob("*.cif")}
        included={Path(n).name for n in crystals.namelist() if n.startswith("multielement-local-crystalflow/") and n.endswith(".cif")}
        assert included==local_cifs and len(local_cifs)>0
        assert sum(n.endswith(".cif") for n in crystals.namelist())==224+len(local_cifs)
        assert crystals.read("multielement-local-crystalflow/verification.json")==(local/"verification.json").read_bytes()
        assert crystals.read("multielement/exp-structures.pdf")==(ROOT/"paper/exp-structures.pdf").read_bytes()
    with ZipFile(ROOT/NAMES[5]) as tables:
        assert tables.read("expansion/verified-claims.json")==(ROOT/"expansion/verified-claims.json").read_bytes()
    (ROOT/"release-sha256.txt").write_text("".join(f"{v}  {k}\n" for k,v in hashes.items()))
    print("Sealed current PDF and five review archives.")

if __name__=="__main__":
    main()
