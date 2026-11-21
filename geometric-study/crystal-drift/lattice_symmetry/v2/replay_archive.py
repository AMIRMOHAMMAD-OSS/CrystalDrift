"""Independently replay the study audit from freshly extracted evidence archives."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from zipfile import ZipFile


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda:fp.read(4<<20),b""):h.update(chunk)
    return h.hexdigest()


def extract(src,dest):
    with ZipFile(src) as z:
        for member in z.infolist():
            name=PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts or not name.parts or name.parts[0]!="crystal-drift":
                raise ValueError(member.filename)
            z.extract(member,dest)
        return len(z.namelist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline",type=Path)
    parser.add_argument("overlay",type=Path)
    parser.add_argument("fresh_directory",type=Path)
    args=parser.parse_args()
    dest=args.fresh_directory.resolve()
    dest.mkdir(parents=True,exist_ok=False)
    baseline=args.baseline.resolve()
    assert sha(baseline)=="644c0456a4f015edb1df00ab766246a1c4922f692bb9329368cd7451c5ec84ec"
    try:
        os.link(baseline,dest/baseline.name)
    except OSError:
        shutil.copyfile(baseline,dest/baseline.name)
    nbase=extract(baseline,dest)
    nnew=extract(args.overlay,dest)
    root=dest/"crystal-drift"
    study=root/"lattice_symmetry/v2"
    inv=json.loads((study/"release/evidence-inventory.json").read_text(encoding="utf8"))
    assert len(inv)+1==nnew
    for name,rec in inv.items():
        file=dest/name
        assert file.stat().st_size==rec["bytes"] and sha(file)==rec["sha256"],name
    audit=study/"audit.json"
    expected=sha(audit)
    subprocess.run([sys.executable,str(study/"verify_study.py")],cwd=root,check=True)
    assert sha(audit)==expected
    receipt=dict(baseline_members=nbase,overlay_members=nnew,audit_sha256=expected,
                 fixed_baseline_sha256=sha(baseline),overlay_sha256=sha(args.overlay),verified=True)
    (dest/"archive-replay.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf8")
    print(receipt,flush=True)


if __name__=="__main__":main()
