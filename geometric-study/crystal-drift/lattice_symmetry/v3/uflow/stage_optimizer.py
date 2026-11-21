"""Move only a completed fit's verified restart state to local scratch; EMA files stay with the study."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as handle:
        while chunk:=handle.read(1024*1024):h.update(chunk)
    return h.hexdigest()

def stage(tag,scratch):
    fit=(HERE/"fits"/tag).resolve()
    assert fit.parent==HERE/"fits" and (fit/"complete.json").is_file()
    source=fit/"resume.pt"
    target=(scratch/tag/"resume.pt").resolve()
    assert target.parent.parent==scratch
    if not source.is_file():
        meta=json.loads((fit/"resume-location.json").read_text())
        assert digest(target)==meta["sha256"]
        return
    target.parent.mkdir(exist_ok=True,parents=True)
    wanted=digest(source)
    if target.exists():
        assert digest(target)==wanted
    else:
        temp=target.with_suffix(".part")
        shutil.copyfile(source,temp)
        assert digest(temp)==wanted and ZipFile(temp).testzip() is None
        temp.replace(target)
    receipt=dict(sha256=wanted,bytes=source.stat().st_size,source="resume.pt",scratch_file=str(target),
                 role="restart state for a completed fit; all EMA checkpoints remain alongside this receipt")
    (fit/"resume-location.json").write_text(json.dumps(receipt,indent=2)+"\n")
    source.unlink()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("scratch",type=Path);p.add_argument("tags",nargs="+")
    a=p.parse_args()
    root=a.scratch.resolve()
    assert root.name=="crytaldrift-v3-uflow-optimizer" and root.parent==Path("/tmp")
    for tag in a.tags:stage(tag,root)
