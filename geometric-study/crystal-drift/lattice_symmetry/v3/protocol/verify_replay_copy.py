"""Read-only check that previous temporary replay files are reproducible from preserved archives."""
import hashlib
import json
from pathlib import Path
import zlib
from zipfile import ZipFile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
SCRATCH=ROOT.parents[1]/"tmp/crytaldrift-evidence-check-01"
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(4<<20),b""):h.update(chunk)
    return h.hexdigest()
def main():
    releases=[ROOT/"crystal-drift-research.zip",ROOT/"lattice_symmetry/v2/release/CrytalDrift-periodic-evidence.zip"]
    receipt=json.loads((ROOT/"lattice_symmetry/v2/release/archive-replay.json").read_text())
    assert [sha(s) for s in releases]==[receipt["fixed_baseline_sha256"],receipt["overlay_sha256"]]
    manifest={}
    for s in releases:
        with ZipFile(s) as f:
            manifest.update({m.filename:(m.file_size,m.CRC) for m in f.infolist() if not m.is_dir()})
    bad=[];extra=[];count=0
    for p in SCRATCH.rglob("*"):
        if not p.is_file():continue
        key=p.relative_to(SCRATCH).as_posix()
        if key in manifest:
            size,crc=manifest[key];h=0
            with p.open("rb") as f:
                for chunk in iter(lambda:f.read(4<<20),b""):h=zlib.crc32(chunk,h)
            if p.stat().st_size!=size or h!=crc:bad.append(key)
            count+=1
        elif key=="crystal-drift-research.zip":assert sha(p)==receipt["fixed_baseline_sha256"]
        elif key=="archive-replay.json":assert json.loads(p.read_text())==receipt
        else:extra.append(key)
    report=dict(verified_archived_files=count,mismatched=bad,extra=extra,all_archive_members_present=count==len(manifest),
         backup_archive_sha=receipt["fixed_baseline_sha256"],overlay_sha=receipt["overlay_sha256"])
    (HERE/"temporary-replay-check.json").write_text(json.dumps(report,indent=2))
    print(count,len(bad),len(extra),count==len(manifest))
if __name__=="__main__":main()
