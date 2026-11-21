"""Measure how many bytes of the completed scientific archive need new local ZIP chunks."""
import hashlib
import json
import zipfile
from pathlib import Path

from backup import evidence_blocks
from backup_delta import HERE, RESEARCH


def main():
    archive=Path("/home/amir/crytaldrift-v3-evidence/release/CrytalDrift-v3-research-evidence.zip")
    known=set()
    for volume in sorted((RESEARCH/"drive-backup-rebuilt").glob("pack-*.zip")):
        with zipfile.ZipFile(volume) as z:known.update(z.namelist())
    h=hashlib.sha256()
    seen=set()
    n=local=repeat=total=0
    with archive.open("rb") as f:
        for block in evidence_blocks(archive,f,archive.stat().st_size):
            key=hashlib.sha256(block).hexdigest()
            h.update(block)
            total+=len(block)
            if key in seen:repeat+=len(block)
            elif key in known:local+=len(block)
            seen.add(key)
            n+=1
    source=json.loads(archive.with_suffix(".receipt.json").read_text())
    if h.hexdigest()!=source["sha256"] or total!=source["bytes"]:raise ValueError("Archive-aware segmentation changed the scientific evidence")
    report=dict(science_zip_sha256=h.hexdigest(),bytes=total,blocks=n,distinct_blocks=len(seen),
                bytes_reused_from_rebuilt_local_chunks=local,repeated_bytes_within_evidence_zip=repeat,
                upper_bound_new_chunk_payload_bytes_before_outer_compression=total-local-repeat)
    target=HERE/"evidence-archive-chunk-plan.json"
    if target.exists():
        if json.loads(target.read_text())!=report:raise ValueError("Different segmentation estimate")
    else:
        with target.open("x") as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
