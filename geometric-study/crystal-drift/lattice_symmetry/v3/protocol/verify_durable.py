"""Verify an incremental edition against an independently restored initial edition.

Initial files with identical content/chunk tuples compose directly with the earlier full
byte-verification receipt. Every other file is reconstructed from actual ZIP members.
Local immutable volumes are preferred; a missing inherited member is read through the
read-only Drive mount. Nothing is extracted or written to that mount.
"""
import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

from backup import digest
from backup_delta import HERE, RESEARCH
from verify_endpoint import files

MOUNT = Path("/home/amir/crytaldrift-v3-evidence/drive-readonly/My Drive/CrystalDrift/CrytalDrift full research backup 2026-11-21")


def identity(row):
    if "sha256" in row:
        return (row["bytes"], row["sha256"], tuple(map(tuple,row["chunks"])))
    return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edition",choices=("rebuilt","final"))
    parser.add_argument("--plan",action="store_true")
    a=parser.parse_args()
    stage=RESEARCH/("drive-backup-"+a.edition)
    base=RESEARCH/"drive-backup-recovery-base"
    original=json.loads((HERE/"drive-catalog.json").read_text())
    proof=json.loads((HERE/"initial-backup-verification.json").read_text())
    gate=json.loads((HERE/"initial-drive-gate.json").read_text())
    if digest(HERE/"drive-catalog.json")!=gate["backup_index_sha256"]:raise ValueError("Initial catalog identity")
    if proof["successful_exit"] or proof["observed_output"]!=f"Verified reconstructed files {original['files']}":raise ValueError("Initial restore was not verified")
    if original["logical_bytes"]!=proof["logical_bytes"] or digest(base/"FILES.ndjson")!=original["volumes"][-1]["sha256"]:raise ValueError("Initial file inventory differs")
    old=files(base/"FILES.ndjson")
    initial={identity(r) for r in old.values() if "sha256" in r}
    catalog=json.loads((stage/"BACKUP.json").read_text())
    rows=files(stage/"FILES.ndjson")
    items=[r for r in rows.values() if not r["path"].endswith("/")]
    if len(items)!=catalog["files"] or sum(r.get("bytes",0) for r in items)!=catalog["logical_bytes"]:raise ValueError("Edition totals")
    by_name={r["name"]:r for r in catalog["volumes"]}
    if len(by_name)!=len(catalog["volumes"]) or digest(stage/"FILES.ndjson")!=by_name["FILES.ndjson"]["sha256"]:raise ValueError("Edition inventory identity")
    todo=[r for r in rows.values() if "sha256" in r and identity(r) not in initial]
    borrowed={v for r in todo for v,k in r["chunks"] if not (stage/v).is_file() and not (base/v).is_file()}
    chunks={v for r in rows.values() for v,k in r.get("chunks",[])}
    if not chunks.issubset(by_name):raise ValueError("Unknown pack in manifest")
    updated=sum(r["bytes"] for r in todo)
    print("Changed byte streams",len(todo),updated,"missing historical packs needed",len(borrowed),sorted(borrowed),flush=True)
    if a.plan:return
    # Verify local immutable ZIP payloads and attest to published, byte-read original identities.
    mounted_used=set()
    current=[]
    unreferenced_pending=[]
    for name,v in by_name.items():
        if not name.startswith("pack-"):continue
        local=stage/name
        if local.exists():
            if local.stat().st_size!=v["bytes"] or digest(local)!=v["sha256"]:raise ValueError("Local pack digest "+name)
            current.append(name)
        else:
            attestation=HERE/"drive-verified"/(name+".json")
            if not attestation.is_file() and name not in chunks:
                upload=json.loads((HERE/"drive-receipts"/(name+".json")).read_text())
                if upload.get("name")!=name:raise ValueError("Unknown historical published object "+name)
                unreferenced_pending.append(name)
                continue
            remote=json.loads(attestation.read_text())
            if not remote["remote_readback"] or remote["sha256"]!=v["sha256"] or remote["bytes"]!=v["bytes"]:raise ValueError("Historical pack digest "+name)
    opened={}
    def read(vol,key):
        if vol not in opened:
            p=stage/vol
            if not p.is_file():p=base/vol
            if not p.is_file():
                p=MOUNT/vol
                mounted_used.add(vol)
            opened[vol]=zipfile.ZipFile(p)
        return opened[vol].read(key)
    try:
        for i,row in enumerate(todo):
            h=hashlib.sha256()
            n=0
            for vol,key in row["chunks"]:
                for attempt in range(4):
                    try:
                        block=read(vol,key)
                        break
                    except (OSError,zipfile.BadZipFile):
                        if vol in opened:opened.pop(vol).close()
                        if attempt==3:raise
                        time.sleep(1+attempt)
                if hashlib.sha256(block).hexdigest()!=key:raise ValueError("Changed chunk "+vol+" "+key)
                h.update(block)
                n+=len(block)
            if n!=row["bytes"] or h.hexdigest()!=row["sha256"]:raise ValueError("Changed byte stream "+row["path"])
            if i and i%2000==0:print("Reconstructed",i,"/",len(todo),flush=True)
    finally:
        for z in opened.values():z.close()
    report=dict(edition=a.edition,catalog_sha256=digest(stage/"BACKUP.json"),files_manifest_sha256=digest(stage/"FILES.ndjson"),
        files_and_links=catalog["files"],logical_bytes=catalog["logical_bytes"],regular_files=sum("sha256" in r for r in rows.values()),
        identical_previously_reconstructed_files=sum("sha256" in r and identity(r) in initial for r in rows.values()),
        physically_reconstructed_files_this_pass=len(todo),physically_reconstructed_bytes_this_pass=updated,
        inherited_volume_objects_verified_against_prior_remote_bytes=sum(v.startswith("pack-") and v not in current and v not in unreferenced_pending for v in by_name),
        unreferenced_volume_objects_awaiting_remote_readback=unreferenced_pending,
        local_volume_objects_rehashed=len(current),borrowed_volume_member_reads=sorted(mounted_used),
        initial_restore=proof,initial_catalog_sha256=gate["backup_index_sha256"])
    dst=HERE/(a.edition+"-cumulative-reconstruction.json")
    if dst.exists():
        if json.loads(dst.read_text())!=report:raise ValueError("Conflicting stored reconstruction proof")
    else:
        with dst.open("x") as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
