"""Independently read the unchanged physical byte streams after recovery.

Use the current files or the two retained previous scientific ZIPs. Each block must
match the exact SHA listed in the initial ZIP reconstruction, in the same order.
"""
import argparse
import collections
import hashlib
import json
import time
import zipfile
from pathlib import Path

from backup import CHUNK, digest
from backup_delta import HERE, RESEARCH, source_roots
from verify_durable import MOUNT, identity
from verify_endpoint import files


def valid(stream, row):
    total=0
    sha=hashlib.sha256()
    for _,key in row["chunks"]:
        block=stream.read(CHUNK)
        if hashlib.sha256(block).hexdigest()!=key:return False
        total+=len(block)
        sha.update(block)
    return not stream.read(1) and total==row["bytes"] and sha.hexdigest()==row["sha256"]


def old_member(row):
    cache=RESEARCH/"inherited-reconstruction-cache"
    cache.mkdir(exist_ok=True)
    target=cache/row["sha256"]
    if target.exists():
        with target.open("rb") as f:
            if not valid(f,row):raise ValueError("Changed recovered historical object")
        return "verified_historical_ZIP_member"
    temporary=target.with_suffix(".building")
    with temporary.open("wb") as dst:
        for volume,key in row["chunks"]:
            receipt=json.loads((HERE/"drive-verified"/(volume+".json")).read_text())
            if not receipt["remote_readback"]:raise ValueError("Unverified historical archive "+volume)
            for attempt in range(4):
                try:
                    with zipfile.ZipFile(MOUNT/volume) as z:block=z.read(key)
                    if hashlib.sha256(block).hexdigest()!=key:raise ValueError("Historical chunk hash mismatch")
                    dst.write(block)
                    break
                except (OSError,zipfile.BadZipFile):
                    if attempt==3:raise
                    time.sleep(attempt+1)
    with temporary.open("rb") as f:
        if not valid(f,row):raise ValueError("Historical file hash mismatch")
    temporary.replace(target)
    return "verified_historical_ZIP_member"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edition",choices=("rebuilt","final"),default="rebuilt")
    edition=parser.parse_args().edition
    base=RESEARCH/"drive-backup-recovery-base"
    stage=RESEARCH/("drive-backup-"+edition)
    roots=source_roots([])
    base_rows=files(base/"FILES.ndjson")
    old={identity(r) for r in base_rows.values() if "sha256" in r}
    locked=files(stage/"FILES.ndjson")
    refs=collections.defaultdict(list)
    for r in list(locked.values())+list(base_rows.values()):
        k=identity(r)
        if k is not None and k in old:refs[k].append(r["path"])
    needed={}
    for r in locked.values():
        k=identity(r)
        if k is not None and k in old:needed.setdefault(k,r)
    repo=HERE.parents[2]
    previous_replay="tmp/crytaldrift-evidence-check-01/"
    archived={}
    archives=[]
    for path in (repo/"crystal-drift-research.zip",repo/"lattice_symmetry/v2/release/CrytalDrift-periodic-evidence.zip"):
        zip_=zipfile.ZipFile(path)
        archives.append(zip_)
        for entry in zip_.infolist():
            if not entry.is_dir():archived[entry.filename]=zip_
    def physical(logical):
        for prefix,root in roots.items():
            if prefix==logical:return root
            if logical.startswith(prefix+"/"):return root/logical[len(prefix)+1:]
        return None
    def local(logical, row):
        if logical.startswith(previous_replay):
            short=logical[len(previous_replay):]
            if short in archived:
                with archived[short].open(short) as f:
                    if valid(f,row):return "earlier_scientific_ZIP"
            if short=="crystal-drift-research.zip":path=repo/short
            elif short=="archive-replay.json":path=repo/"lattice_symmetry/v2/release"/short
            elif short.startswith("crystal-drift/"):path=repo/short[len("crystal-drift/"):]
            else:return None
        else:path=physical(logical)
        if path is not None and path.is_file() and path.stat().st_size==row["bytes"]:
            with path.open("rb") as stream:
                if valid(stream,row):return "retained_regular_file"
        return None
    methods=collections.Counter()
    failed={}
    fingerprint=hashlib.sha256()
    checked={}
    log=RESEARCH/"independent-inherited-payloads.ndjson"
    if log.is_file():
        for line in log.open():
            try:key,method=json.loads(line)
            except ValueError:continue
            checked[key]=method
    with log.open("a",buffering=1) as journal:
        for i,(key,row) in enumerate(needed.items()):
            sig=hashlib.sha256(json.dumps(key,separators=(",",":")).encode()).hexdigest()
            if sig in checked:method=checked[sig]
            else:
                method=None
                for name in dict.fromkeys(refs[key]):
                    try:method=local(name,row)
                    except OSError:pass
                    if method:break
                if not method:
                    try:method=old_member(row)
                    except (OSError,zipfile.BadZipFile) as exc:print("Historical read unavailable",row["path"],str(exc),flush=True)
                if not method:
                    failed[row["path"]]=row
                    continue
                journal.write(json.dumps((sig,method),separators=(",",":"))+"\n")
            fingerprint.update((sig+"\n").encode())
            methods[method]+=1
            if i and i%2500==0:print("Reconstructed unique old payloads",i,"/",len(needed),"unavailable",len(failed),flush=True)
    for zip_ in archives:zip_.close()
    same=sum(identity(r) in needed for r in locked.values() if "sha256" in r)
    report=dict(edition=edition,regular_file_references=same,unique_payloads=len(needed),
                unique_bytes=sum(k[0] for k in needed),verified_unique_payloads=sum(methods.values()),
                source_counts=dict(methods),success_identity_stream_sha256=fingerprint.hexdigest(),
                original_files_inventory_sha256=digest(base/"FILES.ndjson"),unresolved=failed)
    report_path=RESEARCH/("independent-inherited-"+edition+"-results.json")
    report_path.write_text(json.dumps(report,indent=2))
    if not failed:
        local=HERE/(edition+"-physical-inherited-replay.json")
        if local.exists():
            if json.loads(local.read_text())!=report:raise ValueError("Conflicting physical replay")
        else:
            with local.open("x") as f:json.dump(report,f,indent=2)
    print("Independent physical source replay coverage",same,len(needed),dict(methods),"pending",len(failed),flush=True)


if __name__=="__main__":
    main()
