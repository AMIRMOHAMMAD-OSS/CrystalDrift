"""Reconcile the external raw-pool totals with prior archives and durable source files."""
import collections
import json
from pathlib import Path

from backup_delta import HERE, RESEARCH

PREFIX="research-scratch/tmp/crytaldrift-v3-20261121/raw/"


def main():
    raw=RESEARCH/"raw"
    live={p.relative_to(raw).as_posix():p.stat().st_size for p in raw.rglob("*") if p.is_file()}
    result={"live":{"files":len(live),"bytes":sum(live.values()),
                     "extensions":collections.Counter(Path(p).suffix for p in live)}}
    for name in ("recovery-base","rebuilt"):
        stage=RESEARCH/("drive-backup-"+name)
        archived={}
        for line in (stage/"FILES.ndjson").open():
            row=json.loads(line)
            if row["path"].startswith(PREFIX) and "bytes" in row:archived[row["path"].removeprefix(PREFIX)]=row
        missing={p:row for p,row in archived.items() if p not in live}
        result[name]=dict(files=len(archived),bytes=sum(x["bytes"] for x in archived.values()),missing=missing)
    endpoint=json.loads((HERE/"drive-endpoint-integrity.json").read_text())
    result["endpoint"]={"files":endpoint["raw_files"],"bytes":endpoint["raw_bytes"]}
    print(json.dumps({k:{kk:(len(vv) if kk=="missing" else vv) for kk,vv in v.items()} for k,v in result.items()},indent=2))
    (RESEARCH/"external-raw-inventory.json").write_text(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
