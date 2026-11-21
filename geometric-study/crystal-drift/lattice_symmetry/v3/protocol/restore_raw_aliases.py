"""Materialize the three verified external output aliases missing from the pointer registry."""
import json
import os
from pathlib import Path

from backup import digest
from backup_delta import HERE,RESEARCH


def main():
    evaluation=HERE.parent/"evaluation"
    output=[]
    for panel in ("family","shared","test"):
        target=RESEARCH/"raw"/panel/"meanflow-medium-long"
        alias=evaluation/"aliases"/panel/"meanflow-medium-long.json"
        info=json.loads(alias.read_text())
        if (info["panel"],info["tag"],info["identical_to"])!=(panel,target.name,"meanflow-medium-strict"):
            raise ValueError("Unregistered alias identity")
        source=target.with_name(info["identical_to"])
        srcmeta=json.loads((source/"settings.json").read_text())
        if srcmeta["signature"]!=info["signature"] or json.loads(info["signature"])["weight"]!=info["weight_sha256"]:
            raise ValueError("Source model differs")
        files=sorted(source.glob("?????-?????.npz"))
        if sum(int(p.stem[6:])-int(p.stem[:5]) for p in files)!=info["count"]:raise ValueError("Incomplete candidate coverage")
        target.mkdir(exist_ok=True,parents=True)
        for src in files:
            dst=target/src.name
            if dst.exists():
                if not dst.samefile(src) and digest(src)!=digest(dst):raise ValueError(dst)
            else:os.link(src,dst)
        meta=target/"settings.json"
        if meta.exists():
            if meta.read_bytes()!=alias.read_bytes():raise ValueError(meta)
        else:
            with meta.open("xb") as f:f.write(alias.read_bytes())
        output.append(dict(panel=panel,tag=target.name,source=source.name,source_weight=info["weight_sha256"],
                           shards=len(files),targets=info["count"],settings_sha256=digest(meta)))
    path=HERE/"raw-alias-restoration.json"
    if path.exists():
        if json.loads(path.read_text())!=output:raise ValueError("Aliased model settings changed")
    else:
        with path.open("x") as f:json.dump(output,f,indent=2)
    print(json.dumps(output),flush=True)


if __name__=="__main__":
    main()
