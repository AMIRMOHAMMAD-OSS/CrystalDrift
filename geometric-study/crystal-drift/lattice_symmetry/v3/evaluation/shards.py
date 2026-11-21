"""Run pinned scoring on finished candidate shards, and aggregate only verified ordered outcomes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
from time import sleep
import numpy as np
HERE=Path(__file__).resolve().parent
from score import score
from storage import pool

def registered(panel):
    if panel.startswith("arrangement"):
        data=json.loads((HERE.parent/"geometry/audits/arrangement-frozen.json").read_text())
        return data["covered_ids" if panel.endswith("control") else "uncovered_ids"]
    return json.loads((HERE.parent/"protocol/splits.json").read_text())[panel+"_ids"]

def worker(job):
    score(*job)

def assemble(path,panel,tag,split,n):
    target=HERE/"scores"/panel/tag;source=HERE/"raw"/panel/tag
    rows=[];end=0
    for p in sorted(target.glob("?????-?????.npz")):
        lo,hi=map(int,p.stem.split("-"))
        if lo!=end:break
        with np.load(p) as a:
            assert len(a["ids"])==hi-lo
            rows.append({k:a[k] for k in ("ids","rms","strict_fit","error","log_volume_ratio")})
        end=hi
    if end!=n:return False
    data={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
    if panel!="clock":assert data["ids"].tolist()==registered(panel)
    temporary=target/f"outcomes-{os.getpid()}.pending"
    with temporary.open("wb") as fp:np.savez_compressed(fp,**data)
    temporary.replace(target/"outcomes.npz")
    print("Completed",panel,tag,end,flush=True)
    return True

def main():
    ap=argparse.ArgumentParser();ap.add_argument("panel");ap.add_argument("tags",nargs="+")
    ap.add_argument("--workers",type=int,default=2);ap.add_argument("--wait",action="store_true")
    a=ap.parse_args()
    split={"dev":"val","confirm":"val","family":"train","shared":"train","test":"test","clock":"test",
           "arrangement":"val","arrangement-control":"val"}[a.panel]
    n=64 if a.panel=="clock" else len(registered(a.panel))
    with (ProcessPoolExecutor(a.workers) if a.workers>1 else nullcontext()) as ex:
        while True:
            jobs=[];all_done=True
            for tag in a.tags:
                folder=pool(a.panel,tag)
                dst=HERE/"scores"/a.panel/tag;dst.mkdir(exist_ok=True,parents=True)
                if (dst/"outcomes.npz").exists():continue
                if assemble(folder,a.panel,tag,split,n):continue
                all_done=False
                jobs.extend((p,split,dst/p.name) for p in sorted(folder.glob("?????-?????.npz")) if not (dst/p.name).exists())
            if jobs:
                for _ in (ex.map(worker,jobs) if ex else map(worker,jobs)):pass
                continue
            if all_done or not a.wait:break
            sleep(8)
if __name__=="__main__":main()
