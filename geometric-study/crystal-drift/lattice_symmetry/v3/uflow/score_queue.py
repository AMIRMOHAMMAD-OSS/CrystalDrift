"""Resume small matcher batches, round-robin between native prior and learned proposal pools."""
import argparse
import faulthandler
import hashlib
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections import deque
from pathlib import Path

from evaluate import HERE, evaluate, score
import numpy as np

def score_slice(job):
    source,split,out,span=job
    folder=HERE/"diagnostics"
    folder.mkdir(exist_ok=True)
    status=folder/f"worker-{os.getpid()}.json"
    status.write_text(json.dumps(dict(source=str(source),span=span,start=time.time())))
    with (folder/f"worker-{os.getpid()}.txt").open("a") as fp:
        faulthandler.dump_traceback_later(150,file=fp)
        try:
            score(source,split,out,row_span=span)
        finally:
            faulthandler.cancel_dump_traceback_later()
            status.unlink(missing_ok=True)

def materialize(path,split,dest,parts):
    if dest.exists():
        return True
    if not all(p.exists() for p in parts):
        return False
    full={};fields=("ids","rms","strict_fit","log_volume_ratio","error")
    values={key:[] for key in fields};totals={key:0 for key in ("seconds","forwarded","public_checks")}
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    for part in parts:
        with np.load(part) as d:
            assert str(d["source_sha"])==sha and str(d["split"])==split
            for k in fields:values[k].append(d[k])
            for k in totals:totals[k]+=d[k].item()
    full={k:np.concatenate(rows) for k,rows in values.items()}
    with np.load(path) as d:
        np.testing.assert_array_equal(full["ids"],d["ids"])
    temp=dest.with_suffix(".building.npz")
    np.savez_compressed(temp,**full,**totals,source_sha=sha,split=split,public_all=False,pymatgen="2024.4.13")
    temp.replace(dest)
    return True

def pending_work(tags,panels):
    splits={"confirm":"val","family":"train","shared":"train","test":"test"}
    tasks={tag:deque() for tag in tags}
    complete=True
    for panel in panels:
        for tag in tags:
            source=HERE/"predictions"/tag/"frozen"/panel
            dest=HERE/"scores"/tag/"frozen"/panel
            dest.mkdir(exist_ok=True,parents=True)
            if (dest/"outcomes.npz").exists():continue
            all_scored=(source/"ready.json").is_file()
            for file in sorted(source.glob("*.npz")):
                if ".part" in file.name or (dest/file.name).exists():continue
                lo,hi=map(int,file.stem.split("-"))
                folder=dest/"parts"/file.stem
                folder.mkdir(exist_ok=True,parents=True)
                span=[(i,min(i+16,hi-lo)) for i in range(0,hi-lo,16)]
                parts=[folder/f"{a:03}-{b:03}.npz" for a,b in span]
                if materialize(file,splits[panel],dest/file.name,parts):continue
                all_scored=False
                tasks[tag].extend((file,splits[panel],part,pair) for pair,part in zip(span,parts) if not part.exists())
            if all_scored:
                evaluate(tag,"frozen",1,[panel])
            else:
                complete=False
    return tasks,complete

def drain(tags,panels,workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        active={}
        turns=deque(tags)
        while True:
            tasks,complete=pending_work(tags,panels)
            if complete:return
            running={job[2] for _,job in active.values()}
            tasks={tag:deque(job for job in jobs if job[2] not in running) for tag,jobs in tasks.items()}
            if not any(tasks.values()) and not active:
                time.sleep(8);continue
            last_refresh=time.monotonic()
            while any(tasks.values()) or active:
                while len(active)<workers and any(tasks.values()):
                    # Choose the least represented ready pipeline among live processes.
                    min_live=min(sum(a==t for a,_ in active.values()) for t in tags if tasks[t])
                    while not tasks[turns[0]] or sum(a==turns[0] for a,_ in active.values())!=min_live:
                        turns.rotate(-1)
                    tag=turns[0];turns.rotate(-1)
                    job=tasks[tag].popleft()
                    active[pool.submit(score_slice,job)]=(tag,job)
                if not active:break
                done,_=wait(active,timeout=15,return_when=FIRST_COMPLETED)
                for f in done:
                    f.result()
                    active.pop(f)
                if time.monotonic()-last_refresh>=120:break

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("tags",nargs="+")
    p.add_argument("--panels",nargs="+",default=["confirm","family","shared","test"])
    p.add_argument("--workers",type=int,default=2)
    args=p.parse_args();drain(args.tags,args.panels,args.workers)
