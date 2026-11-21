"""Verify numeric parity, then score a balanced panel of frozen methods with shared-memory workers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import time
import numpy as np
from score import score, truth
from shards import assemble, registered
from storage import HERE, pool

def execute(job):
    path, panel, dest = job
    if not dest.exists():
        score(path, panel, dest)
    return str(dest)

def parity():
    root = Path(os.environ.get("CRYTALDRIFT_SCRATCH", HERE/"parity"))
    root.mkdir(exist_ok=True,parents=True)
    records=[]
    for method in ("prior","angular-small-11","contacts-small-11"):
        name="00000-00096.npz"
        old=HERE/"scores/family"/method/name
        new=root/("linux-"+method+".npz")
        score(pool("family",method)/name,"train",new)
        with np.load(old) as a,np.load(new) as b:
            for key in ("ids","error","strict_fit"):
                np.testing.assert_array_equal(a[key], b[key])
            np.testing.assert_array_equal(np.isfinite(a["rms"]),np.isfinite(b["rms"]))
            np.testing.assert_allclose(a["log_volume_ratio"],b["log_volume_ratio"],atol=1e-10,rtol=0)
            delta=float(np.nanmax(np.abs(a["rms"]-b["rms"])))
            np.testing.assert_allclose(a["rms"],b["rms"],atol=1e-7,rtol=0,equal_nan=True)
            records.append({"method":method,"shape":list(b["rms"].shape),"max_rms_error":delta,
                            "source_sha":str(a["source_sha"]),"matches":int(np.isfinite(b["rms"]).sum())})
    (HERE/"matcher-platform-parity.json").write_text(json.dumps(records,indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("panel");p.add_argument("matrices",nargs="+",type=Path)
    p.add_argument("--workers",type=int,default=4);p.add_argument("--no-wait",action="store_true")
    p.add_argument("--parity",action="store_true");p.add_argument("--limit",type=int)
    args=p.parse_args()
    if args.parity: parity()
    split={"test":"test","dev":"val","confirm":"val","shared":"train","family":"train"}[args.panel]
    n=len(registered(args.panel))
    tags=[]
    for path in args.matrices:
        spec=json.loads(path.read_text());assert spec["frozen"]
        tags.extend(row["tag"] for row in spec["methods"])
    tags=list(dict.fromkeys(tags))
    truth(split)
    method="fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
    ended=set()
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context(method)) as executor:
        while True:
            jobs=[]
            for tag in tags:
                if tag in ended:continue
                folder=pool(args.panel,tag)
                dst=HERE/"scores"/args.panel/tag
                dst.mkdir(exist_ok=True,parents=True)
                if (dst/"outcomes.npz").is_file():
                    ended.add(tag);continue
                for source in folder.glob("?????-?????.npz"):
                    if args.limit and int(source.stem.split("-")[0])>=args.limit:continue
                    if not (dst/source.name).exists():jobs.append((source,split,dst/source.name))
                if not args.limit and not any(x[-1].parent==dst for x in jobs) and assemble(folder,args.panel,tag,split,n):
                    ended.add(tag)
            if not jobs:
                if len(ended)==len(tags) or args.no_wait or args.limit:return
                time.sleep(10);continue
            jobs.sort(key=lambda job:(job[0].name,tags.index(job[0].parent.name)))
            for _ in executor.map(execute,jobs[:max(24,args.workers*12)]):pass

if __name__=="__main__":main()
