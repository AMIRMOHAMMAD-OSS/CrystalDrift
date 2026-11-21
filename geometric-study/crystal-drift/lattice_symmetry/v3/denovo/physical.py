"""Relax prospectively hash-selected raw indices and account for the competing phases explicitly."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];STORE=HERE/"screen";OLD=ROOT/"expansion/physical"
sys.path[:0]=[str(OLD),str(ROOT)]
import run, summarize
import numpy as np

INDICES=sorted(range(10000),key=lambda i:hashlib.sha256(("v3-relax/"+str(i)).encode()).digest())[:8]

def initialize():
    run.HERE=STORE
    run.initialize()

def main():
    ap=argparse.ArgumentParser();ap.add_argument("arms",nargs="+");ap.add_argument("--workers",type=int,default=1);a=ap.parse_args()
    run.HERE=summarize.HERE=STORE
    for s in ("inputs","records","cifs"):(STORE/s).mkdir(parents=True,exist_ok=True)
    phase,train=run.read_refs(); used=set();sources={}
    for arm in a.arms:
        with np.load(HERE/f"pools/{arm}.npz") as d:
            for ix in INDICES:
                z=d["z"][ix];e=set(z[z>0]);key=run.reduced(z)
                refs=sorted(v[1]["id"] for k,v in phase.items() if all(x in e for x,_ in k))
                name=f"{arm}-{ix+1:05}";used.update(refs)
                sources[name]=dict(id=name,role="candidate",arm=arm,index=ix,source_id=str(d["ids"][ix]),
                    z=z[z>0].tolist(),k=d["k"][ix].tolist(),f=d["f"][ix,z>0].tolist(),seen=key in train,
                    reference_ids=refs,pool=f"lattice_symmetry/v3/denovo/pools/{arm}.npz")
    sources.update({v[1]["id"]:v[1] for v in phase.values() if v[1]["id"] in used})
    pending=[]
    for name,r in sources.items():
        src=STORE/f"inputs/{name}.json";out=STORE/f"records/{name}.json"
        if src.is_file():assert json.loads(src.read_text())==r
        else:src.write_text(json.dumps(r)+"\n")
        if out.is_file(): continue
        if r["role"]=="reference":
            for base in (OLD,ROOT/"lattice_symmetry/v2/denovo/screen"):
                baseline=base/f"inputs/{name}.json";energy=base/f"records/{name}.json"
                if baseline.exists() and energy.exists() and json.loads(baseline.read_text())==r:
                    shutil.copyfile(energy,out);break
        if not out.exists():pending.append(name)
    (STORE/"panel.json").write_text(json.dumps(dict(indices_zero_based=INDICES,references=len(used),eligible="all 10000 before energies",novelty_catalog="all 27136 original MP-20 training structures (conservative)"),indent=2))
    with ProcessPoolExecutor(a.workers,initializer=initialize) as exe:
        for _ in exe.map(run.run,pending):pass
    summarize.main()
if __name__=="__main__":main()
