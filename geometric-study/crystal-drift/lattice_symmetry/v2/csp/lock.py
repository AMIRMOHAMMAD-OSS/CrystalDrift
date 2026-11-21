"""Freeze one duration across all three seeds using only predeclared validation records."""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ARMS=("radial","angular","power0","fixed-angular","flow16")
SEEDS=(11,23,37)
DURATIONS=(256,1024,3392)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--arms",nargs="+",default=ARMS)
    args=p.parse_args();out=HERE/"selection.json"
    selected=json.loads(out.read_text()) if out.exists() else {}
    for arm in args.arms:
        scores={};fingerprints={};ids=None
        for u in DURATIONS:
            seeds=[]
            for seed in SEEDS:
                base=HERE/f"fits/{arm}-{seed}"
                val=base/f"val-{u:05}.npz";ck=base/f"update-{u:05}.pt"
                if not val.is_file():break
                with np.load(val) as d:
                    assert sha(ck)==str(d["checkpoint_sha256"])
                    if ids is None:ids=d["ids"].copy()
                    np.testing.assert_array_equal(ids,d["ids"])
                    if len(ids)!=128 or d["energy"].shape!=(128,) or not np.isfinite(d["energy"]).all():raise ValueError(val)
                    seeds.append(float(np.mean(d["energy"])))
                fingerprints[val.relative_to(HERE).as_posix()]=sha(val)
            if len(seeds)==3:scores[str(u)]=seeds
        if len(scores)!=len(DURATIONS):continue
        u=min(DURATIONS,key=lambda u:np.mean(scores[str(u)]))
        record=dict(updates=u,seeds=SEEDS,validation_energy_by_seed=scores,validation_sha256=fingerprints,
                    selected_checkpoint_sha256={str(seed):sha(HERE/f"fits/{arm}-{seed}/update-{u:05}.pt") for seed in SEEDS})
        if arm in selected:
            assert selected[arm]==json.loads(json.dumps(record)),"Frozen model selection changed"
        else:print("LOCK",arm,u,flush=True)
        selected[arm]=record
    tmp=out.with_suffix(".building");tmp.write_text(json.dumps(selected,indent=2)+"\n");tmp.replace(out)
    snapshots=HERE/"locks";snapshots.mkdir(exist_ok=True)
    requested={sha(out)}
    for r in (HERE/"test").glob("*/ready.json"):
        record=json.loads(r.read_text())
        if record["arm"]!="prior":requested.add(record["selection_sha256"])
    for n in range(1,len(selected)+1):
        for names in combinations(selected,n):
            body=(json.dumps({key:selected[key] for key in names},indent=2)+"\n").encode()
            # Match both the LF and Windows-native CRLF form of the recorded file.
            for b in (body,body.replace(b"\n",b"\r\n")):
                h=hashlib.sha256(b).hexdigest()
                if h in requested:(snapshots/(h+".json")).write_bytes(b)
    assert all((snapshots/(h+".json")).is_file() for h in requested)

if __name__=="__main__":main()
