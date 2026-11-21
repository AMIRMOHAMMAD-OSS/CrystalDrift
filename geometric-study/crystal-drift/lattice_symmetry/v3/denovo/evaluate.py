"""Re-use original validity, CrystalNN/Magpie and primitive novelty definitions on protected inputs."""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import csv
for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[name]="1"
HERE=Path(__file__).resolve().parent;V3=HERE.parent;ROOT=V3.parents[1];OLD=ROOT/"expansion/denovo"
site=Path(os.environ.get("CRYSTAL_METRICS_SITE",str(ROOT.parents[1]/"tmp/crystal-genmetrics-deps")))
sys.path[:0]=[str(OLD),str(ROOT),str(site)]
os.environ["CRYSTAL_COMPILED_FINGERPRINT"]="1"
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance
import score, novelty,report
score.HERE=novelty.HERE=report.HERE=HERE
original=score.arrays

@lru_cache(maxsize=4)
def arrays(arm):
    data=original(arm)
    if arm!="training":return data
    kept=json.loads((V3/"protocol/splits.json").read_text())["train_ids"]
    mask=np.isin(data["ids"],kept)
    assert mask.sum()==len(kept)
    return {k:v[mask] for k,v in data.items()}
score.arrays=novelty.arrays=report.arrays=arrays

def ref(kind):
    report.HERE=OLD
    try:return report.load_tiles("reference",kind)
    finally:report.HERE=HERE

def summarize(arm):
    d=arrays(arm);raw=report.load_tiles(arm,"raw");fp=report.load_tiles(arm,"fp");rr=ref("raw");rf=ref("fp")
    assert raw is not None and fp is not None
    const={v.targets[0].id:np.array(ast.literal_eval(v.value)) for v in ast.parse((ROOT/"revision/denovo/diffcsp_constants.txt").read_text()).body if isinstance(v,ast.Assign)}
    def features(p):
        mask=p["error"]==""
        return np.nan_to_num((p["comp"][mask]-const["CompScalerMeans"])/const["CompScalerStds"]),p["struc"][mask]
    rc,rs=features(rf);gc,gs=features(fp)
    def cov(c,s,c0,s0):return (cKDTree(c0).query(c)[0]<=10)&(cKDTree(s0).query(s)[0]<=.4)
    recall=cov(rc,rs,gc,gs);precision=cov(gc,gs,rc,rs)
    valid=raw["comp_valid"]&raw["struct_valid"];eligible=valid&(fp["error"]=="")
    pick=np.random.RandomState(7121).choice(np.flatnonzero(eligible),1000,replace=False)
    nov=list(csv.DictReader((HERE/f"novelty/{arm}.csv").open()))
    assert [r["id"] for r in nov]==d["ids"].tolist()
    unique=np.array([r["representative_index"]==str(i) and not r["error"] for i,r in enumerate(nov)])
    novel=np.array([not r["matched_train_id"] and not r["error"] for r in nov])
    out=dict(arm=arm,n=len(valid),struct_valid=int(raw["struct_valid"].sum()),comp_valid=int(raw["comp_valid"].sum()),
        joint_valid=int(valid.sum()),with_fingerprints=int(eligible.sum()),coverage_precision=int(precision.sum())/len(valid),
        coverage_recall=recall.mean(),reference=len(recall),w1_density=wasserstein_distance(raw["density"][pick],rr["density"]),
        w1_elements=wasserstein_distance(raw["nelem"][pick],rr["nelem"]),unique=int(unique.sum()),novel=int(novel.sum()),
        valid_unique_novel=int((valid&unique&novel).sum()),novelty_catalog=len(arrays("training")["ids"]),fp_errors=int((fp["error"]!="").sum()))
    dest=HERE/"results";dest.mkdir(exist_ok=True)
    (dest/(arm+".json")).write_text(json.dumps(out,indent=2))
    np.savez_compressed(dest/(arm+"-cases.npz"),ids=d["ids"],valid=valid,unique=unique,novel=novel,recall=recall,precision=precision,property_indices=pick)
    print(out,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("arm");ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--stages",nargs="+",default=["raw","fp","novelty","report"])
    a=ap.parse_args()
    for stage in a.stages:
        if stage in ("raw","fp"):
            n=len(arrays(a.arm)["ids"]);size=100 if stage=="raw" else 50
            jobs=[(a.arm,i,min(n,i+size),stage) for i in range(0,n,size) if not (HERE/f"scores/{stage}/{a.arm}/{i:05}-{min(i+size,n):05}.npz").exists()]
            if a.workers==1:
                for job in jobs:score.worker(job)
            else:
                with ProcessPoolExecutor(a.workers) as ex:
                    for _ in ex.map(score.worker,jobs):pass
        elif stage=="novelty":
            sys.argv=["novelty",a.arm,"--workers",str(a.workers)];novelty.main()
        elif stage=="report":summarize(a.arm)
        else:raise ValueError(stage)
        print(a.arm,stage,flush=True)

if __name__=="__main__":main()
