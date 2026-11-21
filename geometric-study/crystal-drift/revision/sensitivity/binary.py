"""Rescore the original ordered binary draws by maximum-site fit and native volume."""
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from data import retrieval_baseline
from standard_match import SETTINGS, make_structure, order, paths


def score(job):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    method, seed, external = job
    assert version("pymatgen") == "2024.4.13"
    if sys.platform == "win32":
        import psutil
        # Keep heavy numerical work off the cores reserved for timing.
        process = psutil.Process()
        permitted = [i for i in process.cpu_affinity() if 12 <= i <= 15]
        if permitted:
            process.cpu_affinity(permitted)
    with np.load(ROOT/"data/multi-test.npz") as test:
        idx = order("multi",test["ids"])
        ids, ref_z, ref_k, ref_f = (test[n][idx] for n in ("ids","z","k","f"))
        if method == "retrieval":
            with np.load(ROOT/"data/multi-train.npz") as train:
                k, f, _ = retrieval_baseline(test,train,20,seed)
            k, f = k[idx], f[idx]
        else:
            path = Path(external) if method=="uflow" else paths("multi",method,seed)
            with np.load(path) as pred:
                lookup = {v:i for i,v in enumerate(pred["ids"])}
                ix = [lookup[v] for v in ids]
                np.testing.assert_array_equal(pred["z"][ix],ref_z)
                k, f = pred["k"][ix], pred["f"][ix]
    assert k.shape[:2] == (128,20)
    matcher = StructureMatcher(**SETTINGS)
    rms = np.full((128,20),np.nan)
    fit = np.zeros((128,20),dtype=bool)
    status = np.full((128,20),"ok",dtype="U40")
    ratios = np.exp((k[:,:,:3].astype(float)-ref_k[:,None,:3]).sum(axis=-1))
    for i in range(128):
        truth=make_structure(ref_k[i],ref_f[i],ref_z[i])
        memo={}
        for j in range(20):
            key=k[i,j].tobytes(),f[i,j].tobytes()
            if key not in memo:
                try:
                    proposal=make_structure(k[i,j],f[i,j],ref_z[i])
                    np.testing.assert_allclose(ratios[i,j],proposal.volume/truth.volume,rtol=1e-4)
                    pair=matcher.get_rms_dist(proposal,truth)
                    memo[key]=(float(pair[0]) if pair is not None else np.nan,
                               bool(matcher.fit(proposal,truth)),"ok")
                except (ValueError,FloatingPointError,np.linalg.LinAlgError,OverflowError) as exc:
                    memo[key]=(np.nan,False,type(exc).__name__)
            rms[i,j],fit[i,j],status[i,j]=memo[key]
    ref=json.loads((ROOT/"structure-matching"/f"multi-{method}-{seed}.json").read_text())
    assert ids.tolist()==ref["material_ids"]
    np.testing.assert_allclose(rms[:,:5],np.array(ref["rms_by_candidate"],float),equal_nan=True,rtol=1e-7,atol=1e-7)
    np.savez_compressed(OUT/f"{method}-{seed}.npz",ids=ids,rms=rms,fit=fit,volume_ratio=ratios,status=status)
    print(method,seed,fit.any(1).sum(),flush=True)


def summarize():
    methods=("exact","flow1","flow16","retrieval","uflow")
    rng=np.random.default_rng(20261120)
    samples=rng.integers(0,128,(10000,128))
    rows=[]; paired=[]; by_case={}
    reference_ids=None
    for m in methods:
        seeds=(2026,) if m=="uflow" else (11,23,37)
        for seed in seeds:
            with np.load(OUT/f"{m}-{seed}.npz") as s:
                if reference_ids is None: reference_ids=s["ids"]
                np.testing.assert_array_equal(reference_ids,s["ids"])
                r=np.isfinite(s["rms"])
                v=s["volume_ratio"]
                for name,hit in (("rms",r),("max_site",s["fit"]),("rms_and_volume",r & (v>=1/1.1) & (v<=1.1))):
                    for k in (1,5,20):
                        h=hit[:,:k].any(1)
                        rows.append(dict(method=m,seed=seed,criterion=name,k=k,hits=int(h.sum()),n=len(h)))
                        by_case.setdefault((m,name,k),[]).append(h)
    for name in ("rms","max_site","rms_and_volume"):
        for k in (1,5,20):
            drift=np.mean(by_case["exact",name,k],0)
            for m in methods:
                v=np.mean(by_case[m,name,k],0)
                gap=drift-v
                low,high=np.percentile(gap[samples].mean(1)*100,[2.5,97.5])
                paired.append(dict(method=m,criterion=name,k=k,mean_pct=v.mean()*100,
                                   drift_minus_method_pp=gap.mean()*100,lower_pp=low,upper_pp=high))
    for filename,records in (("per-seed.csv",rows),("paired.csv",paired)):
        with (OUT/filename).open("w",newline="",encoding="utf8") as f:
            w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--external",type=Path,help="The ordered binary 128 by 20 uFlowCSP archive")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--summarize",action="store_true")
    args=ap.parse_args()
    if args.summarize:
        summarize()
    else:
        jobs=[("uflow",2026,args.external)] if args.external else [
            (m,s,None) for m in ("exact","flow1","flow16","retrieval") for s in (11,23,37)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(score,jobs))
