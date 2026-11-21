"""StructureMatcher recovery on preselected test structures and fixed draw indices."""
import argparse
import csv
import hashlib
import inspect
import json
import os
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

for n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[n] = "1"

import numpy as np

import data


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "structure-matching"
METHODS = {"ab": ("drift","raw","flow1","flow4","flow16","retrieval","exact"), "multi": ("drift","flow1","flow16","retrieval","exact")}
SETTINGS = dict(ltol=.3, stol=.5, angle_tol=10, primitive_cell=True, scale=True,
                attempt_supercell=False, allow_subset=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paths(cohort, name, seed):
    if name == "uflow":
        prefix="multi-" if cohort=="multi" else ""
        return HERE/"uflowcsp-results"/f"{prefix}seed2026-euler1.npz"
    if name.startswith("flow"):
        folder = "baseline-multi-results" if cohort=="multi" else "baseline-results"
        return HERE/folder/f"seed{seed}-euler{name[4:]}.npz"
    if name == "retrieval":
        return None
    folder = "raw" if name == "raw" else cohort+"-exact" if name == "exact" else cohort
    return HERE/"drift-results"/folder/f"seed{seed}.npz"


def order(cohort, ids):
    if cohort == "ab":
        return np.arange(len(ids))
    return np.array(sorted(range(len(ids)), key=lambda j: hashlib.sha256(ids[j].encode()).hexdigest())[:128])


def make_structure(k, f, z):
    from pymatgen.core import Lattice, Structure
    G = data.unpack_gram(k)
    if not np.isfinite(G).all() or np.linalg.det(G)<=.01:
        raise ValueError("Invalid cell volume (<0.1 Å³)")
    coordinates = np.stack([np.zeros(3),f]) if f.ndim==1 else f[z>0]
    return Structure(Lattice(np.linalg.cholesky(G)), z[z>0].tolist(), coordinates, to_unit_cell=True)


def run(job):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    cohort, name, seed = job
    path = paths(cohort,name,seed)
    destination = OUTPUT/f"{cohort}-{name}-{seed}.json"
    prefix = "multi-" if cohort=="multi" else ""
    reference_path = data.DATA/f"{prefix}test.npz"
    train_path = data.DATA/f"{prefix}train.npz"
    train,test = np.load(train_path),np.load(reference_path)
    sig = dict(input=sha(path) if path else "data.retrieval_baseline with 20 draws",test=sha(reference_path),
               train=sha(train_path) if path is None else None,seed=seed,script=sha(Path(__file__)),
               pymatgen=version("pymatgen"),settings=SETTINGS)
    if destination.is_file():
        previous=json.loads(destination.read_text())
        if previous.get("signature")==sig:return previous["summary"]
    tm = perf_counter()
    idx=order(cohort,test["ids"])
    if path is None:
        k,coord,_=data.retrieval_baseline(test,train,20,seed)
        k,coord=k[idx,:5],coord[idx,:5]
    else:
        candidate=np.load(path)
        if name=="uflow" and cohort=="multi" and len(candidate["ids"])==len(idx):
            np.testing.assert_equal(test["ids"][idx],candidate["ids"])
            np.testing.assert_equal(test["z"][idx],candidate["z"])
            k,coord=candidate["k"][:,:5],candidate["f"][:,:5]
        else:
            np.testing.assert_equal(test["ids"],candidate["ids"])
            np.testing.assert_equal(test["z"],candidate["z"])
            k,coord=candidate["k"][idx,:5],candidate["f"][idx,:5]
    z,ids=test["z"][idx],test["ids"][idx]
    kt,ft=test["k"][idx],test["f"][idx]
    if k.shape[1] < 5:raise ValueError("Five deposited candidates per case required")
    matcher=StructureMatcher(**SETTINGS)
    dist=np.full((len(idx),5),np.nan)
    errors=0
    for i in range(len(idx)):
        ref=make_structure(kt[i],ft[i],z[i]);memo={}
        for j in range(5):
            key=(k[i,j].tobytes(),coord[i,j].tobytes())
            if key not in memo:
                try:
                    pred=make_structure(k[i,j],coord[i,j],z[i])
                    rms=matcher.get_rms_dist(pred,ref)
                    memo[key]=float(rms[0]) if rms is not None else np.nan
                except (ValueError,FloatingPointError,np.linalg.LinAlgError,OverflowError):
                    memo[key]=np.nan;errors+=1
            dist[i,j]=memo[key]
    rates={}
    for budget in (1,5):
        a=dist[:,:budget]
        ok=np.isfinite(a).any(1)
        rates[f"match_rate_at_{budget}"]=float(ok.mean())
        rates[f"hits_at_{budget}"]=int(ok.sum())
        rates[f"mean_best_normalized_rms_among_hits_at_{budget}"]=float(np.nanmin(a[ok],axis=1).mean()) if ok.any() else None
    summary={"cohort":cohort,"method":name,"seed":seed,"n_targets":len(idx),**rates,"exceptions":errors,
             "runtime_seconds":round(perf_counter()-tm,3)}
    result=dict(signature=sig,summary=summary,material_ids=ids.tolist(),
                rms_by_candidate=[[float(v) if np.isfinite(v) else None for v in row] for row in dist])
    tmp=destination.with_suffix(".writing");tmp.write_text(json.dumps(result,indent=1)+"\n");tmp.replace(destination)
    print(cohort,name,seed,summary["hits_at_1"],summary["hits_at_5"],summary["runtime_seconds"],flush=True)
    return summary


def main():
    from pymatgen.analysis.structure_matcher import StructureMatcher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--external",action="store_true",help="Include aligned uFlow prediction files if present")
    parser.add_argument("--cohort",choices=("all","ab","multi"),default="all")
    args=parser.parse_args()
    print("pymatgen",version("pymatgen"),"matcher",inspect.signature(StructureMatcher),flush=True)
    OUTPUT.mkdir(exist_ok=True)
    jobs=[(c,m,s) for c,methods in METHODS.items() if args.cohort in ("all",c)
          for m in methods for s in (11,23,37) if paths(c,m,s) is None or paths(c,m,s).is_file()]
    if args.external:
        for c in METHODS:
            if args.cohort in ("all",c) and paths(c,"uflow",2026).exists():
                jobs.append((c,"uflow",2026))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        summaries=list(executor.map(run,jobs))
    name="summary.csv" if args.cohort=="all" else f"summary-{args.cohort}.csv"
    with (OUTPUT/name).open("w",encoding="utf-8",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    if args.cohort == "all":
        grouped = {}
        for row in summaries:
            grouped.setdefault((row["cohort"],row["method"]),[]).append(row)
        results=[]
        for (c,m),rows in sorted(grouped.items()):
            results.append(dict(cohort=c,method=m,seeds=" ".join(str(r["seed"]) for r in rows),cases=rows[0]["n_targets"],
                                mean_match_rate_at_1=np.mean([r["match_rate_at_1"] for r in rows]),
                                mean_match_rate_at_5=np.mean([r["match_rate_at_5"] for r in rows])))
        with (OUTPUT/"by-method.csv").open("w",encoding="utf-8",newline="") as out:
            writer=csv.DictWriter(out,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)


if __name__=="__main__":
    main()
