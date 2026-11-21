"""Audit the 128-case headline and timed candidate sequences with the same matcher."""
import argparse
import ctypes
import hashlib
import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0,str(ROOT))
import data
from standard_match import SETTINGS,make_structure,order,paths

METHODS = ("exact","flow1","flow16","retrieval","uflow")
SEEDS = (11,23,37)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(method,seed,kind,cache,overwrite,verify_existing):
    dest = OUT/f"rms-{kind}-{method}-{seed}.npz"
    if dest.exists() and not (overwrite or verify_existing): return True
    historical = kind=="headline"
    raw = OUT/(f"episode-{method}-{seed}.npz" if not historical else "uflow-20.npz")
    if historical and method != "uflow":
        raw = paths("multi",method,seed)
    if raw is not None and not raw.exists(): return False
    with np.load(ROOT/"data/multi-test.npz") as t:
        test = {key:t[key] for key in ("ids","z","k","f")}
    ix = order("multi",test["ids"])
    ids,z,kt,ft = (test[key][ix] for key in ("ids","z","k","f"))
    expected = json.loads((ROOT/"structure-matching"/f"multi-{method}-{seed}.json").read_text())
    assert list(ids)==expected["material_ids"] and len(set(ids))==128
    old = None if not cache or not historical or method=="uflow" else cache/f"match20-{method}-{seed}.npz"
    if old and old.exists():
        with np.load(old) as src:
            np.testing.assert_equal(src["ids"],ids)
            d = src["rms"]
        assert d.shape==(128,20)
        source = old.name
        source_sha = sha(old)
        errors = json.loads(old.with_suffix(".json").read_text())["exception_count"]
    else:
        if raw is None:
            with np.load(ROOT/"data/multi-train.npz") as train:
                k,f,_ = data.retrieval_baseline(test,train,20,seed)
            k,f = k[ix],f[ix]
        else:
            with np.load(raw) as pred:
                if len(pred["ids"]) == 128:
                    np.testing.assert_equal(pred["ids"],ids)
                    np.testing.assert_equal(pred["z"],z)
                    k,f = pred["k"],pred["f"]
                else:
                    np.testing.assert_equal(pred["ids"],test["ids"])
                    np.testing.assert_equal(pred["z"],test["z"])
                    k,f = pred["k"][ix],pred["f"][ix]
        assert k.shape==(128,20,6) and f.shape==(128,20,8,3)
        start = time.perf_counter()
        matcher = StructureMatcher(**SETTINGS)
        d = np.full((128,20),np.nan)
        errors = 0
        for i in range(128):
            ref = make_structure(kt[i],ft[i],z[i])
            memo = {}
            for j in range(20):
                key = k[i,j].tobytes(),f[i,j].tobytes()
                if key not in memo:
                    try:
                        pred = make_structure(k[i,j],f[i,j],z[i])
                        rms = matcher.get_rms_dist(pred,ref)
                        memo[key] = rms[0] if rms is not None else np.nan
                    except (ValueError,FloatingPointError,np.linalg.LinAlgError,OverflowError):
                        memo[key] = np.nan
                        errors += 1
                d[i,j] = memo[key]
        source = raw.name if raw else "train-only retrieval replay"
        source_sha = sha(raw) if raw else sha(ROOT/"data/multi-train.npz")
        print("matched",method,seed,kind,round(time.perf_counter()-start,1),flush=True)
    if historical:
        np.testing.assert_allclose(d[:,:5],np.array(expected["rms_by_candidate"],float),
                                   equal_nan=True,rtol=1e-7,atol=1e-7)
    if verify_existing:
        with np.load(dest) as prior:
            np.testing.assert_equal(prior["ids"],ids)
            np.testing.assert_allclose(prior["rms"],d,equal_nan=True,atol=1e-7,rtol=1e-7)
            assert int(prior["exception_count"])==errors
    else:
        np.savez_compressed(dest,ids=ids,rms=d,exception_count=errors,source=source,
                            source_sha256=source_sha,settings=json.dumps(SETTINGS),pymatgen=version("pymatgen"))
    return True


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods",nargs="+",default=METHODS,choices=METHODS)
    parser.add_argument("--kinds",nargs="+",default=["headline","episode"],choices=["headline","episode"])
    parser.add_argument("--verified-cache",type=Path,help="Optional original 20-draw audit caches; each five-draw prefix is checked.")
    parser.add_argument("--watch",action="store_true")
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--verify-existing",action="store_true",help="Regenerate and compare the full ordered RMS matrices.")
    parser.add_argument("--cores",nargs="+",type=int,default=[10,12])
    args=parser.parse_args()
    assert version("pymatgen")=="2024.4.13"
    if sys.platform=="win32":
        kernel=ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype=ctypes.c_void_p
        kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
        assert kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(),sum(1<<v for v in args.cores))
    elif hasattr(os,"sched_setaffinity"):
        os.sched_setaffinity(0,args.cores)
    jobs=[(m,s,kind) for kind in args.kinds for m in args.methods for s in ([2026] if m=="uflow" else SEEDS)]
    while jobs:
        jobs=[job for job in jobs if not score(*job,args.verified_cache,args.overwrite,args.verify_existing)]
        if not jobs or not args.watch: break
        time.sleep(3)
