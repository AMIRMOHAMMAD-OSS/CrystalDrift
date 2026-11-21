"""Raw generation metrics for all 1,024 proposals and the 1,229 binary test cells."""
import argparse
import ast
import csv
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from importlib.metadata import version
import itertools
import json
import os
from pathlib import Path
import sys
import warnings

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from standard_match import make_structure


@lru_cache(maxsize=None)
def featurizers():
    from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
    from matminer.featurizers.composition import ElementProperty
    return CrystalNNFingerprint.from_preset("ops"),ElementProperty.from_preset("magpie")


@lru_cache(maxsize=None)
def valid_composition(symbols,counts):
    import smact
    from smact.screening import pauling_test
    if len(symbols)==1 or all(s in smact.metals for s in symbols):
        return True
    elements=smact.element_dictionary(symbols)
    ens=[elements[s].pauling_eneg for s in symbols]
    for states in itertools.product(*(elements[s].oxidation_states for s in symbols)):
        if sum(n*v for n,v in zip(counts,states)):
            continue
        try:
            if pauling_test(states,ens):
                return True
        except TypeError:
            return True  # Same missing-electronegativity convention as DiffCSP.
    return False


def inspect_one(job):
    from pymatgen.core import Element
    label,uid,z,k,f=job
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        crystal=make_structure(k,f,z)
        elements,counts=np.unique(z[z>0],return_counts=True)
        symbols=tuple(Element.from_Z(int(j)).symbol for j in elements)
        chem=valid_composition(symbols,tuple(counts))
        d=crystal.distance_matrix.copy()
        np.fill_diagonal(d,np.inf)
        geom=float(d.min())>=.5 and crystal.volume>=.1
        site,props=featurizers()
        err=""
        try:
            cf=np.asarray(props.featurize(crystal.composition),float)
            sf=np.mean([site.featurize(crystal,i) for i in range(len(crystal))],axis=0)
            if not np.isfinite(sf).all():
                raise ValueError("nonfinite site fingerprint")
        except Exception as exc:
            err=type(exc).__name__+": "+str(exc)[:180]
            sf=np.full(61,np.nan)
            cf=np.full(132,np.nan)
        row=dict(cohort=label,id=uid,volume=float(crystal.volume),density=float(crystal.density),
                 num_elements=len(elements),min_distance=float(d.min()),struct_valid=geom,
                 comp_valid=chem,fp_ok=not bool(err),fp_error=err)
        return row,cf,sf


def pin():
    import psutil
    p=psutil.Process()
    if hasattr(p,"cpu_affinity"):
        allowed=[i for i in p.cpu_affinity() if i not in (0,2,4)]
        if allowed:p.cpu_affinity(allowed)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers",type=int,default=3)
    args=ap.parse_args()
    jobs=[]
    for label,path in (("generated",ROOT/"de-novo/all-draws.npz"),("reference",ROOT/"data/multi-test.npz")):
        with np.load(path) as a:
            jobs.extend((label,*t) for t in zip(a["ids"],a["z"],a["k"],a["f"]))
    rows=[]; cf=[]; sf=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=pin) as pool:
        for row,c,s in pool.map(inspect_one,jobs,chunksize=24):
            rows.append(row);cf.append(c);sf.append(s)
    with (HERE/"per-cell.csv").open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    const={stmt.targets[0].id:np.asarray(ast.literal_eval(stmt.value)) for stmt in
           ast.parse((HERE/"diffcsp_constants.txt").read_text()).body if isinstance(stmt,ast.Assign)}
    cf=np.asarray(cf); sf=np.asarray(sf)
    np.savez_compressed(HERE/"fingerprints.npz",ids=[r["id"] for r in rows],cohort=[r["cohort"] for r in rows],
                        magpie=cf,crystalnn=sf)
    cf=np.nan_to_num((cf-const["CompScalerMeans"])/const["CompScalerStds"],nan=0.)
    gen=np.array([r["cohort"]=="generated" for r in rows]); good=np.array([r["fp_ok"] for r in rows])
    density=np.array([r["density"] for r in rows]); nelems=np.array([r["num_elements"] for r in rows])
    chem=np.array([r["comp_valid"] for r in rows]); geom=np.array([r["struct_valid"] for r in rows])
    valid=gen&geom&chem&good
    # Each space takes its own nearest neighbor, per DiffCSP compute_cov.
    a=gen&good;b=~gen&good
    dc=cdist(cf[a],cf[b]); ds=cdist(sf[a],sf[b])
    precision=np.count_nonzero((dc.min(1)<=10)&(ds.min(1)<=.4))/gen.sum()
    recall=np.count_nonzero((dc.min(0)<=10)&(ds.min(0)<=.4))/(~gen).sum()
    result=dict(draws=int(gen.sum()),reference=int((~gen).sum()),struct_valid=int((gen&geom).sum()),
                comp_valid=int((gen&chem).sum()),both=int((gen&geom&chem).sum()),valid_with_fp=int(valid.sum()),
                generated_fp_failures=int((gen&~good).sum()),reference_fp_failures=int((~gen&~good).sum()),
                coverage_precision=precision,coverage_recall=recall,
                density_w1_all=float(wasserstein_distance(density[gen],density[~gen])),
                density_w1_valid=float(wasserstein_distance(density[valid],density[~gen])),
                num_elements_w1_valid=float(wasserstein_distance(nelems[valid],nelems[~gen])),
                versions={s:version(s) for s in ("matminer","SMACT","pymatgen","numpy","scipy")},
                reference_commit="7121d159826efa2ba9500bf299250d96da37f146")
    (HERE/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
