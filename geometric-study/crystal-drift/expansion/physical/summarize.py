"""Optimize competing-phase mixtures at each candidate composition, then test identity."""
import csv
from collections import defaultdict
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
from time import perf_counter
for name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[name]="1"
import numpy as np
from scipy.optimize import linprog
from pymatgen.analysis.structure_matcher import StructureMatcher
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT)]
from run import reduced
from standard_match import make_structure,SETTINGS


def write_csv(name,rows):
    keys=sorted(set().union(*(row.keys() for row in rows)))
    with (HERE/name).open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def fractions(z,elements):
    return np.array([(np.array(z)==e).mean() for e in elements])


def main():
    source={p.stem:json.loads(p.read_text()) for p in (HERE/"inputs").glob("*.json")}
    outcome={p.stem:json.loads(p.read_text()) for p in (HERE/"records").glob("*.json")}
    assert source.keys()==outcome.keys(), set(source)-set(outcome)
    matcher=StructureMatcher(**SETTINGS)
    with np.load(ROOT/"scaled/mp20-data/train.npz") as d:
        train={k:d[k] for k in ("ids","z","k","f")}
    train_formula=defaultdict(list)
    for i,row in enumerate(train["z"]):train_formula[reduced(row)].append(i)
    @lru_cache(maxsize=500)
    def train_structure(i):return make_structure(train["k"][i],train["f"][i],train["z"][i])
    representatives=defaultdict(list);rows=[]
    for name,raw in sorted(source.items()):
        if raw["role"]!="candidate":continue
        started=perf_counter()
        rec=outcome[name];z=np.array(raw["z"]);key=reduced(z);elements=np.unique(z)
        row=dict(id=name,arm=raw["arm"],draw=raw["index"]+1,seen=raw["seen"],converged=rec["converged"],
                 status=rec["status"],wall_seconds=rec["wall_seconds"],contacts_ok=False,unique=False,novel=False,
                 complete_hull=False,energetic_pass=None,screened_sun=False,possible_sun=False,reference_count=len(raw["reference_ids"]))
        rows.append(row)
        if "final" not in rec:
            row["postprocess_seconds"]=perf_counter()-started
            continue
        for frame in ("initial","final"):
            for field,number in rec[frame].items():row[frame+"_"+field]=number
        row["contacts_ok"]=rec["final"]["min_ratio"]>=.6
        struct=make_structure(np.array(rec["final_k"]),np.array(rec["final_f"]),z)
        group=representatives[(raw["arm"],key)]
        duplicates=[other for other,obj in group if matcher.fit(struct,obj)]
        row["unique"]=not duplicates;row["representative"]=duplicates[0] if duplicates else name
        if row["unique"]:group.append((name,struct))
        neighbor=next((i for i in train_formula[key] if matcher.fit(struct,train_structure(i))),None)
        row["novel"]=neighbor is None;row["training_match"]="" if neighbor is None else str(train["ids"][neighbor])
        refnames=[m for m in raw["reference_ids"] if outcome[m]["converged"] and "final" in outcome[m]]
        terminal={source[m]["z"][0] for m in refnames if len(set(source[m]["z"]))==1}
        missing=set(elements)-terminal
        row["missing_elements"]=" ".join(map(str,sorted(missing)))
        row["reference_failures"]=len(raw["reference_ids"])-len(refnames)
        row["complete_hull"]=not missing and row["reference_failures"]==0
        if refnames:
            refs=np.stack([fractions(source[m]["z"],elements) for m in refnames])
            energies=[outcome[m]["final"]["energy"] for m in refnames]
            res=linprog(energies,A_eq=refs.T,b_eq=fractions(z,elements),bounds=(0,None),method="highs")
            if res.success:
                gap=rec["final"]["energy"]-float(res.fun)
                row.update(partial_signed_gap=gap,active_reference_mixture=" ".join(f"{m}={w:.7g}" for m,w in zip(refnames,res.x) if w>1e-7))
                if row["complete_hull"]:
                    row.update(signed_gap=gap,above_augmented_hull=max(0.,gap),energetic_pass=gap<=.10)
            else:assert not row["complete_hull"],res.message
        passed=rec["converged"] and row["contacts_ok"] and row["unique"] and row["novel"]
        row["screened_sun"]=bool(passed and row["energetic_pass"] is True)
        row["possible_sun"]=bool(passed and (row["energetic_pass"] is True or row["energetic_pass"] is None))
        row["postprocess_seconds"]=perf_counter()-started
    write_csv("per-candidate.csv",rows)
    summary=[]
    for arm in sorted({r["arm"] for r in rows}):
        for group in ("all","seen","unseen"):
            selected=[r for r in rows if r["arm"]==arm and (group=="all" or (group=="seen")==r["seen"])]
            refs=set(mid for r in selected for mid in source[r["id"]]["reference_ids"])
            if selected:
                summary.append(dict(arm=arm,group=group,n=len(selected),
                    converged=sum(r["converged"] for r in selected),short=sum("final_min_ratio" in r and r["final_min_ratio"]<.60 for r in selected),
                    evaluated=sum("final_energy" in r for r in selected),complete_hulls=sum(r["complete_hull"] for r in selected),
                    thermally_screened=sum(r["energetic_pass"] is True for r in selected),
                    unique=sum(r["unique"] for r in selected),novel=sum(r["novel"] for r in selected),
                    sun_lower=sum(r["screened_sun"] for r in selected),sun_upper=sum(r["possible_sun"] for r in selected),
                    candidate_core_seconds=sum(r["wall_seconds"] for r in selected),
                    postprocess_seconds=sum(r["postprocess_seconds"] for r in selected),
                    reference_core_seconds=sum(outcome[m]["wall_seconds"] for m in refs),
                    unique_reference_phases=len(refs)))
    write_csv("by-arm.csv",summary)
    sources=[x for x in source.values() if x["role"]=="reference"]
    (HERE/"reference-library.json").write_text(json.dumps(dict(name="MP20 near-hull CHGNet library",
        units="eV per atom",phases=len(sources),converged=sum(outcome[x["id"]]["converged"] for x in sources),
        total_reference_core_seconds=sum(outcome[x["id"]]["wall_seconds"] for x in sources),
        phase_ids=sorted(x["id"] for x in sources)),indent=1)+"\n")
    print([{k:r[k] for k in ("arm","n","converged","complete_hulls","sun_lower","sun_upper")} for r in summary if r["group"]=="all"])


if __name__=="__main__":main()
