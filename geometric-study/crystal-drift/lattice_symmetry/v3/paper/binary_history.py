"""Render the separate 128-target fixed-basis benchmark from ordered public-matcher records."""
import csv

import numpy as np

from write import ROOT, table


def wbm_replication():
    """One paired, five-draw comparison on the earlier formula-disjoint WBM cohort."""
    methods=("drift","flow1","flow16","retrieval")
    seeds=(11,23,37)
    with np.load(ROOT/"revision/full/wbm/test-crosswalk.npz") as data:
        assert len(set(data["ids"]))==128 and data["rms"].shape==(128,12,5)
        assert data["arms"].tolist()==[f"{name}-seed{s}" for name in methods for s in seeds]
        hits=np.isfinite(data["rms"]).any(axis=2).reshape(128,4,3)
    counts=["/".join(map(str,row)) for row in hits.sum(axis=0)]
    target_difference=(hits[:,0].astype(float)-hits[:,2]).mean(axis=1)
    effects,frequency=np.unique(target_difference,return_counts=True)
    boot=np.random.default_rng(7331).multinomial(128,frequency/128,size=5000)@effects/128
    low,high=100*np.quantile(boot,[.025,.975])
    endpoints=["0.00" if abs(v)<.005 else f"{v:+.2f}" for v in (low,high)]
    return (f"On the 128 WBM targets, five draws recovered {counts[0]} targets for fixed-basis drift, "
            f"{counts[1]} for one Euler step, {counts[2]} for 16 Euler steps and {counts[3]} for training "
            f"retrieval (seeds 11/23/37). The drift-minus-16-step difference was {100*target_difference.mean():+.2f} "
            f"percentage points (95\\% interval {endpoints[0]} to {endpoints[1]}), averaging the three paired outcomes "
            "per target before resampling targets.")


def binary_comparison():
    folder=ROOT/"revision/binary"
    with (folder/"target-crosswalk.csv").open(encoding="utf8",newline="") as src:
        targets=list(csv.DictReader(src))
    ids=[r["material_id"] for r in targets]
    assert len(set(ids))==len(ids)==128
    for i,r in enumerate(targets):
        species=r["atomic_numbers"].split()
        assert int(r["case"])==i and len(set(species))==2 and 2<=len(species)<=8
        assert r["in_uflow_train"]=="False" and r["in_uflow_test"]=="True"
    with (folder/"headline-prefix.csv").open(encoding="utf8",newline="") as src:
        earlier={(r["method"],int(r["draws"])):r for r in csv.DictReader(src)}
    rows=[]
    for key,name,calls in (("exact","Fixed-basis CrytalDrift",1),("flow1","Flow, one Euler step",1),
                           ("flow16","Flow, 16 Euler steps",16),("retrieval","Training retrieval",0),
                           ("uflow","Released uFlowCSP",1)):
        seeds=(2026,) if key=="uflow" else (11,23,37)
        scores=[]
        for seed in seeds:
            with np.load(folder/f"rms-headline-{key}-{seed}.npz") as data:
                assert data["ids"].tolist()==ids
                r=data["rms"]
            assert r.shape==(128,20)
            per_k=[]
            for k in (1,5,20):
                best=np.min(np.where(np.isfinite(r[:,:k]),r[:,:k],np.inf),axis=1)
                hit=np.isfinite(best)
                per_k.append((hit.mean(),best[hit].mean()))
            scores.append(per_k)
        avg=np.mean(scores,axis=0)
        for k,endpoint in zip((1,5,20),avg):
            archived=earlier[key,k]
            np.testing.assert_allclose(endpoint,[float(archived["mr"]),float(archived["rmse"])])
        rows.append([name,str(calls),*[f"{100*mr:.2f}" for mr,_ in avg],*[f"{rms:.3f}" for _,rms in avg]])
    return table("lrrrrrrr",["Historical pipeline","$S$","MR@1","MR@5","MR@20",
                             "RMS@1","RMS@5","RMS@20"],rows,
        "Fixed-basis precursor on the 128 hash-selected, two-to-eight-site binary MP-20 targets. "
        "MR is a percentage. RMS is the mean best normalized displacement over matched targets. "
        "The four compact or retrieval rows average seeds 11, 23 and 37 on identical targets. "
        "The released uFlowCSP row uses its full-chemistry weights and sampling seed 2026. "
        "The number of evaluations per target is $kS$.","tab:historical-binary")
