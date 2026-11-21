"""Independently audit labelled inputs, checkpoint identity, ordered outcomes and physical comparisons."""
import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from charts import HERE,expand,label
from geometry import cohort,quality,reference,TOLS
from labels import ROOT,SETTINGS,dataset,read
from data import unpack_gram
from report import LABELS
from train_template import signature

def resume_check(path):
    source=HERE/"training/angular-template-256.pt"
    states=[torch.load(f,map_location="cpu",weights_only=True) for f in (path,source)]
    d=max(float((p-states[1]["model"][k]).abs().max()) for k,p in states[0]["model"].items())
    assert d==0 and states[0]["step"]==states[1]["step"]==256
    result=dict(step=256,split_at=64,max_parameter_difference=d,canonical_checkpoint=signature(source),
                replay_checkpoint=signature(path))
    (HERE/"restart.json").write_text(json.dumps(result,indent=2)+"\n")

def verify():
    s=json.loads((HERE/"summary.json").read_text());c=cohort("test");matcher=StructureMatcher(**SETTINGS)
    assert list(LABELS)==list(s["arms"])
    tests=json.loads((HERE/"template-tests.json").read_text())
    assert tests["positions"]==1731 and tests["hall_groups"]==230 and len(tests["templates"])==39
    results={"source":{},"cases":{}}
    reciprocal_max=0.
    readsrc=lambda p: json.loads(p.read_text())
    for tag in ("train","val","test"):
        labels=list(read(tag));ids,cells,fs,zs=dataset(tag)
        assert [r["id"] for r in labels]==ids.tolist() and len(set(ids))==len(labels)
        assert sum("error" not in r for r in labels)==s["labels"][tag]["accepted"]
        for i in np.random.default_rng(734).choice(len(labels),8,replace=False):
            if "error" in labels[i]:continue
            z=zs[i,zs[i]>0];mask=zs[i]>0
            new=label(cells[i],fs[i,mask],z)
            rec=labels[i]
            assert new["group"]==rec["group"] and new["template"]==[tuple(v) for v in rec["template"]]
            f,sz=expand(new["group"],new["template"],new["u"])
            m=matcher.get_rms_dist(Structure(cells[i],z,fs[i,mask]),Structure(new["cell"],sz,f))
            np.testing.assert_allclose(m[0],rec["roundtrip_rms"],atol=2e-8)
        results["source"][tag]=signature(ROOT/f"scaled/mp20-data/{tag}.npz")
    for arm,report in s["arms"].items():
        pool=HERE/f"pools/{arm}.npz";meta=readsrc(HERE/f"pools/{arm}.json")
        assert meta["pool_sha256"]==signature(pool)
        if "checkpoint" in meta:
            path=ROOT/str(meta["checkpoint"]).replace("\\","/")
            assert path.is_file() and signature(path)==meta["sha256"]
        elif arm.startswith("DiffCSPpp"):
            src=readsrc(HERE/"external/source.json")
            assert src["checkpoint_sha256"]==meta["checkpoint_sha256"]
            assert signature(HERE/"external/weights.npz")==src["weights_sha256"]
            assert meta["nfe"]==2*meta["steps"]
        with np.load(pool) as b:L,F,Z,ids=b["L"],b["F"],b["Z"],b["id"]
        reciprocal_max=max(reciprocal_max,float(np.linalg.norm(np.linalg.inv(L),axis=-2).max()))
        assert ids.tolist()==[r["id"] for r in c]
        with gzip.open(HERE/f"scores/{arm}.jsonl.gz","rt") as fi:rows=list(map(json.loads,fi))
        n,k=L.shape[:2]
        assert n==128 and len(rows)==n*k
        hits=[]
        for i,r in enumerate(c):
            for j in range(k):
                row=rows[i*k+j]
                assert (row["id"],row["draw"])==(r["id"],j+1)
            if i in [1,2,17,99]:
                ref=reference(r,"test")
                for j in {0,k-1}:
                    mask=Z[i]>0;rr=rows[i*k+j]
                    rec=quality(L[i,j],F[i,j,mask],Z[i,mask],r,ref)
                    for key in ["valid","contains0.01","exact0.01","exactops0.01","detected_order0.01","requested_order","occupancy0.01","direct0.01"]:
                        assert rec[key]==rr[key]
                    if rec["rms"] is None:assert rr["rms"] is None
                    else:np.testing.assert_allclose(rr["rms"],rec["rms"],atol=2e-7)
        scores=np.array([np.inf if v.get("rms") is None else v["rms"] for v in rows]).reshape(n,k)
        for pref,r in report["recovery"].items():
            best=scores[:,:int(pref)].min(-1);hit=np.isfinite(best)
            assert int(hit.sum())==r["hits"]
            np.testing.assert_allclose(best[hit].mean(),r["matched_rms"])
        screens=readsrc(HERE/f"relaxation/{arm}.json")
        assert [r["id"] for r in screens]==ids[:16].tolist()
        for i,status in enumerate(screens):
            path=HERE/f"relaxation/inputs/{arm}-{i:03}.json"
            if not path.exists():assert status["status"]=="no raw geometry";continue
            q=readsrc(path);mask=Z[i]>0
            assert q["target"]==ids[i] and q["draw"]==1 and np.array_equal(q["z"],Z[i,mask])
            np.testing.assert_allclose(q["f"],F[i,0,mask])
            saved=readsrc(HERE/f"relaxation/records/{arm}-{i:03}.json")
            assert status["status"]==saved["status"] and status["converged"]==saved["converged"]
            if "final_k" in saved:
                assert (HERE/f"relaxation/cifs/{arm}-{i:03}-relaxed.cif").exists()
                lattice=np.linalg.cholesky(unpack_gram(saved["final_k"]))
                reciprocal_max=max(reciprocal_max,float(np.linalg.norm(np.linalg.inv(lattice),axis=0).max()))
        results["cases"][arm]=dict(ids=n,raw=n*k,physical=len(screens),sha256=signature(pool))
    with np.load(HERE/"full/one-draw.npz") as b:
        assert b["ids"].tolist()==dataset("test")[0].tolist()
        hits=np.isfinite(b["rms"])
        re=readsrc(HERE/"full/summary.json")
        assert hits.sum(0).tolist()==re["hits"] and re["cohort"]==9046
        np.testing.assert_allclose(np.nanmean(b["rms"],axis=0),re["mean_matched_rms"])
    assert reciprocal_max*max(TOLS)<.5
    results["max_reciprocal_column_norm"]=reciprocal_max
    r=readsrc(HERE/"restart.json");assert r["max_parameter_difference"]==0.
    for p in ("charts.npz","chart-source.json","DESIGN.md","training/selection.json","training-orbit/selection.json",
              "full/one-draw.npz","template-tests.json","template-timing.json","operation-order-audit.json","summary.json","restart.json",
              "../paper/wyckoff.tex","../paper/wyckoff-supplement.tex"):
        results["source"][p]=signature(HERE/p)
    (HERE/"verified.json").write_text(json.dumps(results,indent=2)+"\n")
    print("Verified labelled source data, all ordered scores, template and physical cohorts.")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--restart",type=Path);a=p.parse_args()
    if a.restart:resume_check(a.restart)
    else:verify()
