"""Independently check numerical bounds, split isolation, negative-proof searches and coverage counts."""
import hashlib,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;V3=HERE.parent;OUT=HERE/"audits"
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    checks={}
    for device in ("cpu","cuda"):
        data=read(HERE/f"verification-{device}.json")
        for dtype,x in data["cases"].items():
            feats=[val for k,block in x["descriptor"].items() for a,val in block.items()
                   if a!="directional_derivative_error"]
            vec=[v["representation"] for v in x["finite_bank"].values()]
            maximum=max(feats+vec)
            assert maximum<(5e-11 if dtype.endswith("64") else 2e-4),(device,dtype,maximum)
            assert abs(x["witnesses"]["identical_local_shells"]["base"])<1e-7
            assert x["witnesses"]["identical_local_shells"]["reciprocal"]>.004
            checks[f"{device}-{dtype}"]=maximum
    for name in ("angular","contacts","reciprocal"):
        r=read(HERE/f"decoder-selected-{name}.json")
        spec=next(x for x in read(V3/"evaluation/first-frozen.json")["methods"] if x["tag"]==name+"-small-11")
        assert r["checkpoint_sha256_or_init"]==spec["sha"]==digest(V3/spec["checkpoint"])
        for dt,vals in r["by_precision"].items():
            assert vals["fixed_orbits"]>=1 and vals["minimum_gram_eigenvalue"]>0
            assert max(vals["relative_parameter_gradient"].values()) < (5e-11 if dt.endswith("64") else 2e-4)
    pyr=read(OUT/"pyxtal-roundtrip.json");assert len(pyr["records"])==160
    assert all(r["spglib"]["exact_operations"] and r["input_hall"]==r["pyxtal_hall"] for r in pyr["records"])
    checked=[r["reduction_diagnostic"] for r in pyr["records"] if "reduction_diagnostic" in r]
    checks["stock_pyxtal_bidirectional_matches"]=sum(r["rms"] is not None and r["reverse_rms"] is not None for r in pyr["records"])
    checks["stock_failures_with_independent_primitive_match"]=sum(r["separately_spglib_reduced_rms"] is not None and r["separately_spglib_reduced_reverse_rms"] is not None for r in checked)
    inv=read(V3/"models/cache/inventory.json");train=set(inv["eligible"]);splits=read(V3/"protocol/splits.json")
    assert not train.intersection(splits["shared_ids"]+splits["family_ids"])
    case=read(OUT/"shared-geometry.json");assert {r["id"] for r in case["records"]}==set(splits["shared_ids"])
    assert all(a["donor"] in train for r in case["records"] for a in r["attempts"])
    assert sum(any(a["rms"] is not None for a in r["attempts"]) for r in case["records"])==case["matched"]
    regfile=OUT/"arrangement-registration.json";classfile=OUT/"arrangement-progress.json"
    reg,state,freeze=map(read,[regfile,classfile,OUT/"arrangement-frozen.json"])
    assert freeze["registration_sha256"]==state["registration_sha"]==digest(regfile)
    assert freeze["classification_sha256"]==digest(classfile)
    u=reg["registered"];rr=state["rows"]
    assert len({v["id"] for v in u})==len(u)==512
    assert set(x["id"] for x in u).isdisjoint(reg["excluded_ids"])
    assert len(rr)<=512 and (len(rr)==512 or freeze["counts"]["uncovered"]==96)
    for x,y in zip(u,rr):
        assert (x["id"],x["key"])==(y["id"],y["key"])
        tried=[v["id"] for v in y["attempts"]]
        assert set(x["donors"])<=train and tried==x["donors"][:len(tried)]
        assert 0<len(x["donors"])<=64
        hit=any(v["rms"] is not None for v in y["attempts"])
        if y["status"]=="uncovered":
            assert tried==x["donors"] and not hit and all("error" not in a for a in y["attempts"])
        elif y["status"]=="covered":assert hit
    checks["uncovered_validation_cases"]=len(freeze["uncovered_ids"])
    checks["eligible_anonymous_shared_family_matches"]=case["matched"]
    coverage={}
    for file in OUT.glob("reachable-*.json"):
        m=read(file)
        with np.load(file.with_suffix(".npz")) as d:
            assert len(d["ids"])==m["ids"] and int(d["truth_available"].sum())==m["truth_available"]
            for k in (1,5,20):
                assert m["reach_by_k"][str(k)]=={name:int(d[name][:,:k].any(1).sum())
                    for name in ("canonical","equivalent","family","group","fixed")}
                assert m["fixed_draws_by_k"][str(k)]==d["fixed"][:,:k].sum()
            assert np.all(~d["canonical"]|d["equivalent"])
            assert np.all(~d["equivalent"]|d["family"])
        coverage[m["panel"]]=m["ids"]
    checks["coverage_records"]=coverage
    (HERE/"geometry-audit.json").write_text(json.dumps(checks,indent=2)+"\n",encoding="utf-8")
    print(checks)

if __name__=="__main__":main()
