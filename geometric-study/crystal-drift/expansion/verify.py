"""Cross-check manuscript endpoints against individual ordered records and source hashes."""
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARMS = ("empirical-drift","learned-drift","learned-flow16","learned-uflow","crystalflow-joint")

def read_json(p):
    return json.loads(p.read_text())

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def timed():
    gpu=HERE/"training/gpu"
    expected={"drift":27,"flow16":6,"meanflow":33,"uflow":41,"retrieval":88}
    out={}
    for key,hits in expected.items():
        with np.load(gpu/f"{key}-episodes.npz") as t,np.load(gpu/f"{key}-rms.npz") as score,np.load(gpu/f"{key}-10000.npz") as b:
            np.testing.assert_array_equal(t["ids"],score["ids"])
            assert len(t["ids"])==len(set(t["ids"]))==128
            before=t["elapsed_ms"]<=100
            value=int((before&np.isfinite(score["rms"])).any(1).sum())
            assert value==hits and b["counts"].sum()==10000
            out[key]=dict(targets=128,hits=value,mean_completed=float(before.sum(1).mean()),
                          fixed_workload_s=float(b["elapsed_s"]))
    ext=read_json(HERE/"external/timing-s5/runtime.json")
    return dict(compact_and_one_step=out,external_timing=ext)

def formula(row):
    counts=Counter(int(x) for x in row if x)
    divisor=np.gcd.reduce(list(counts.values()))
    return tuple(sorted((z,n//divisor) for z,n in counts.items()))

def audit_cohort(h):
    known_ids=set();known_formulas=set()
    for path in [*(ROOT/f"scaled/mp20-data/{s}.npz" for s in ("train","val","test")),ROOT/"scaled/alex-validation-transfer/val.npz"]:
        with np.load(path) as source:
            known_ids.update(str(i).lower() for i in source["ids"])
            known_formulas.update(formula(z) for z in np.unique(source["z"],axis=0))
    with np.load(h/"cohort.npz") as selected,np.load(ROOT/"scaled/alex-data/val.npz") as archive:
        ids=selected["ids"]; forms={formula(z) for z in selected["z"]}
        assert len(set(ids))==len(forms)==512 and not (forms&known_formulas)
        for mid in ids:
            assert str(mid).lower() not in known_ids
            assert not (set(re.findall(r"(?:mp|mvc)-\d+",str(mid).lower()))&known_ids)
        for name in ("ids","z","k","f"):
            np.testing.assert_array_equal(selected[name],archive[name][selected["validation_index"]])
        return ids.copy()

def hybrid_volume():
    h=HERE/"hybrid"
    source_ids=audit_cohort(h)
    design=read_json(h/"audit.json")
    assert design["selected_formula_overlap"]==design["selected_id_overlap"]==0
    assert digest(h/"DESIGN.md")==design["design_sha256"]
    with np.load(h/"confirmation-scores.npz") as data:
        # The proposal writer orders targets by its evaluation hash after selection.
        # Crosswalk IDs explicitly, preserving the native draw axis in every bank.
        assert set(data["ids"])==set(source_ids)
        for path in sorted((h/"pools").glob("*-confirm.npz")):
            with np.load(path) as bank:
                np.testing.assert_array_equal(data["ids"],bank["ids"])
        assert len(set(data["ids"]))==512
        hit=np.isfinite(data["rms"]).reshape(512,4,3,20)
        donors=hit[:,3]
        ref=donors.any(-1)
        out={}
        for i,arm in enumerate(("drift","flow1","flow16")):
            mix=donors[:,:,0]|hit[:,i,:,:19].any(-1)
            out[arm]={"hybrid":float(mix.mean()),"retrieved_20":float(ref.mean()),
                      "exclusive":float((mix&~ref).mean()),"lost":float((ref&~mix).mean())}
        np.testing.assert_allclose([out[k]["hybrid"]*100 for k in out],[39.39,35.09,41.80],atol=.005)
    v=HERE/"volume"
    summary=read_json(v/"summary.json")
    with np.load(v/"corrected-scores.npz") as data,np.load(ROOT/"revision/full/mp20/complete-crosswalk-k20.npz") as orig:
        np.testing.assert_array_equal(data["ids"],orig["ids"])
        assert data["rms"].shape==(9046,3,20) and summary["exceptions"]==0
        ix=[orig["arms"].tolist().index(f"drift-seed{x}") for x in [11,23,37]]
        stats=[]
        for raw,fit,vol in ((orig["rms"][:,ix],orig["strict_fit"][:,ix],orig["log_volume_ratio"][:,ix]),(data["rms"],data["strict_fit"],data["log_volume_ratio"])):
            passed=np.isfinite(raw)
            stats.append({f"{n}@{k}":float(test[:,:,:k].any(2).mean())
                          for n,test in (("mr",passed),("maximum_site",fit),("volume_match",passed&(np.abs(vol)<=np.log(1.1)))) for k in (1,5,20)})
        for k in stats[0]:np.testing.assert_allclose([a[k] for a in stats],[summary[k]["raw"],summary[k]["corrected"]])
    return dict(hybrid=out,volume=stats)

def denovo():
    d=HERE/"denovo";out={}
    for arm in (*ARMS,"crystalflow-joint-local"):
        s=read_json(d/f"panel/{arm}.json")
        with np.load(d/f"panel/{arm}-cases.npz") as c,np.load(d/f"panel/{arm}-coverage.npz") as cv,np.load(d/f"energies/{arm}.npz") as e,np.load(d/"energies/reference.npz") as ref:
            assert len(set(c["ids"]))==10000 and len(ref["ids"])==9046 and len(set(e["ids"]))==1000
            for flag,key in (("struct_valid","struct_valid"),("comp_valid","comp_valid"),("valid","joint_valid"),("unique","unique"),("structural_novel","structural_novel")):
                assert int(c[flag].sum())==s[key]
            assert int((c["valid"]&c["unique"]&c["structural_novel"]).sum())==s["valid_unique_novel"]
            np.testing.assert_array_equal(c["ids"][e["index"]],e["ids"])
            np.testing.assert_array_equal(np.sort(c["w1_sample"]),e["index"])
            assert c["valid"][e["index"]].all()
            er=wasserstein_distance(ref["energy_eV_per_atom"],e["energy_eV_per_atom"])
            recall=((cv["ref_chem"]<=10)&(cv["ref_struc"]<=.4)).sum()/9046
            prec=((cv["generated_chem"]<=10)&(cv["generated_struc"]<=.4)).sum()/10000
            np.testing.assert_allclose([recall,prec,er],[s["coverage_recall"],s["coverage_precision"],s["w1_formation_energy"]])
            out[arm]={k:s[k] for k in ("joint_valid","coverage_recall","coverage_precision","w1_formation_energy","formula_unseen_training","valid_unique","valid_unique_novel")}
    assert out["learned-drift"]["formula_unseen_training"]==6478
    return out

def physical():
    p=HERE/"physical";per=list(csv.DictReader((p/"per-candidate.csv").open()))
    assert len(per)==80
    lib=read_json(p/"reference-library.json")
    assert len(lib["phase_ids"])==lib["converged"]==671
    inputs={f.stem:read_json(f) for f in (p/"inputs").glob("*.json")}
    outcomes={f.stem:read_json(f) for f in (p/"records").glob("*.json")}
    assert inputs.keys()==outcomes.keys() and len(inputs)==751
    out={}
    for arm in ARMS:
        rows=[r for r in per if r["arm"]==arm];assert len(rows)==16
        hits=0
        for r in rows:
            raw=inputs[r["id"]];rec=outcomes[r["id"]]
            elements=np.unique(raw["z"])
            def normalized(z):
                return (np.asarray(z)[:,None]==elements[None,:]).mean(0)
            comp=normalized(raw["z"]); refs=raw["reference_ids"]
            assert all(outcomes[mid]["converged"] for mid in refs)
            basis=np.column_stack([normalized(inputs[mid]["z"]) for mid in refs])
            energies=np.array([outcomes[mid]["final"]["energy"] for mid in refs])
            # Add the candidate to the reference set explicitly. The stored screen uses
            # the signed base-reference difference with its negative part truncated.
            augmented=linprog(np.r_[energies,rec["final"]["energy"]],
                              A_eq=np.column_stack([basis,comp]),b_eq=comp,bounds=(0,None),method="highs")
            assert augmented.success
            np.testing.assert_allclose(rec["final"]["energy"]-augmented.fun,float(r["above_augmented_hull"]),atol=5e-8)
            assert (not r["training_match"])==(r["novel"]=="True")
            assert r["complete_hull"]=="True" and not r["missing_elements"]
            good=rec["converged"] and rec["final"]["min_ratio"]>=.6 and float(r["above_augmented_hull"])<=.1 and r["novel"]=="True" and r["unique"]=="True"
            assert good==(r["screened_sun"]=="True");hits+=good
        out[arm]=hits
    assert tuple(out.values())==(2,1,2,3,5)
    assert len(list((p/"cifs").glob("*.cif")))==160
    return out

def main():
    condition=read_json(HERE/"training/verification.json")
    external=read_json(HERE/"external/csp-s5/audit.json")
    assert condition["total_generated_and_scored"]==2713800 and external["proposals"]==180920
    with np.load(HERE/"external/csp-s5/score-crosswalk.npz") as ext:
        er=ext["rms"]
        assert er.shape==(9046,20)
        hist=read_json(HERE/"external/csp-s5/summary.json")
        for k in [1,5,20]:
            sub=er[:,:k];hit=np.isfinite(sub).any(1)
            vals=np.where(np.isfinite(sub),sub,np.inf).min(1)[hit]
            np.testing.assert_allclose([hit.mean(),vals.mean()],[hist[f"9046@{k}"]["mr"],hist[f"9046@{k}"]["rmse"]])
    theory=read_json(HERE/"weak-limit.json")
    assert theory["exchange_sign_error"]<1e-12 and theory["series"][-1]["max_expected_query_norm"]<1e-15
    assert theory["series"][-1]["fixed_first_positive_norm"]>.4
    dft=read_json(HERE/"dft/summary.json")
    assert len(dft["results"])==8
    for r in dft["results"].values():assert "JOB DONE" in (HERE/"dft"/r["file"]).read_text(errors="replace")
    all_raw=denovo()
    secondary_raw=all_raw.pop("crystalflow-joint-local")
    timing=read_json(HERE/"denovo/pools/crystalflow-joint-local.json")
    local=read_json(HERE/"physical-joint-local/verification.json")
    local_audit=read_json(HERE/"external/joint-local-audit.json")
    assert timing["nfe"]==100 and timing["draws"]==local_audit["unique_ids"]==10000
    assert len(timing["b1_ms"])==128 and min(timing["b1_ms"])>0
    assert timing["elapsed_s"]==local["costs"]["generation_gpu_s"]>0
    per_local=list(csv.DictReader((HERE/"physical-joint-local/per-candidate.csv").open()))
    with np.load(HERE/"denovo/pools/crystalflow-joint-local.npz") as raw:
        assert raw["ids"][:16].tolist()==local["verified_draw_ids"]
    assert [int(r["draw"]) for r in per_local]==list(range(1,17))
    hits=sum(r["screened_sun"]=="True" for r in per_local)
    assert hits==local["strata"]["all"]["hits"]
    np.testing.assert_allclose(3600*hits/local["costs"]["total_charged_s"],local["costs"]["hits_per_charged_hour"])
    result=dict(timing=timed(),**hybrid_volume(),raw_multielement=all_raw,physical_hits=physical(),
                independent_joint=dict(raw=secondary_raw,screen=local["strata"],costs=local["costs"]),
                compact_proposals=condition["total_generated_and_scored"],external_proposals=external["proposals"],
                dft_single_points=len(dft["results"]),dft_energies=dft["relaxed_minus_raw_eV_per_atom"],
                weak_limit=theory["series"][-1],source_sha256={})
    files=["training/verification.json","external/csp-s5/audit.json","denovo/panel/audit.json",
           "hybrid/confirmation-scores.npz","volume/corrected-scores.npz","physical/per-candidate.csv","dft/summary.json"]
    files.extend(["denovo/pools/crystalflow-joint-local.json","denovo/panel/crystalflow-joint-local-cases.npz",
                  "physical-joint-local/verification.json","external/step-annealing-factorial.json",
                  "external/joint-local-audit.json"])
    for f in files:result["source_sha256"][f]=digest(HERE/f)
    (HERE/"verified-claims.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("Verified compact and external predictions, time budgets, hybrid, volume, 60,000 de novo draws, physical screens and DFT.")

if __name__=="__main__":
    main()
