"""Independently crosswalk the ordered samples, property subsets and surrogate screen."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linprog
from scipy.stats import wasserstein_distance

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ARMS=("fixed-basis","radial","angular")


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def check_pools(arms=ARMS):
    base=ROOT/"expansion/denovo/pools/learned-drift.npz"
    with np.load(base) as b:
        ids=b["ids"];z=b["z"]
    result={}
    for arm in arms:
        p=HERE/"pools"/(arm+".npz");clock=read(p.with_name(arm+"-cpu.json"))
        assert clock["checkpoint_sha256"]==sha(ROOT/clock["checkpoint"]) and clock["data_sha256"]==sha(p)
        if arm!="fixed-basis":
            lock=p.with_name(arm+"-selection.json")
            assert sha(lock)==clock["validation_selection_sha256"]
            assert read(lock)["selected_checkpoint_sha256"]["11"]==clock["checkpoint_sha256"]
        with np.load(p) as b,np.load(HERE/"panel"/(arm+"-cases.npz")) as c,np.load(HERE/"panel"/(arm+"-coverage.npz")) as d:
            np.testing.assert_array_equal(ids,b["ids"])
            np.testing.assert_array_equal(ids,c["ids"])
            np.testing.assert_array_equal(z,b["z"])
            assert len(ids)==10000 and len(set(ids))==10000 and len(clock["b1_ms"])==128
            np.testing.assert_allclose(b["elapsed_by_batch"].sum(),clock["elapsed_s"])
            x=b["lattice"]
            lam,vec=np.linalg.eigh(x@x.transpose(0,2,1))
            s=(vec*(.5*np.log(lam))[:,None,:])@vec.transpose(0,2,1)
            k=np.concatenate([np.diagonal(s,axis1=1,axis2=2),np.sqrt(2)*s[:,[0,0,1],[1,2,2]]],1)
            np.testing.assert_allclose(k,b["k"],rtol=1e-4,atol=2e-5)
            summary=read(HERE/"panel"/(arm+".json"))
            for flag,stat in (("struct_valid","struct_valid"),("comp_valid","comp_valid"),("valid","joint_valid"),
                              ("unique","unique"),("structural_novel","structural_novel")):
                assert int(c[flag].sum())==summary[stat]
            V=c["valid"]; U=c["unique"]; N=c["structural_novel"]
            assert int((V&U&N).sum())==summary["valid_unique_novel"]
            recall=np.count_nonzero((d["ref_chem"]<=10)&(d["ref_struc"]<=.4))/9046
            precision=np.count_nonzero((d["generated_chem"]<=10)&(d["generated_struc"]<=.4))/10000
            np.testing.assert_allclose([recall,precision],[summary["coverage_recall"],summary["coverage_precision"]])
            chosen=np.random.RandomState(7121).choice(np.flatnonzero(V),size=1000,replace=False)
            np.testing.assert_array_equal(np.sort(chosen),np.sort(c["w1_sample"]))
            ref_files=sorted((ROOT/"expansion/denovo/scores/raw/reference").glob("*.npz"))
            # Baseline can have overlapping chunk sizes. Reconstruct an exact partition by ID.
            refrows={}
            for file in ref_files:
                with np.load(file) as dref:
                    for m,r,t in zip(dref["ids"],dref["density"],dref["nelem"]):refrows[str(m)]=(r,t)
            assert len(refrows)==9046
            vals=np.array(list(refrows.values()))
            np.testing.assert_allclose(wasserstein_distance(c["density"][chosen],vals[:,0]),summary["w1_density"])
            np.testing.assert_allclose(wasserstein_distance(c["nelem"][chosen],vals[:,1]),summary["w1_nelems"])
            with np.load(HERE/"energies"/(arm+".npz")) as en,np.load(HERE/"energies/reference.npz") as ref:
                assert len(ref["ids"])==9046 and len(en["ids"])==1000
                np.testing.assert_array_equal(ids[en["index"]],en["ids"])
                np.testing.assert_array_equal(en["index"],np.sort(chosen))
                ew=wasserstein_distance(ref["energy_eV_per_atom"],en["energy_eV_per_atom"])
                np.testing.assert_allclose(ew,summary["w1_formation_energy"])
            result[arm]=dict(checkpoint=clock["checkpoint"],checkpoint_sha256=clock["checkpoint_sha256"],draws=len(ids),
                             valid=int(V.sum()),valid_unique=int((V&U).sum()),valid_unique_novel=int((V&U&N).sum()),
                             coverage_recall=recall,coverage_precision=precision,energy_w1=ew,
                             property_records=1000,raw_sha256=sha(p),unique_per_cpu_min=int((V&U).sum())*60/clock["elapsed_s"])
    return result


def _reduced_counts(z):
    el,count=np.unique(np.asarray(z)[np.asarray(z)>0],return_counts=True)
    return el,count//np.gcd.reduce(count)


def quick_archive_check():
    store=HERE/"screen"
    data={p.stem:read(p) for p in (store/"inputs").glob("*.json")}
    outcomes={p.stem:read(p) for p in (store/"records").glob("*.json")}
    assert set(data)==set(outcomes)
    with (store/"per-candidate.csv").open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==32
    from collections import defaultdict
    sys.path.insert(0,str(ROOT))
    from standard_match import make_structure,SETTINGS
    from pymatgen.analysis.structure_matcher import StructureMatcher
    matcher=StructureMatcher(**SETTINGS)
    formula=lambda z: tuple(zip(*_reduced_counts(z)))
    with np.load(ROOT/"scaled/mp20-data/train.npz") as train:
        tz,tk,tf=[train[k] for k in ("z","k","f")]
    catalog=defaultdict(list)
    for j,zj in enumerate(tz):catalog[formula(zj)].append(j)
    settings=read(store/"settings.json")
    primary=ROOT/"expansion/physical"
    for name,hashes in settings["reused_phases"].items():
        assert hashes["input"]==sha(store/"inputs"/(name+".json"))==sha(primary/"inputs"/(name+".json"))
        assert hashes["record"]==sha(store/"records"/(name+".json"))==sha(primary/"records"/(name+".json"))
    results={}
    for arm in ARMS[1:]:
        representatives=defaultdict(list)
        armrows=sorted([r for r in rows if r["arm"]==arm],key=lambda r:int(r["draw"]))
        assert [int(r["draw"]) for r in armrows]==list(range(1,17))
        hits=0
        with np.load(HERE/"pools"/(arm+".npz")) as b:
            for i,row in enumerate(armrows):
                key=row["id"];d=data[key];o=outcomes[key]
                live=b["z"][i]>0
                assert d["source_id"]==b["ids"][i] and int(d["index"])==i
                np.testing.assert_array_equal(d["z"],b["z"][i,live])
                np.testing.assert_allclose(d["k"],b["k"][i],rtol=0,atol=0)
                np.testing.assert_allclose(d["f"],b["f"][i,live],rtol=0,atol=0)
                if "final" not in o:
                    assert row["screened_sun"]=="False";continue
                fkey=formula(d["z"])
                assert (fkey in catalog)==(row["seen"]=="True")
                structure=make_structure(np.array(o["final_k"]),np.array(o["final_f"]),np.array(d["z"]))
                unique=all(not matcher.fit(structure,p) for p in representatives[fkey])
                if unique:representatives[fkey].append(structure)
                novel=all(not matcher.fit(structure,make_structure(tk[j],tf[j],tz[j])) for j in catalog.get(fkey,[]))
                assert unique==(row["unique"]=="True") and novel==(row["novel"]=="True")
                stem=store/"cifs"/key
                for kind in ("raw","relaxed"):
                    assert stem.with_name(key+"-"+kind+".cif").exists()
                refs=d["reference_ids"];el=np.unique(d["z"])
                if row["complete_hull"]!="True":
                    assert row["screened_sun"]=="False";continue
                assert all(outcomes[m]["converged"] for m in refs)
                E=[outcomes[m]["final"]["energy"] for m in refs]+[o["final"]["energy"]]
                comp=lambda zz: (np.asarray(zz)[:,None]==el[None,:]).mean(0)
                rhs=comp(d["z"])
                A=np.column_stack([comp(data[m]["z"]) for m in refs]+[rhs])
                sol=linprog(E,A_eq=A,b_eq=rhs,bounds=(0,None),method="highs")
                assert sol.success
                gap=E[-1]-sol.fun
                np.testing.assert_allclose(gap,float(row["above_augmented_hull"]),atol=1e-8)
                passed=o["converged"] and o["final"]["min_ratio"]>=.60 and gap<=.10 and row["novel"]=="True" and row["unique"]=="True"
                assert passed==(row["screened_sun"]=="True")
                hits+=int(passed)
        results[arm]=dict(draws=16,hits=hits)
    return results


if __name__=="__main__":
    summary=dict(pools=check_pools(),screen=quick_archive_check())
    (HERE/"per-draw-verification.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(summary)
