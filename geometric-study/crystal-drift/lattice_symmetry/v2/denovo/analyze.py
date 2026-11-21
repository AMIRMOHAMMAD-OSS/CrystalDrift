"""Rebuild the raw and physical tables from ordered draw and phase records."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.optimize import linprog

HERE=Path(__file__).resolve().parent
V2=HERE.parent
ROOT=HERE.parents[2]
ARMS=("fixed-basis","radial","angular")


def read(p):return json.loads(p.read_text(encoding="utf8"))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_csv(name,rows):
    with (HERE/name).open("w",newline="",encoding="utf8") as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def fractions(z,els):
    return (np.asarray(z)[:,None]==els).mean(0)


def raw():
    data={};rows=[];booleans={};cover={};ids=None;common_z=None
    ref_root=ROOT/"expansion/denovo"
    with np.load(ref_root/"panel/learned-drift-cases.npz") as fixed:
        fixed_ids=fixed["ids"]
    tiles=sorted((ref_root/"scores/raw/reference").glob("*.npz"))
    density={};elements={}
    for tile in tiles:
        with np.load(tile) as t:
            density.update(zip(t["ids"].tolist(),t["density"].tolist()))
            elements.update(zip(t["ids"].tolist(),t["nelem"].tolist()))
    assert len(density)==len(elements)==9046
    for arm in ARMS:
        d=read(HERE/"panel"/(arm+".json"))
        cost=read(HERE/"pools"/(arm+"-cpu.json"))
        source=HERE/"pools"/(arm+".npz")
        assert cost["data_sha256"]==digest(source)
        ckpt=ROOT/cost["checkpoint"]
        assert cost["checkpoint_sha256"]==digest(ckpt)
        with np.load(source) as bank,np.load(HERE/"panel"/(arm+"-cases.npz")) as c,np.load(HERE/"panel"/(arm+"-coverage.npz")) as v:
            np.testing.assert_array_equal(bank["ids"],c["ids"])
            np.testing.assert_array_equal(c["ids"],fixed_ids)
            assert c["ids"].shape==(10000,) and d["fingerprints_complete"]
            if common_z is None:common_z=bank["z"].copy()
            np.testing.assert_array_equal(common_z,bank["z"])
            flags={"struct_valid":c["struct_valid"],"comp_valid":c["comp_valid"],"joint_valid":c["valid"],
                   "unique":c["unique"],"structural_novel":c["structural_novel"],"valid_unique":c["unique"]&c["valid"],
                   "valid_unique_novel":c["valid"]&c["unique"]&c["structural_novel"]}
            for k,b in flags.items():assert int(b.sum())==d[k],(arm,k)
            booleans[arm]={k:x.copy() for k,x in flags.items()}
            assert int((~c["seen_reduced_formula"]).sum())==6478
            sub=c["w1_sample"]
            assert len(np.unique(sub))==1000 and flags["joint_valid"][sub].all()
            np.testing.assert_allclose(wasserstein_distance(list(density.values()),c["density"][sub]),d["w1_density"])
            np.testing.assert_allclose(wasserstein_distance(list(elements.values()),c["nelem"][sub]),d["w1_nelems"])
            success=(v["ref_struc"]<=.4)&(v["ref_chem"]<=10)
            prec=(v["generated_struc"]<=.4)&(v["generated_chem"]<=10)
            np.testing.assert_allclose([success.mean(),prec.sum()/10000],[d["coverage_recall"],d["coverage_precision"]])
            cover[arm]=success.copy()
            rows.append(dict(arm=arm,attempts=10000,checkpoint=cost["checkpoint"],sha256=cost["checkpoint_sha256"],
                             finite=int(bank["valid_output"].sum()),**{k:int(b.sum()) for k,b in flags.items()},
                             unseen_draws=6478,coverage_recall=d["coverage_recall"],coverage_precision=d["coverage_precision"],
                             density_w1=d["w1_density"],elements_w1=d["w1_nelems"],cpu_seconds=cost["elapsed_s"],
                             median_cpu_ms=float(np.median(cost["b1_ms"])),valid_unique_per_second=d["valid_unique"]/cost["elapsed_s"],
                             energy_w1=d.get("w1_formation_energy")))
        data[arm]=d
    write_csv("raw-results.csv",rows)
    paired={}
    g=np.random.default_rng(80613)
    for endpoint,a,b in [(k,booleans["angular"][k],booleans["radial"][k]) for k in ("struct_valid","joint_valid","valid_unique_novel")]+[("coverage_recall",cover["angular"],cover["radial"])]:
        dif=a.astype(int)-b
        counts=np.bincount(dif+1,minlength=3)
        boot=g.multinomial(len(dif),counts/len(dif),size=10000)
        paired[endpoint]={"sampling_unit":"held-out reference conditional on generated pools" if endpoint=="coverage_recall" else "paired draw",
                          "n":len(dif),"angular_only":int(counts[2]),"radial_only":int(counts[0]),
                          "pp":100*dif.mean(),"interval":np.quantile(100*(boot[:,2]-boot[:,0])/len(dif),[.025,.975]).tolist()}
    return rows,paired


def screen():
    base=HERE/"screen"
    primary=ROOT/"expansion/physical"
    per=list(csv.DictReader((base/"per-candidate.csv").open()))
    assert len(per)==32
    result=[]
    for arm in ("angular","radial"):
        pool=np.load(HERE/"pools"/(arm+".npz"))
        rows=[r for r in per if r["arm"]==arm]
        assert [int(r["draw"]) for r in rows]==list(range(1,17))
        refs=set()
        for i,row in enumerate(rows):
            src=read(base/"inputs"/(row["id"]+".json"))
            out=read(base/"records"/(row["id"]+".json"))
            assert src["source_id"]==pool["ids"][i]
            np.testing.assert_allclose(src["k"],pool["k"][i],rtol=0,atol=0)
            ref_ids=src["reference_ids"];refs.update(ref_ids)
            if "final" in out and row["complete_hull"]=="True":
                els=np.unique(src["z"])
                phases=[read(base/"inputs"/(k+".json")) for k in ref_ids]
                states=[read(base/"records"/(k+".json")) for k in ref_ids]
                assert all(s["converged"] for s in states)
                es=[s["final"]["energy"] for s in states]+[out["final"]["energy"]]
                a=np.stack([fractions(s["z"],els) for s in phases+[src]],axis=1)
                lp=linprog(es,A_eq=a,b_eq=fractions(src["z"],els),bounds=(0,None),method="highs")
                assert lp.success
                np.testing.assert_allclose(es[-1]-lp.fun,float(row["above_augmented_hull"]),atol=1e-7)
                hit=out["converged"] and out["final"]["min_ratio"]>=.6 and es[-1]-lp.fun<=.10 and row["novel"]=="True" and row["unique"]=="True"
                assert hit==(row["screened_sun"]=="True")
        costs=read(HERE/"pools"/(arm+"-cpu.json"))
        candidates=sum(float(r["wall_seconds"]) for r in rows)
        phases=sum(read(base/"records"/(k+".json"))["wall_seconds"] for k in refs)
        scoring=sum(float(r["postprocess_seconds"]) for r in rows)
        hits=sum(r["screened_sun"]=="True" for r in rows)
        result.append(dict(arm=arm,draws=16,hits=hits,upper=sum(r["possible_sun"]=="True" for r in rows),
            converged=sum(r["converged"]=="True" for r in rows),short=sum(bool(r["final_min_ratio"]) and float(r["final_min_ratio"])<.6 for r in rows),
            complete=sum(r["complete_hull"]=="True" for r in rows),unique_phases=len(refs),generation_cpu_s=costs["elapsed_s"],
            candidate_worker_s=candidates,phase_worker_s=phases,ranking_worker_s=scoring,
            hits_per_charged_worker_hour=3600*hits/(costs["elapsed_s"]+candidates+phases+scoring)))
        pool.close()
    write_csv("physical-results.csv",result)
    return result


def main():
    raw_rows,paired=raw()
    physical=screen()
    data=dict(raw=raw_rows,angular_minus_radial=paired,physical=physical)
    (HERE/"verified.json").write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf8")
    from write_results import write
    write(data)
    print("Verified three 10,000-draw banks and 32 fresh relaxation/hull records.")


if __name__=="__main__":main()
