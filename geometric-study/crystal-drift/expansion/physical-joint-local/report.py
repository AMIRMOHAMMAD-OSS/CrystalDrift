"""Check candidate identities, recompute screening gates and costs, and write the paper fragment."""
import csv
import hashlib
import json
from pathlib import Path

from screen import ARM,HERE,POOL,PRIMARY,ROOT,digest,save
from primary_snapshot import snapshot
import numpy as np
from scipy.optimize import linprog
from pymatgen.analysis.structure_matcher import StructureMatcher
from standard_match import SETTINGS,make_structure


def truth(text):return text=="True"


def formula(z):
    elements,counts=np.unique(np.asarray(z)[np.asarray(z)>0],return_counts=True)
    if not len(elements):return ()
    return tuple(zip(elements,counts//np.gcd.reduce(counts)))


def fractions(z,els):
    return np.array([np.mean(np.asarray(z)==e) for e in els])


def check():
    # Preserve the primary comparator and count every stopped or malformed draw.
    baseline=json.loads((HERE/"primary-before.json").read_text())
    assert snapshot()==baseline
    meta=json.loads((HERE/"settings.json").read_text())
    assert meta["pool_sha256"]==digest(POOL) and meta["design_sha256"]==digest(HERE/"DESIGN.md")
    receipt=json.loads((HERE/"protocol-receipt.json").read_text())
    assert receipt["protocol_sha256"]==meta["design_sha256"] and receipt["pool_present_at_receipt"] is False
    sources={p.stem:json.loads(p.read_text()) for p in (HERE/"inputs").glob("*.json")}
    outcomes={p.stem:json.loads(p.read_text()) for p in (HERE/"records").glob("*.json")}
    assert sources.keys()==outcomes.keys()
    for n,ref in meta["reused_references"].items():
        assert digest(HERE/"inputs"/f"{n}.json")==digest(PRIMARY/"inputs"/f"{n}.json")==ref["input_sha256"]
        assert digest(HERE/"records"/f"{n}.json")==digest(PRIMARY/"records"/f"{n}.json")==ref["record_sha256"]
        assert ref["charged_worker_seconds"]==outcomes[n]["wall_seconds"]
    with (HERE/"per-candidate.csv").open(newline="") as stream:
        rows=sorted(csv.DictReader(stream),key=lambda r:int(r["draw"]))
    assert [int(r["draw"]) for r in rows]==list(range(1,17))
    with np.load(POOL) as d:
        for i,r in enumerate(rows):
            inp=sources[r["id"]];occ=d["z"][i]>0
            assert inp["source_id"]==str(d["ids"][i])==meta["source_ids"][i]
            np.testing.assert_array_equal(d["z"][i,occ],inp["z"])
            np.testing.assert_allclose(d["k"][i],inp["k"],atol=0,rtol=0,equal_nan=True)
            np.testing.assert_allclose(d["f"][i,occ],np.asarray(inp["f"]).reshape(-1,3),atol=0,rtol=0,equal_nan=True)
    with np.load(ROOT/"scaled/mp20-data/train.npz") as d:
        tz,tk,tf=d["z"],d["k"],d["f"]
    groups={}
    for i,z in enumerate(tz):groups.setdefault(formula(z),[]).append(i)
    matcher=StructureMatcher(**SETTINGS)
    representatives={}
    refs=set()
    for i,r in enumerate(rows):
        n=r["id"];inp=sources[n];rec=outcomes[n];chem=formula(inp["z"])
        assert truth(r["seen"])==(chem in groups)
        refs.update(inp["reference_ids"])
        if "final" not in rec:
            assert not truth(r["screened_sun"])
            continue
        assert (HERE/"cifs"/f"{n}-raw.cif").is_file() and (HERE/"cifs"/f"{n}-relaxed.cif").is_file()
        final=make_structure(np.array(rec["final_k"]),np.array(rec["final_f"]),np.array(inp["z"]))
        offered=representatives.setdefault(chem,[])
        first=all(not matcher.fit(final,other) for other in offered)
        if first:offered.append(final)
        novel=all(not matcher.fit(final,make_structure(tk[j],tf[j],tz[j])) for j in groups.get(chem,[]))
        assert first==truth(r["unique"]) and novel==truth(r["novel"])
        eligible=[m for m in inp["reference_ids"] if outcomes[m]["converged"] and "final" in outcomes[m]]
        elements=np.unique(inp["z"])
        terminals={sources[m]["z"][0] for m in eligible if len(set(sources[m]["z"]))==1}
        complete=len(eligible)==len(inp["reference_ids"]) and set(elements)<=terminals
        assert complete==truth(r["complete_hull"])
        energetic=False
        if complete:
            a=np.stack([fractions(sources[m]["z"],elements) for m in eligible],axis=1)
            energies=[outcomes[m]["final"]["energy"] for m in eligible]
            sol=linprog(energies,A_eq=a,b_eq=fractions(inp["z"],elements),bounds=(0,None),method="highs-ds")
            assert sol.success
            gap=rec["final"]["energy"]-sol.fun
            np.testing.assert_allclose(gap,float(r["signed_gap"]),rtol=0,atol=1e-8)
            energetic=gap<=.10
            assert energetic==truth(r["energetic_pass"])
        hit=rec["converged"] and rec["final"]["min_ratio"]>=.60 and first and novel and energetic
        assert hit==truth(r["screened_sun"])
    assert refs==set(meta["new_references"])|set(meta["reused_references"])
    strata={}
    for name in ("all","seen","unseen","undefined"):
        chosen=[r for i,r in enumerate(rows) if name=="all" or
                (name=="undefined" and not meta["defined_formulas"][i]) or
                (meta["defined_formulas"][i] and name in ("seen","unseen") and truth(r["seen"])==(name=="seen"))]
        strata[name]=dict(draws=len(chosen),hits=sum(truth(r["screened_sun"]) for r in chosen),
                          upper=sum(truth(r["possible_sun"]) for r in chosen),
                          converged=sum(truth(r["converged"]) for r in chosen),
                          valid_contact=sum(truth(r["contacts_ok"]) for r in chosen),
                          short=sum(bool(r.get("final_min_ratio")) and float(r["final_min_ratio"])<.60 for r in chosen),
                          energetic=sum(truth(r["energetic_pass"]) for r in chosen),
                          uniquely_first=sum(truth(r["unique"]) for r in chosen),
                          novel=sum(truth(r["novel"]) for r in chosen),
                          complete=sum(truth(r["complete_hull"]) for r in chosen))
    assert sum(strata[x]["draws"] for x in ("seen","unseen","undefined"))==16
    allrow=next(r for r in csv.DictReader((HERE/"by-arm.csv").open()) if r["group"]=="all")
    assert strata["all"]["hits"]==int(allrow["sun_lower"])
    refcost=sum(outcomes[n]["wall_seconds"] for n in sorted(refs))
    costs=dict(candidate_worker_s=sum(float(r["wall_seconds"]) for r in rows),
               distinct_reference_worker_s=refcost,postprocess_worker_s=sum(float(r["postprocess_seconds"]) for r in rows),
               reused_reference_worker_s=sum(w["charged_worker_seconds"] for w in meta["reused_references"].values()))
    np.testing.assert_allclose(refcost,float(allrow["reference_core_seconds"]))
    timing=POOL.with_suffix(".json")
    if timing.is_file():
        t=json.loads(timing.read_text())
        assert t["draws"]==10000 and t["batch"]==24 and t["nfe"]==100 and t["arm"]==ARM
        assert len(t["b1_ms"])==128 and np.all(np.asarray(t["b1_ms"])>0)
        np.testing.assert_allclose(t["per_second"],10000/t["elapsed_s"])
        costs.update(generation_gpu_s=t["elapsed_s"],structures_per_second=t["per_second"],
                     median_batch_one_ms=float(np.median(t["b1_ms"])))
        total=costs["candidate_worker_s"]+refcost+costs["postprocess_worker_s"]+t["elapsed_s"]
        costs.update(total_charged_s=total,hits_per_charged_hour=3600*strata["all"]["hits"]/total)
    files={p.relative_to(HERE).as_posix():digest(p) for k in ("inputs","records","cifs") for p in (HERE/k).iterdir() if p.is_file()}
    result=dict(strata=strata,number_of_phases=len(refs),reused_phases=len(meta["reused_references"]),
                new_phases=len(meta["new_references"]),converged_phases=sum(outcomes[m]["converged"] for m in refs),
                costs=costs,primary_files=baseline,protocol_receipt=receipt,sha256=files,verified_draw_ids=meta["source_ids"])
    save(HERE/"verification.json",result)
    return result


def fraction(r):
    return f"{r['hits']}/{r['draws']}" if r["draws"] else "no draws"


def paper(r):
    a=r["strata"]["all"];s=r["strata"]["seen"];u=r["strata"]["unseen"];c=r["costs"]
    paragraphs=[r"\paragraph{Independent local joint-generator screen.}",
        "Before opening the timed 10,000-draw CrystalFlow replay, we fixed draws 1--16 for a separate application of the same CHGNet protocol. "
        f"We found {a['hits']} screened hits in 16 draws ({fraction(s)} at training-seen reduced formulas and {fraction(u)} at training-unseen formulas). "
        f"Of the 16, {a['converged']} reached the force criterion and {a['short']} finished with a short contact. "
        f"The restricted-hull energy criterion passed {a['energetic']}. Within this local arm {a['uniquely_first']} were first representatives and {a['novel']} lacked a match in the deposited training structures. "
        f"Complete competing-phase coverage was available for {a['complete']}/16 cases. "
        f"The phase library comprised {r['number_of_phases']} distinct structures, including {r['reused_phases']} byte-verified reference relaxations from the primary screen."]
    if a["upper"]!=a["hits"]:paragraphs.append(f"Incomplete reference coverage gives a possible upper count of {a['upper']} hits.")
    if "generation_gpu_s" in c:
        paragraphs.append(f"Generating all 10,000 proposals took {c['generation_gpu_s']:.2f} GPU seconds ({c['structures_per_second']:.1f} structures/s at batch 24). "
            f"The 16 candidates, their distinct reference phases and the final hull/matcher analysis consumed {c['candidate_worker_s']:.1f}, {c['distinct_reference_worker_s']:.1f} and {c['postprocess_worker_s']:.2f} single-thread CPU worker seconds, respectively. "
            f"Charging the full generation workload and every distinct phase to this cohort gives {c['hits_per_charged_hour']:.2f} screened hits per summed device-hour. "
            "The deposited CrystalFlow cohort in Table~\\ref{t:screen} yielded five screened hits in a separate sequence of sixteen draws.")
    (ROOT/"paper/exp-screen-joint.tex").write_text("\n".join(paragraphs)+"\n",encoding="utf8")


if __name__=="__main__":
    result=check()
    paper(result)
    print(result["strata"],result["costs"])
