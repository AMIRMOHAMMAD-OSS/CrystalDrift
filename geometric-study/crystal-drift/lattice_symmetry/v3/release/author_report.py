"""Rebuild the author's review sheet and main joint cost/quality table from frozen records."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
V3 = HERE.parent
EVAL = V3 / "evaluation"
TIMING = EVAL / "timing"
M = "contacts-medium-long"
A = "angular-medium-long"
F = "meanflow-medium-long"
U = "released-uflow"
G = "geometry-donor"

def read(path):
    return json.loads(path.read_text(encoding="utf8"))

def rows(path):
    with path.open(encoding="utf8", newline="") as stream:
        return list(csv.DictReader(stream))

def draw(data, tag, k=5, group="all"):
    return data["quality"][f"{group}/{k}/{tag}"]

def delta(data, first, second="prior", k=5, group="all"):
    return data["paired"][f"{group}/{k}/{first}--{second}"]

def interval(q, parameter="pp"):
    ci = q["ci95_pp" if parameter=="pp" else "native_ci95_pp"]
    return f"{q[parameter]:+.2f} pp [{ci[0]:+.2f}, {ci[1]:+.2f}]"

def count(q):
    return f"{q['hits_per_seed'][0]:,}/{q['n']:,} ({100*q['mr']:.2f}%)"

def source_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def headline_table(data):
    methods = [
        ("prior", "Chemical shortlist", "Symbolic orbit", "None", "None", 0, None),
        (G, "Chemical shortlist", "Training geometry", "None", "None", 0, None),
        ("angular-small-long", "Chemical shortlist", "Symbolic orbit", "Orbit", "Angular / Laplace", 218497, "small-angular-laplace-11-8pass"),
        (A, "Chemical shortlist", "Symbolic orbit", "Orbit", "Angular / Laplace", 2098297, "medium-angular-laplace-11-8pass"),
        (M, "Chemical shortlist", "Symbolic orbit", "Orbit", "Angular plus typed contacts / Laplace", 2098297, "medium-contacts-laplace-11-8pass"),
        (F, "Chemical shortlist", "Symbolic orbit", "Orbit", "MeanFlow", 2098297, "medium-meanflow-laplace-11-commonorder-8pass"),
        ("uflow-small-11-long", "Formula", "Gaussian", "Native transformer", "MeanFlow", 233868, "small-11-long"),
        ("uflow-medium-11-long", "Formula", "Gaussian", "Native transformer", "MeanFlow", 2135052, "medium-11-long"),
        (U, "Formula", "Gaussian", "Released transformer", "MeanFlow", 131215116, None),
    ]
    latency = {r["method"]:r for r in rows(TIMING/"latency.csv")}
    with np.load(TIMING/"cpu-first.npz") as d:
        cpu = {m: np.percentile(d["first_ms"][:,i],[50,95]) for i,m in enumerate(d["methods"].astype(str))}
    result=[]
    for tag, lookup, prior, backbone, descriptor, params, directory in methods:
        native = tag.startswith("uflow-")
        nfe=int(params>0)
        fit=read(V3/("uflow" if native else "models")/"fits"/directory/"complete.json") if directory else {}
        exposures=fit.get("rows",fit.get("anchor_exposures",""))
        seconds=fit.get("seconds","")
        selected=exposures if native else 13184 if directory else ""
        with np.load(TIMING/(tag+"-10k.npz")) as d:
            fixed=float(d["total_seconds"])
        for k in (1,5,20):
            q=draw(data,tag,k)
            result.append(dict(targets=q["n"],dataset="official MP-20",method=tag,inference_input="unit-cell formula",
                lookup=lookup,prior=prior,backbone=backbone,training_field=descriptor,registered_parameters=params,
                k=k,S=nfe,kS=k*nfe,completed_training_anchors=exposures,selected_training_anchors=selected,
                training_GPU_hours=round(float(seconds)/3600,6) if seconds!="" else "",training_time_source="complete.json" if fit else "",
                hits=q["hits_per_seed"][0],MR=q["mr"],mean_best_normalized_RMS=q["rms_matched"],
                native_volume_MR=q["native"],max_site_MR=q["strict"],error_draws=q["errors"],
                cpu_p50_ms=float(cpu[tag][0]),cpu_p95_ms=float(cpu[tag][1]),gpu_p50_ms=float(latency[tag]["first_median"]),
                gpu_p95_ms=float(latency[tag]["first_p95"]),gpu_10000_s=fixed,batch32_draws_per_s=10000/fixed,
                score_file=data["source"][tag][0]))
    with (HERE/"MAIN-COMPARISON.csv").open("w",newline="",encoding="utf8") as stream:
        w=csv.DictWriter(stream,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    return result

def author_review():
    test,family,shared,unseen=[read(EVAL/"reports"/(p+".json")) for p in ("test","family","shared","arrangement")]
    costs=headline_table(test)
    c=next(r for r in costs if r["method"]==M and r["k"]==5)
    cu=next(r for r in costs if r["method"]==U and r["k"]==5)
    at100={r["method"]:r for r in rows(TIMING/"elapsed.csv") if float(r["ms"])==100}
    tpair=next(r for r in rows(TIMING/"paired.csv") if float(r["ms"])==100 and r["first"]==M and r["second"]==U)
    high=draw(test,M,5,"large")
    physical = next(r for r in rows(V3/"denovo/screen/by-arm.csv") if r["group"]=="all" and r["arm"]==M)
    dng = {tag:read(V3/"denovo/results"/(tag+".json")) for tag in ["prior",M,U]}
    main=delta(test,M); h=delta(family,"contacts-small-11"); sha=delta(shared,"contacts-small-11")
    available=[m for m in test['source'] if not m.endswith('-3seeds') and ('all/5/'+m) in test['quality']]
    matched_orbit=[m for m in available if m.startswith(('angular-','contacts-','reciprocal-','meanflow-','sitegraph-','radial4-','reference-scale-'))]
    best_hits=max(draw(test,m)['hits_per_seed'][0] for m in matched_orbit)
    tied=sorted(m for m in matched_orbit if draw(test,m)['hits_per_seed'][0]==best_hits)
    donor_update=draw(test,'geometry-donor-angular')
    pro=[
        "# CrytalDrift author review",
        f"**What survives the prior.** On all 9,046 public MP-20 compositions, the selected 2M contact model recovered {count(draw(test,M))} at five draws. Returning the identical symbolic starts recovered {count(draw(test,'prior'))}. There were {main['first_only'][0]} gains and {main['second_only'][0]} losses, a net change of {interval(main)}. Ordinary angular drift recovered {count(draw(test,A))} and same-backbone MeanFlow recovered {count(draw(test,F))}. Adding contacts to angular drift changed recovery by {interval(delta(test,M,A))}. The new descriptor supplies a verifiable invariant construction. This comparison finds little practical extra recovery from its contact channels.",
        f"**Capacity and transfer of coordinates.** The 136M contact fit matched the 2M fit at {draw(test,'reference-scale-11')['hits_per_seed'][0]:,} recoveries. Requiring native-volume agreement reduced the 136M count to {draw(test,'reference-scale-11')['native_hits'][0]:,}. The highest five-draw count across frozen orbit and graph fits using symbolic starts was {best_hits:,}/9,046 ({', '.join(tied)}). Direct transfer of complete training geometries recovered {count(draw(test,G))}. Applying an angular network to those coordinates recovered {count(donor_update)}.",
        f"**Volume and larger cells.** Requiring agreement within the same multiplicative factor of 1.10 on physical volume gives {draw(test,M)['native_hits'][0]:,} contact hits against {draw(test,'prior')['native_hits'][0]:,} prior hits, a change of {interval(main,'native_pp')}. On {high['n']:,} cells with at least nine sites, scale-enabled MR@5 changed by {interval(delta(test,M,group='large'))}. Conditional displacement and the separate maximum-site decision are in MAIN-COMPARISON.csv.",
        f"**Controls and time.** Released uFlowCSP recovered {count(draw(test,U))} at five draws. The contact-minus-released gap was {interval(delta(test,M,U))}. On the 64-target interleaved RTX 3060 Laptop clock at 100 ms per target, contacts completed {at100[M]['completed_draws']} proposals and recovered {at100[M]['hits']} targets. Released uFlowCSP completed {at100[U]['completed_draws']} and recovered {at100[U]['hits']}, a paired difference of {float(tpair['pp']):+.2f} pp [{float(tpair['low']):+.2f}, {float(tpair['high']):+.2f}]. Retrieval of full training geometries recovered {at100[G]['hits']}/64. CPU batch-one medians were {c['cpu_p50_ms']:.2f} and {cu['cpu_p50_ms']:.2f} ms for contacts and released uFlowCSP. The corresponding GPU 10,000-proposal workloads took {c['gpu_10000_s']:.2f} and {cu['gpu_10000_s']:.2f} s. The clock includes lookup and host-side output.",
        f"**Training and transfer.** The 2,098,297-parameter contact fit processed {c['completed_training_anchors']:,} training anchors over eight traversals of 26,419 retained structures. Validation selected its {c['selected_training_anchors']:,}-anchor checkpoint. At five draws the small contact model recovered {count(draw(family,'contacts-small-11'))} on 24 withheld families against {count(draw(family,'prior'))} ({interval(h)}). None of those families had its exact occupied pattern in the donor bank. Shared families with held-out chemistries gave {interval(sha)}. The exploratory previously unmatched arrangements gave {count(draw(unseen,'contacts-small-11'))} and the same result for their prior.",
        f"**Generated structures.** Seven methods each produced 10,000 candidates at the same sampled training compositions. The valid, unique and novel counts were {dng['prior']['valid_unique_novel']:,} for the symbolic prior, {dng[M]['valid_unique_novel']:,} for the 2M contact update and {dng[U]['valid_unique_novel']:,} for released uFlowCSP. On eight prospectively selected draws, the contact model had {physical['sun_lower']}/8 joint successes under the restricted CHGNet screen. The illustrated threshold-passing SrPbS2 relaxes to a training donor. The screen measures surrogate relaxation against a finite phase library.",
        "**Author decision.** The paired improvement over the symbolic seed, its loss on physical volume, and the weak transfer to unrepresented families jointly support a paper on what a one-call periodic update learns. Direct geometry retrieval and the restricted physical screen constrain the discovery claim. Validating discrete occupations and native cell volume on unseen families is a more discriminating next experiment than another descriptor on the same starting distribution.",
    ]
    if 'all/20/contacts-small-3seeds' in family['quality']:
        q=draw(family,'contacts-small-3seeds',20);z=draw(family,'prior',20);d=delta(family,'contacts-small-3seeds',k=20)
        j=next(i for i,para in enumerate(pro) if para.startswith('**Training and transfer.'))
        pro[j]+=f" At 20 draws the small contact fits recovered {'/'.join(map(str,q['hits_per_seed']))} versus {z['hits_per_seed'][0]} for the prior ({interval(d)})."
    v=next(r for r in rows(V3/'geometry/audits/seed-volume-pattern.csv') if r['method']==M and r['k']=='5' and r['training_match']==r['availability']=='all')
    j=next(i for i,paragraph in enumerate(pro) if paragraph.startswith('**Volume and larger cells.'))
    pro[j]+=f" Restoring each starting proposal's own volume while keeping updated sites and shape recovered {int(v['seed_volume']):,} native hits."
    (HERE/"MORNING.md").write_text("\n\n".join(pro)+"\n",encoding="utf8")
    claim_sources={
        "Official, equal-draw and native-volume recovery": ["evaluation/reports/test.json","evaluation/long-frozen.json","release/MAIN-COMPARISON.csv"],
        "Protection, pairing and shortlist availability":["protocol/splits.json","protocol/analysis-plan.md","evaluation/reports/family.json","evaluation/reports/shared.json","geometry/audits/reachable-family.json"],
        "Anonymous arrangements and subgroup sensitivity":["geometry/audits/arrangement-frozen.json","evaluation/reports/arrangement.json","geometry/audits/matched-group-tolerances-family.json","geometry/audits/supplied-confirm/result.json"],
        "Actual time, draw cap, calls and fixed workload":["evaluation/timing/settings.json","evaluation/timing/elapsed.csv","evaluation/timing/paired.csv","evaluation/timing/cpu-first.npz","release/verified-clocks.json"],
        "Exposure, selection and stopping":["models/fits/medium-contacts-laplace-11-8pass/complete.json","models/fits/medium-contacts-laplace-11-8pass/selection.json","models/fixed-training-cost.csv","models/paired-anchor-order.json","uflow/native-three-seed.json"],
        "Physical features, finite banks, motif collisions and symmetry":["geometry/supplement.tex","geometry/occupancy-supplement.tex","geometry/conditioned-supplement.tex","models/bank-diagnostics.json","models/native-chart-roundtrip.json","uflow/native-symmetry.json"],
        "Empirical-composition generation and the illustrated structures":["denovo/results/"+M+".json","denovo/paired.csv","denovo/screen/by-arm.csv","geometry/crystal-gallery-extended.json"],
        "Train-only retrieval and secondary ranking":["evaluation/duplicates.csv","geometry/audits/ranking-paired.json","geometry/audits/seed-volume-pattern-paired.csv"],
    }
    hashes={}
    if (V3/"uflow/published-paired.json").is_file():
        claim_sources["Released uFlowCSP and its exact original zero-call proposals"]=["uflow/published-paired.json","uflow/published-prefix-replay.json","uflow/scores/published-prior/frozen/test/outcomes.npz"]
    if (HERE/"SEED-RESULTS.csv").is_file():
        claim_sources["Independent fits and identical initial noise"]=["release/SEED-RESULTS.csv","uflow/verified-noise-identity.json"]
    audit=["# Claim-to-evidence crosswalk","Each path starts at lattice_symmetry/v3. The panel reduction lists its contributing ordered score files under source. Table fractions retain unmatched targets in their denominator and use only recovered targets for displacement."]
    for claim,paths in claim_sources.items():
        audit.extend(["## "+claim,""])
        for path in paths:
            p=V3/path
            if not p.exists() and path=="denovo/paired.csv":p=V3/"denovo/results/paired.csv";path=p.relative_to(V3).as_posix()
            hashes[path]=source_hash(p)
            audit.append(f"- [{path}](../{path})")
        audit.append("")
    (HERE/"CLAIM-AUDIT.md").write_text("\n".join(audit)+"\n",encoding="utf8")
    (HERE/"evidence-index.json").write_text(json.dumps(hashes,indent=2)+"\n",encoding="utf8")

if __name__=="__main__":
    author_review()
