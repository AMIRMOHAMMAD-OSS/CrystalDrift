"""Typeset auxiliary results from the complete per-case reductions and verified CIF archives."""
import csv
import gzip
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;OUT=HERE/"audits";CAT=HERE.parent/"denovo/catalog"
read=lambda f:json.loads(f.read_text())
def tex_int(n):return f"{int(n):,}"
def scientific(value):
    mantissa,exponent=f"{value:.2e}".split("e")
    return rf"\({mantissa}\times10^{{{int(exponent)}}}\)"
def rng(value,lo,hi):return f"{float(value):+.2f} [{float(lo):+.2f}, {float(hi):+.2f}]"
def row(*cols):return " & ".join(str(c) for c in cols)+r"\\"

def occupation_comparison():
    v2=HERE.parents[1]/"v2/wyckoff"
    historic=read(v2/"full/summary.json")
    with np.load(v2/"full/one-draw.npz") as d:
        assert d["arms"].tolist()==["prior","angular-C"] and len(set(d["ids"]))==9046
        r=d["rms"];h=np.isfinite(r).sum(axis=0);m=np.nanmean(r,axis=0)
    assert np.array_equal(h,historic["hits"])
    np.testing.assert_allclose(m,historic["mean_matched_rms"])
    summary=read(v2/"summary.json")["arms"];recovery={}
    for tag in ("prior","angular-C","DiffCSPpp-S20"):
        seen={}
        with gzip.open(v2/f"scores/{tag}.jsonl.gz","rt") as source:
            for line in source:
                rec=json.loads(line);values=seen.setdefault(rec["id"],[])
                assert rec["draw"]==len(values)+1
                values.append(rec.get("rms") is not None and np.isfinite(rec["rms"]))
        assert len(seen)==128 and all(len(x)==20 for x in seen.values())
        recovery[tag]=sum(any(draws) for draws in seen.values())
        assert recovery[tag]==summary[tag]["recovery"]["20"]["hits"]
    lo,hi=historic["paired_95"]
    lines=[r"\paragraph{What is known during generation.}",
           r"In formula-only CSP the Cartesian Gaussian arm receives counts and noise. The symbolic arm infers an occupied pattern from a protected training catalog and samples its continuous geometry. The full-geometry donor arm transfers an actual training arrangement. Table~\ref{tab:main} compares all three at the same draw budgets. In a fourth diagnostic the target itself supplies its canonical Hall setting and element-labelled occupied letters. The target's continuous parameters and volume remain withheld.",
           f"In the preceding periodic release the supplied-occupation prior and its matched one-call angular update recovered "
           f"{tex_int(h[0])} and {tex_int(h[1])} of 9,046 official targets in one draw. Mean matched RMS was "
           f"{m[0]:.3f} and {m[1]:.3f}. The paired difference was {100*historic['paired_angular_minus_prior']:+.2f} points "
           f"(95\\% interval {100*lo:+.2f} to {100*hi:+.2f}), with {historic['rejected']} rejected labels counted as misses. "
           f"On a separate 128-target supplied-occupation panel, twenty draws recovered {recovery['prior']}, "
           f"{recovery['angular-C']} and {recovery['DiffCSPpp-S20']} targets with the prior, the angular model and "
           f"the released 40-call DiffCSP++ sampler, respectively. These are earlier conventional-cell weights and seeds."]
    newer=OUT/"supplied-confirm/result.json"
    if newer.is_file():
        report=read(newer)
        lines.append("We subsequently applied the frozen medium angular and contact models to the 96 confirmation "
                     "targets with their inferred or supplied occupations. Formula-only draws visit up to five chemical "
                     "shortlist entries. All five supplied draws use the same true occupation. The protected cell prior "
                     "and random-coordinate sampler are the same, but the conditional cell donor can differ. Each model refines exactly "
                     "the zero-call seeds in its own regime. All five attempts per target retain their position.")
        lines.append(r"\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{4pt}"+"\n"+
            r"\caption{Exploratory input-information sensitivity for frozen weights on 96 confirmation targets. Native is a scale-enabled match within a factor 1.1 of the physical target volume on the same draw. RMS@5 averages the best normalized displacement among recovered targets.}\label{tab:supplied-confirm}"+"\n"+
            r"\begin{tabular}{llrrrr}\hline"+"\n"+row("Occupation","Update","Hits@1","Hits@5","Native@5","RMS@5")+r"\hline")
        for regime in ("inferred","supplied"):
            for tag,label in (("prior","Seed"),("angular-medium-long","Angular"),("contacts-medium-long","Contacts")):
                ends=report["endpoints"][regime+"/"+tag]
                lines.append(row(regime.capitalize(),label,ends["1"]["hits"],ends["5"]["hits"],
                                 ends["5"]["native"],f"{ends['5']['best_rms']:.3f}"))
        lines.append(r"\hline\end{tabular}\end{table}")
        delta=report["paired"]["supplied/contacts-medium-long minus supplied/prior"]["5"]
        info=report["paired"]["supplied/contacts-medium-long minus inferred/contacts-medium-long"]["5"]
        err=report["endpoints"]["supplied/prior"]["5"]["errors"]
        lines.append(f"The contact update changed five-draw recovery over its identical supplied seed by {delta['pp']:+.2f} points "
            f"(95\\% interval {delta['interval'][0]:+.2f} to {delta['interval'][1]:+.2f}). Supplying the occupation changed "
            f"the frozen contact model's recovery by {info['pp']:+.2f} points versus formula-only inference "
            f"(95\\% interval {info['interval'][0]:+.2f} to {info['interval'][1]:+.2f}). "
            f"The supplied seed had {err} failed attempts among 480. All five belong to "
            r"\texttt{mp-1079901}. Its deposited cell has 20 atoms, whereas its supplied pattern expands to 10 "
            "in the model's primitive chart. The failure remains in the denominator.")
        disagreement=report["earlier_confirmation"]
        changed=sorted({mid for v in disagreement.values() for mid in v["ids"]})
        if changed:
            max_l=max(v["max_cell_entry_A"] for v in disagreement.values())
            max_f=max(v["max_wrapped_fractional_site_entry"] for v in disagreement.values())
            ids=", ".join(r"\texttt{"+mid+"}" for mid in changed)
            lines.append(f"The CPU rerun of inferred generation differed from the archived confirmation hit flags "
                f"on {ids}. The maximum cell-entry and wrapped fractional-site differences were "
                f"{scientific(max_l)}~"+r"\AA{}"+f" and {scientific(max_f)}. We retain both realizations. The registered "
                "checkpoint tables use their original output arrays.")
    return lines

def main():
    doc=[r"\section{Geometry learned after the template and volume choices}\label{app:conditioned-results}",
         *occupation_comparison(),r"\paragraph{Discrete selection.}"]
    rank=read(OUT/"ranking-coverage-test.json");ranking=read(OUT/"ranking-paired.json")
    doc.append(r"Reordering the same train-only candidates isolates discrete choice. On the 9,046 public test cases, "
        r"the numbers supplied with the setting-equivalent, element-labelled occupied pattern at one, five and twenty draws are "
        +", ".join("/".join(tex_int(rank["by_method"][m][str(k)]["equivalent"]) for k in (1,5,20))+
              " for "+t for m,t in (("lookup","chemical lookup"),("frequency","training frequency"),("learned","learned ranking")))+
        r". At five draws the ranker gains "+str(ranking["test"]["learned"]["5"]["ranking_only"])+
        " targets and loses "+str(ranking["test"]["learned"]["5"]["lookup_only"])+
        r" against chemical ordering. The paired increase is "+f'{ranking["test"]["learned"]["5"]["points"]:.2f}'+
        r" percentage points (95\% target-bootstrap interval "+
        " to ".join(f"{x:.2f}" for x in ranking["test"]["learned"]["5"]["ci_95_points"])+
        r"). Twenty draws exhaust the common shortlist. These counts measure availability before the continuous decoder acts.")
    sc=OUT/"selector-conditional.csv"
    if sc.is_file():
        matches={v["tag"]:v for v in csv.DictReader(sc.open()) if v["panel"]=="test" and v["k"]=="5"}
        if {"prior","learned-rank","empirical-rank"}<=set(matches):
            learned=matches["learned-rank"]
            doc.append("Sampling fresh geometry directly from the ordered patterns recovers "+
                ", ".join(tex_int(matches[t]["matched"]) for t in ("prior","empirical-rank","learned-rank"))+
                " targets for chemical, frequency and learned ordering at five draws. These results use the seed geometry "
                "without refinement. Learned ordering uses one selector call per shortlist. It changes recovery by "+
                f"{float(learned['change_vs_lookup_prior_pp']):+.2f} points (95\% interval {float(learned['low']):+.2f} "
                f"to {float(learned['high']):+.2f}) relative to chemical ordering.")
    doc.append(r"\paragraph{Recovery on protected structures.}")
    family=read(OUT/"matched-group-tolerances-family.json")
    shared=read(OUT/"matched-group-tolerances-shared.json")
    uncovered=read(OUT/"family-effects-arrangement.json")
    covered=read(OUT/"family-effects-arrangement-control.json")
    a=uncovered["contacts-small-11"]["rows"]["all"]
    ctrl=covered["contacts-small-11"]["rows"]["all"]
    names=("prior","contacts-small-11","uflow-prior-11","uflow-small-11","uflow-medium-11")
    assert all(all(v["exact"]==0 for v in family[name]["comparisons"].values()) for name in names)
    doc.append("At five draws the contact refiner's best accepted geometries recover the reference space-group type for "+
        f'0/{family["contacts-small-11"]["matches"]} excluded-family matches and '+
        f'{shared["contacts-small-11"]["comparisons"]["raw/0.01"]["exact"]}/{shared["contacts-small-11"]["matches"]} '+
        r"represented-family matches at 0.01~\AA. In the excluded-family panel the zero-call symbolic prior, "
        r"the native Gaussian prior and both reduced uFlowCSP models likewise have zero correct types among their best "
        r"accepted geometries. For all five arms the zero persists at 0.10~\AA{} and after diagnostic "
        r"rescaling to each true volume. A matcher can accept a packing whose independently detected group differs.")
    group_dir=OUT/"accepted-groups-family"
    if (group_dir/"summary.json").is_file():
        grouped=read(group_dir/"summary.json")
        tags=[f"contacts-{size}-{seed}" for size in ("small","medium") for seed in (11,23,37)]
        if all(tag in grouped for tag in tags):
            def exact_ids(tag):
                with np.load(group_dir/f"{tag}.npz") as records:
                    detected=records["detected"]
                    correct=((detected>0)&(detected==records["reference"][:,None])).any(axis=1)
                    return set(records["ids"][correct])
            seed_ids=exact_ids("prior");learned_ids=exact_ids(tags[0])
            assert all(exact_ids(tag)==learned_ids for tag in tags)
            assert all(grouped[tag]["rows"]["20"]["correct_group_on_added"]==0 for tag in tags)
            gains=sorted({grouped[tag]["rows"]["20"]["added_vs_prior"] for tag in tags})
            doc.append(r"At twenty draws we checked every accepted raw candidate at 0.01~\AA{} on the 228 "
                f"withheld-family targets. At least one candidate had the true group on {len(seed_ids)} targets for the "
                f"prior and on {len(learned_ids)} for each of the six contact fits (two sizes, three seeds). "
                f"Only {len(seed_ids&learned_ids)} target was common to both sets. None of the {min(gains)}--{max(gains)} "
                "additional targets recovered by an individual contact fit had a correct-group accepted candidate.")
    k5=ctrl["scale-5"];e=k5["paired_gain"]
    gains=(100*e["estimate"],100*e["lower"],100*e["upper"])
    doc.append("On the 25 secondary unfamiliar arrangements, prior and contact refiner recover the same targets at every "
        "reported draw budget. Their counts are "+
        "/".join(str(a[f"scale-{k}"]["prior_hits"]) for k in (1,5,20))+
        " at one, five and twenty draws. The observed zero difference comes from 25 cases. "
        "On the 96 covered controls, their five-draw recoveries are "+
        f'{k5["prior_hits"]} and {k5["learned_hits_per_seed"][0]}, a gain of {gains[0]:.2f} points '+
        r"(95\% interval "+f"{gains[1]:.2f} to {gains[2]:.2f} across occupied-pattern clusters).")
    if "contacts-small-three-seeds" in covered:
        rep=covered["contacts-small-three-seeds"]["rows"]["all"]["scale-5"]
        angular=covered["angular-small-three-seeds"]["rows"]["all"]["scale-5"]
        assert all(uncovered[arm+"-small-three-seeds"]["rows"]["all"]["scale-5"]["learned_hits_per_seed"]==[1,1,1]
                   for arm in ("contacts","angular"))
        doc.append("All three angular and all three contact fits recover exactly one unfamiliar target at five draws. "
            "On the covered controls their per-seed five-draw counts are "+
            "/".join(str(n) for n in angular["learned_hits_per_seed"])+" for angular and "+
            "/".join(str(n) for n in rep["learned_hits_per_seed"])+" for contacts.")
    doc.append(r"\paragraph{Cell volume and shortlist conditioning.}")
    doc.append(r"For candidate $j$, retain its learned fractional sites and cell shape while multiplying its "
        r"lattice by $(V_{\rm seed,j}/V_{\rm out,j})^{1/3}$. This deterministic intervention uses the same "
        r"sampled seed at inference. Isotropic rescaling preserves the mathematical scale-enabled displacement. "
        r"Directly rematching the 480 small-angular development draws confirmed the same match indicators. "
        r"The official sensitivity reuses the original scale-enabled match for each draw. "
        r"For a retained matching draw, replace its native-volume indicator by "
        r"$|\log(V_{\rm seed,j}/V_{\rm target})|\leq\log(1.1)$. Table~\ref{tab:geometry-volume} "
        r"applies this gate before taking an OR over draws. Thus every counted candidate meets both conditions.")
    def records(file):return list(csv.DictReader(file.open()))
    s=records(OUT/"seed-volume-pattern.csv");effect=records(OUT/"seed-volume-pattern-paired.csv")
    sel=lambda source,method,dup,avail:[v for v in source if v["method"]==method and v["training_match"]==dup and
                               v["availability"]==avail and v["k"]=="5"]
    tags={"prior":"Symbolic seed","angular-small-11":"Small angular","contacts-small-11":"Small contacts",
        "reciprocal-small-11":"Small reciprocal","meanflow-small-11":"Small MeanFlow","angular-medium-11":"Medium angular",
        "contacts-medium-11":"Medium contacts","meanflow-medium-11":"Medium MeanFlow","untrained-small":"Untrained small",
        "angular-small-long":"Extended search, small angular","angular-medium-long":"Extended search, medium angular",
        "contacts-medium-long":"Extended search, medium contacts","meanflow-medium-long":"Extended search, medium MeanFlow"}
    doc.extend([r"\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{4pt}",
        r"\caption{Five-draw recovery on 9,046 official targets. Counts are numbers of matched targets. "
        r"The paired difference compares seed-volume-qualified with raw-volume-qualified recovery in "
        r"percentage points, with a 95\% target-bootstrap interval. Seed 11 is used for the fits.}\label{tab:geometry-volume}",
        r"\begin{tabular}{lrrrr}\hline",row("Method","Scale","Raw volume","Seed volume","Difference [95\% interval]")+r"\hline"])
    for name,label in tags.items():
        values=sel(s,name,"all","all")
        if not values:continue
        v=values[0];doc.append(row(label,tex_int(v["scale"]),tex_int(v["learned_volume"]),tex_int(v["seed_volume"]),
                         rng(v["seed_volume_minus_learned_pp"],v["seed_volume_minus_learned_low"],v["seed_volume_minus_learned_high"])))
    doc.append(r"\hline\end{tabular}\end{table}")
    replicated={f"{arm}-{tier}-{seed}" for arm in ("angular","contacts")
                for tier in ("small","medium") for seed in (11,23,37)}
    fixed=[v for v in s if v["method"] in replicated and v["training_match"]=="all" and
           v["availability"]=="all" and v["k"]=="5"]
    if {v["method"] for v in fixed}==replicated:
        def bounds(key):
            values=[int(v[key]) for v in fixed]
            return tex_int(min(values))+"--"+tex_int(max(values))
        doc.append("Across 12 first-traversal angular and contact fits (two sizes, three paired seeds), "
            f"five-draw scale-enabled recovery was {bounds('scale')} of 9,046. Using each output cell's "
            f"native volume gave {bounds('learned_volume')} targets. Retaining the seed volume gave "
            f"{bounds('seed_volume')} targets across these same fits. These ranges describe fitted models on "
            "one fixed test panel.")
    architectures={"sitegraph-small":"widened site graph","angular-large-11":"15M orbit model",
                   "reference-scale-11":"reference-scale orbit model"}
    available=[(title,sel(s,model,"all","all")[0]) for model,title in architectures.items() if sel(s,model,"all","all")]
    if available:
        entries=[f"{tex_int(v['learned_volume'])}/{tex_int(v['seed_volume'])} for the {title}" for title,v in available]
        doc.append("For the architecture controls, five-draw native-volume/seed-volume counts were "+
                   ", ".join(entries)+". Each pair uses the same geometric proposals.")
    name=next(t for t in ("contacts-medium-long","contacts-small-11","angular-small-11","prior")
              if sel(s,t,"all","all"))
    label={"contacts-medium-long":"extended medium contact refiner","contacts-small-11":"small contact refiner",
           "angular-small-11":"small angular refiner","prior":"symbolic prior"}[name]
    f=sel(s,name,"all","setting_equivalent_reachable")[0];u=sel(s,name,"all","known_unreachable")[0]
    x=sel(s,name,"exclude_tight_train_match","setting_equivalent_reachable")[0]
    y=sel(s,name,"exclude_tight_train_match","known_unreachable")[0]
    doc.append(f"At five draws the {label} recovers {tex_int(f['scale'])}/{tex_int(f['n'])} targets with a "
        f"reachable labelled occupation. Only {tex_int(f['same_pattern_scale'])} have an accepted draw carrying that "
        f"occupation. Among {tex_int(u['n'])} known targets with no such shortlisted occupation, {tex_int(u['scale'])} "
        "nevertheless pass geometric matching. After removing 90 tight species-preserving training matches, the "
        f"respective recovery counts are {tex_int(x['scale'])}/{tex_int(x['n'])} and {tex_int(y['scale'])}/{tex_int(y['n'])}. "
        "Unassigned reference occupations remain in the full denominator.")
    if name!="prior":
        ee=next(v for v in sel(effect,name,"exclude_tight_train_match","all") if v["criterion"]=="seed_volume")
        doc.append("On these 8,956 filtered targets this seed-volume update recovers "+
            tex_int(ee["method_hits"])+" against "+tex_int(ee["prior_hits"])+" for the identical zero-call prior, a "
            "paired difference of "+f"{float(ee['pp']):+.2f} points (95\% interval {float(ee['low']):+.2f} "
            f"to {float(ee['high']):+.2f}).")
    doc.append(r"\paragraph{Symmetry of de novo proposals.}")
    g=read(CAT/"spacegroup-summary.json");b=g["arms"]["contacts-small"];p=g["arms"]["prior"]
    doc.append(r"The explicit P1 CIF representation lists every site. Independently detected symmetry can be much higher. "
        r"At 0.01~\AA{} the raw symbolic prior occupies "+str(p["distinct_positive_types"])+" detected group types "
        "and the contact update occupies "+str(b["distinct_positive_types"])+". The corresponding Jensen--Shannon "
        f"distances to the full official test distribution are {p['js_distance_base2_to_test']:.3f} and "
        f"{b['js_distance_base2_to_test']:.3f}. We use the square root of the divergence in bits and retain "
        "unsuccessful detections as one additional category. For released angular and released uFlowCSP, "+
        tex_int(g["arms"]["released-angular"]["p1"])+" and "+tex_int(g["arms"]["released-uflow"]["p1"])+
        r" of 10,000 raw proposals have type $P1$ at this tolerance.")
    learned=["angular-small","contacts-small","angular-medium-long","contacts-medium-long"]
    if set(learned)<=set(g["arms"]) and "paired_draws_with_different_detection" in g:
        paired=g["paired_draws_with_different_detection"]
        same=all(v==0 for k,v in paired.items() if set(k.split("/"))<=set(learned))
        changed={v for k,v in paired.items() if "prior" in k.split("/") and set(k.split("/"))&set(learned)}
        assert same and len(changed)==1
        cif=[read(CAT/f"{arm}.json") for arm in g["arms"]]
        assert all(a["crc_verified"] and a["cifs"]==a["attempts"]==10000 for a in cif)
        doc.append(f"The small and validation-selected medium angular and contact fits assign identical "
            f"groups on all 10,000 paired draws, including {10000-b['detected']} detection failures. "
            f"Each differs from the symbolic seed on {changed.pop()} draws. The archive preserves all "
            f"{tex_int(sum(a['cifs'] for a in cif))} CIFs across the {len(cif)} pipelines. Independent CIF "
            f"read-back checked the cell metric and wrapped sites in {sum(len(a['read_back']) for a in cif)} cases.")
    doc.append(r"\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{spacegroups.pdf}"
        r"\caption{As-generated groups compared with the official test structures and the exact training structures "
        r"whose formulas were drawn. P1 and inversion-only cells are shown individually. Other assigned types are "
        r"grouped by crystal system. Every generated row uses all 10,000 draws.}\label{fig:raw-groups}\end{figure}")
    arms=["prior","angular-small","contacts-small"]
    if all((OUT/f"subgroup-{arm}.json").is_file() for arm in arms):
        op=[read(OUT/f"subgroup-{arm}.json") for arm in arms]
        doc.append("An independent native-basis operation test on the 256 prespecified raw draws found "
            "the requested subgroup in "+", ".join(str(v["unrelaxed_hash256_summary"]["contains"]) for v in op)+
            " structures for the prior, angular and contact methods, respectively. The detected operation sets "
            "exactly equalled the requests in "+", ".join(str(v["unrelaxed_hash256_summary"]["exact_operations"]) for v in op)+
            ". All eight raw members per method in the separate relaxation panel passed the exact test. After CHGNet, "+
            ", ".join(str(v["summary"]["0.01"]["relaxed"]["exact_operations"]) for v in op)+
            r" retained exact agreement at 0.01~\AA. The post-relaxation comparison allows a common origin shift "
            r"estimated from the mean wrapped site displacement. Eight paired cases give an illustrative sensitivity.")
    medium=["angular-medium-long","contacts-medium-long"]
    if all((OUT/f"subgroup-{arm}.json").is_file() for arm in medium):
        retained=[read(OUT/f"subgroup-{arm}.json")["summary"]["0.01"]["relaxed"]["exact_operations"] for arm in medium]
        doc.append(f"The selected medium angular and contact fits retain their requested operation sets in "
                   f"{retained[0]}/8 and {retained[1]}/8 relaxed cells, respectively, at the same tolerance.")
    (HERE/"conditioned-supplement.tex").write_text("\n\n".join(doc)+"\n",encoding="utf8")

if __name__=="__main__":main()
