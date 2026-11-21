"""Write the manuscript's empirical sections from final target-level reductions."""
import csv
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from write import HERE,V3,ROOT,table,interval,literature

EVAL=V3/"evaluation"
DEN=V3/"denovo"
MAIN="contacts-medium-long"
FLOW="meanflow-medium-long"
NATIVE="released-uflow"

def read(p):return json.loads(p.read_text(encoding="utf8"))
def text(name,s):(HERE/name).write_text(s.rstrip()+"\n",encoding="utf8")
def number(x):return f"{x:.2f}"
def pct(x):return number(100*x)
def r3(x):return f"{x:.3f}" if x is not None else "--"
def rows(path):
    with path.open(encoding="utf8") as f:return list(csv.DictReader(f))
def draw(rep,tag,k=5,stratum="all"):return rep["quality"][f"{stratum}/{k}/{tag}"]
def pair(rep,a,b="prior",k=5,stratum="all"):return rep["paired"][f"{stratum}/{k}/{a}--{b}"]
def count(q):return f"{q['hits_per_seed'][0]:,}/{q['n']:,}"
def ci_text(d):
    return f"{d['pp']:+.2f} percentage points (95\\% interval {d['ci95_pp'][0]:+.2f} to {d['ci95_pp'][1]:+.2f})"
def figure(file,caption,label,width=r"\linewidth"):
    return r"\begin{figure}[htbp]\centering\includegraphics[width="+width+"]{"+file+r".pdf}\caption{"+caption+r"}\label{"+label+r"}\end{figure}"
def mixed_table(rep,settings,clock):
    latency={v["method"]:v for v in rows(clock/"latency.csv")}
    cpu=np.load(clock/"cpu-first.npz");names=cpu["methods"].astype(str).tolist()
    clat={name:float(np.median(cpu["first_ms"][:,i])) for i,name in enumerate(names)}
    points=[("Symbolic prior","prior",0),("Training geometry","geometry-donor",0),("Angular 0.2M","angular-small-long",1),
            ("Angular 2M","angular-medium-long",1),("Contact 2M",MAIN,1),("Orbit MeanFlow 2M",FLOW,1),
            ("uFlowCSP-small$^*$","uflow-small-11-long",1),("uFlowCSP-medium$^*$","uflow-medium-11-long",1),
            ("Released uFlowCSP",NATIVE,1)]
    rr=[]
    for name,m,S in points:
        source=clock/(m+"-10k.npz")
        with np.load(source) as x:minutes=float(x["total_seconds"])/60
        for k in (1,5,20):
            q=draw(rep,m,k)
            rr.append([name if k==1 else "",k,S,k*S,pct(q["mr"]),r3(q["rms_matched"]),
                number(clat[m]) if k==1 else "",f"{float(latency[m]['first_median']):.1f}" if k==1 else "",number(minutes) if k==1 else ""])
    return table("lrrrrrrrr",["Method","$k$","$S$","$kS$","MR (\\%)","RMS","CPU ms","GPU ms","10k min"],rr,
        "Raw formula-only prediction on all 9,046 MP-20 targets. $^*$uFlowCSP-small and uFlowCSP-medium were retrained by us. RMS averages the best normalized displacement among recovered targets. CPU and GPU columns give warm median first-proposal latency through native host arrays on the same 64 formulas. The last column gives GPU minutes for a fixed 10,000-candidate workload at batch 32. New networks use their frozen seed-11 choices from the four-look exposure experiment.","tab:main")

def controls(rep,working=False):
    ss=[("Symbolic prior","prior",0),("Untrained orbit map","untrained-small",1),("Angular 0.2M","angular-small-3seeds",1),
        ("Contacts 0.2M","contacts-small-3seeds",1),("Reciprocal 0.2M","reciprocal-small-11",1),
        ("Angular 2M","angular-medium-3seeds",1),("Contacts 2M","contacts-medium-3seeds",1),
        ("Strict orbit MeanFlow 2M","meanflow-medium-strict",1),("Training geometry","geometry-donor",0),
        ("Training geometry + angular","geometry-donor-angular",1),("Frequency ranking","empirical-rank",0),
        ("Learned ranking","learned-rank","$1/k$"),("Ranking + angular","learned-rank-angular","$1+1/k$"),
        ("Released angular","released-angular",1),("Local 16-step flow","matched16",16)]
    out=[]
    for name,m,nfe in ss:
        if working and m.endswith("-3seeds") and "all/5/"+m not in rep["quality"]:
            m=m.removesuffix("-3seeds")+"-11"
        q=draw(rep,m);q20=draw(rep,m,20)
        out.append([name,nfe,q["seeds"],pct(q["mr"]),pct(q20["mr"]),r3(q20["rms_matched"]),pct(q["native"])])
    return table("lrrrrrr",["Pipeline","$S$","Fits","MR@5","MR@20","RMS@20","Native@5"],out,
        "Seed, capacity and information controls on the official split. Percentages average the independently fitted weights in each row at the same target and draw count. Native requires a hit and physical-volume agreement on that same draw. The learned ranker and refiner each contribute one evaluation when combined.","tab:control")

def de_novo():
    arms=[("Prior","prior"),("Angular 0.2M","angular-small"),("Contact 0.2M","contacts-small"),
        ("Angular 2M","angular-medium-long"),("Contact 2M","contacts-medium-long"),("Released angular","released-angular"),
        ("Released uFlowCSP","released-uflow")]
    qq={m:read(DEN/"results"/(m+".json")) for _,m in arms}
    rr=[]
    for name,m in arms:
        q=qq[m];assert q["n"]==10000
        rr.append([name,f"{q['struct_valid']:,}",f"{q['joint_valid']:,}",f"{q['unique']:,}",
                   f"{q['novel']:,}",f"{q['valid_unique_novel']:,}",pct(q["coverage_recall"]),pct(q["coverage_precision"]),
                   r3(q["w1_density"])])
    ref=qq["prior"];q=qq["contacts-medium-long"]
    return [
        r"\section{Unconditional sampling at training compositions}",
        r"We draw an exact native-cell formula uniformly over retained training records, then draw its crystal. This empirical law preserves the observed frequency of each composition. It assigns zero mass to unknown compositions. Every method receives the same 10,000 sampled formulas in the same order. The materials question is whether a proposed arrangement meets structural and chemical criteria at those compositions.",
        table("lrrrrrrrr",["Generator","Struct.","Joint","Unique","Novel","VUN","COV-R","COV-P","$W_1(\\rho)$"],rr,
        "Raw de novo results. Each of the first five numeric columns counts successes among 10,000 ordered proposals, including failures. VUN requires joint validity, uniqueness and novelty together. Coverage percentages use CrystalNN and Magpie fingerprints. Density distance is in g/cm$^3$. The source data include the number-of-elements distance and all selected property indices.","tab:dng"),
        f"Of 10,000 proposals, {q['comp_valid']:,} have compositions passing the shared chemical validity rule. The symbolic prior produced {ref['valid_unique_novel']:,} valid, unique and novel structures. The extended contact model produced {q['valid_unique_novel']:,}. The 10,000 input formulas and their element counts agree exactly across methods. The number-of-elements Wasserstein distances of {r3(ref['w1_elements'])} and {r3(q['w1_elements'])} describe the independently sampled sets of 1,000 valid structures for which each method returned fingerprints. Raw novelty uses direct matching against {q['novelty_catalog']:,} eligible training structures.",
        r"\input{physical-results}"
    ]

def compose(working=False):
    rep=read(EVAL/"reports/test.json");fam=read(EVAL/"reports/family.json")
    assert len(rep["ids"])==9046 and len(fam["ids"])==228
    clock=EVAL/"timing";ts=read(clock/"settings.json")
    a=draw(rep,MAIN);p=draw(rep,"prior");b=draw(rep,"angular-medium-long");f=draw(rep,FLOW);u=draw(rep,NATIVE)
    primary=pair(rep,MAIN);kernel=pair(rep,MAIN,"angular-medium-long");gap=pair(rep,MAIN,NATIVE)
    h=pair(fam,"contacts-small-11");t100={r["method"]:r for r in rows(clock/"elapsed.csv") if float(r["ms"])==100}
    vol=next(r for r in rows(V3/'geometry/audits/seed-volume-pattern.csv') if r['method']==MAIN and r['k']=='5' and r['training_match']==r['availability']=='all')
    elapsed_pair=next(r for r in rows(clock/"paired.csv") if r["first"]==MAIN and r["second"]==NATIVE and float(r["ms"])==100)
    abstract=(r"One neural evaluation can predict a periodic cell and all its sites, but a structural prior can recover targets before the network is called. "
        r"We adapt training-time attraction and repulsion to a formula-conditioned generator that jointly predicts a lattice and its occupied sites. Its periodic angular field admits two separately tested additions, typed inter-orbit contacts and reciprocal-space intensities. "
        f"On the 9,046-target MP-20 test split, five proposals recovered {pct(p['mr'])}\\% from the symbolic prior and {pct(a['mr'])}\\% after a contact-trained 2M-parameter update, a gain of {ci_text(primary)}. "
        f"The angular update reached {pct(b['mr'])}\\%, same-backbone MeanFlow {pct(f['mr'])}\\%, released uFlowCSP {pct(u['mr'])}\\% and direct training-geometry retrieval {pct(draw(rep,'geometry-donor')['mr'])}\\%. "
        f"At 100 ms per formula on a single GPU, the contact update recovered {t100[MAIN]['hits']}/64 targets, released uFlowCSP {t100[NATIVE]['hits']}/64, and zero-call retrieval of training geometries {t100['geometry-donor']['hits']}/64. "
        f"With complete prototype families withheld, the seed-11 small contact update recovered 49/228 against 46/228 for its prior. "
        f"Contact and ordinary angular drift give nearly identical recovery. Restoring starting volumes to the contact model raises native-volume-qualified hits from {int(vol['learned_volume']):,} to {int(vol['seed_volume']):,}, above {p['native_hits'][0]:,} for the symbolic prior.")
    text("abstract.tex",abstract)
    text("lead-figure.tex",figure("operating-points","Recovery as a function of candidate count and actual elapsed proposal time. The same 64 official targets enter both panels. Method order is interleaved, and each episode stops after 20 completed attempts. Scoring takes place after the timers stop.","fig:time"))
    section=[r"\clearpage\section{Recovery at equal draws and elapsed time}",
        f"At five draws the selected contact model gained {primary['first_only'][0]:,} previously missed targets and lost {primary['second_only'][0]:,} hits from its symbolic prior. Its net change was {ci_text(primary)}. The contact--angular contrast on the same capacity was {ci_text(kernel)}. Relative to the released uFlowCSP replay, its five-draw change was {ci_text(gap)}. Among {gap['common_counts'][0]:,} targets both methods recovered, their mean best normalized displacements were {r3(gap['common_rms_a'])} and {r3(gap['common_rms_b'])}.",
        f"At 100 ms, the contact pipeline completed {int(t100[MAIN]['completed_draws']):,} draws over 64 formulas and recovered {t100[MAIN]['hits']}. The released native comparator completed {int(t100[NATIVE]['completed_draws']):,} and recovered {t100[NATIVE]['hits']}. The paired recovery difference was {float(elapsed_pair['pp']):+.2f} points (95\\% interval {float(elapsed_pair['low']):+.2f} to {float(elapsed_pair['high']):+.2f}). A completed first proposal includes the formula lookup, neighbor construction, transfers and decoded host arrays. Matching and relaxation run separately.",
        mixed_table(rep,ts,clock),r"\section{What the update adds to its seed}",controls(rep,working),
        f"Full-geometry transfer remained competitive after excluding the {len(rep['ids'])-draw(rep,'geometry-donor',5,'no_tight_training_match')['n']} targets with a tight training match. It recovered {count(draw(rep,'geometry-donor',5,'no_tight_training_match'))} at five draws. Within the full official split, its success counts were {count(draw(rep,'geometry-donor',5,'seen'))} for exact compositions represented in the protected training rows and {count(draw(rep,'geometry-donor',5,'unseen'))} for the other compositions. Element reassignment within a retrieved prototype can transfer a packing to a different composition.",
        figure("increments","Five-draw recovery changes after subtracting the corresponding zero-call symbolic prior. Intervals resample targets, preserving the pairing of all models and draws.","fig:increments"),
        figure("sites","Twenty-draw recovery across site-count strata on the official test split.","fig:sites"),
        r"\input{training-results}",r"\input{transfer-results}",r"\input{related}",*de_novo()]
    text("results.tex","\n\n".join(section))
    text("final-appendices.tex","\n".join(r"\input{"+name+"}" for name in ("kernel-results","replication-results","physical-diagnostics","clock-detail","literature-detail","scale-details")))
    large=pair(rep,MAIN,k=5,stratum="large")
    published=V3/"uflow/published-paired.json"
    native_evidence=""
    if published.is_file():
        q=read(published)['metrics']['20'];m=q['scale'];n=q['native']
        native_evidence=(f"The released transformer recovered {m['released_hits']:,} targets at 20 draws, compared with {m['zero_hits']:,} from its exact archived Gaussian starts. Requiring physical-volume agreement gave {n['released_hits']:,} and {n['zero_hits']:,}. Its learned contribution is consequently large even after allowing uniform rescaling in the main matcher. ")
    text("discussion.tex",r"\section{Discussion}"+ "\n\n"+
        f"The engineering target was a three-point gain over an identical symbolic seed at five draws. Our extended contact estimate was {ci_text(primary)} on the official distribution and {ci_text(large)} on cells with at least nine sites. For the controlled first-traversal small network, withholding prototype families reduced the estimate to {ci_text(h)}. A scale-enabled match on that challenge can occur even when the requested occupied pattern differs from the reference. "+
        f"Keeping the learned sites and cell shape while restoring each seed's volume changed native-volume-qualified recovery by {float(vol['seed_volume_minus_learned_pp']):+.2f} points (95\\% interval {float(vol['seed_volume_minus_learned_low']):+.2f} to {float(vol['seed_volume_minus_learned_high']):+.2f}). This sensitivity uses a volume available from the starting proposal. It identifies a volume-calibration failure alongside the geometric improvement. "+
        f"Full training geometries supplied the strongest starting point in the time experiment, recovering {t100['geometry-donor']['hits']}/64 targets at 100 ms without a network call. This result depends on the donor catalog and on how often the relevant families occur in training. On the protected family panel none of the 228 targets had its exact occupied pattern in that catalog. "+
        r"The mathematical results establish invariance of the stated feature maps and subgroup containment of the decoder under its chart assumptions. Finite feature collisions, incomplete discrete support and sparse composition-specific reference banks remain limitations of the learned field. "+
        native_evidence+r"The same-backbone experiment measures the effect of the training objective at a fixed seed and decode path. Native uFlowCSP also changes the architecture, training exposure, initialization and supervision. These changes belong to the pipeline comparison. Physical-volume qualification, retrospective relaxation and subgroup recovery answer additional questions that scale-enabled recovery alone leaves open.")
    literature()

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--working",action="store_true")
    compose(parser.parse_args().working)
