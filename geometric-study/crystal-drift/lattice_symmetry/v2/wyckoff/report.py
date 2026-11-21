"""Recompute protocol-C tables, bootstrap comparisons and editable subgroup/return plot."""
import argparse
import csv
import gzip
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from charts import HERE,chart
from labels import read
from geometry import TOLS,cohort

PAPER=HERE.parent/"paper"
LABELS={"prior":"Template prior","angular-A":"A-trained transfer","angular-C":"C angular drift",
        "orbit-C":"C angular + orbit","DiffCSPpp-S20":"DiffCSP++ (40 NFE)","DiffCSPpp-S999-test-128x1":"DiffCSP++ (1,998 NFE)"}
def system(g):
    for n,name in [(2,"Triclinic"),(15,"Monoclinic"),(74,"Orthorhombic"),(142,"Tetragonal"),(167,"Trigonal"),(194,"Hexagonal"),(230,"Cubic")]:
        if g<=n:return name
def csvfile(p,rows):
    with p.open("w",newline="",encoding="utf8") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def pct(x):return f"{x*100:.1f}"
def outjson(file,body):file.write_text(json.dumps(body,indent=1,allow_nan=False)+"\n")
def interval(x):return np.quantile(x,[.025,.975]).tolist()
def group_audit():
    counts={};structure=[]
    for split in ["train","val","test"]:
        r=list(read(split));good=[v for v in r if "error" not in v]
        errors=Counter(v["error"] for v in r if "error" in v)
        counts[split]=dict(total=len(r),accepted=len(good),rejects=errors,groups=len(set(v["group"] for v in good)),
                       maximum_reconstructed_site_A=max(v["err"] for v in good),
                       maximum_roundtrip_rms=max(v["roundtrip_rms"] for v in good),
                       conventional_differs=sum(v["std_n"]!=v["native_n"] for v in good))
        cnt=Counter((system(t["group"]),int(chart(t["group"],l)[2].sum())) for t in good for _,l in t["template"])
        structure.extend(dict(split=split,system=s,dof=k,orbits=n) for (s,k),n in sorted(cnt.items()))
    csvfile(HERE/"label-systems-dof.csv",structure)
    return counts

def main(partial=False):
    targets=cohort("test");selected={r["id"]:r for r in targets};rng=np.random.default_rng(917157)
    sample=rng.integers(0,128,(3000,128));small=rng.integers(0,16,(3000,16))
    summary={};hits={};groups=[];rates=[]
    for a,name in LABELS.items():
        b=HERE/f"scores/{a}.jsonl.gz";meta=HERE/f"pools/{a}.json"
        if not b.exists():
            if partial:continue
            raise FileNotFoundError(b)
        with gzip.open(b,"rt") as fi:rows=list(map(json.loads,fi))
        kmax=len(rows)//len(targets)
        assert len(rows)==kmax*128
        keyed=np.array([[r["index"],r["draw"]-1] for r in rows])
        np.testing.assert_array_equal(keyed,np.array(list(np.ndindex(128,kmax))))
        rms=np.array([np.nan if r.get("rms") is None else r["rms"] for r in rows]).reshape(128,kmax)
        arm=summary[a]=dict(metadata=json.loads(meta.read_text()),errors=sum("error" in r for r in rows),recovery={},retention={})
        hits[a]={}
        for k in (1,5,20):
            if k>kmax:continue
            sub=rms[:,:k];hit=np.isfinite(sub).any(-1);best=np.where(np.isfinite(sub),sub,np.inf).min(-1)
            hits[a][k]=hit
            arm["recovery"][k]=dict(k=k,targets=128,hits=int(hit.sum()),match_rate=hit.mean(),
                                matched_rms=float(best[hit].mean()) if hit.any() else None,mr_ci=interval(hit[sample].mean(1)))
        post=HERE/f"relaxation/{a}.json"
        for stage,data,boot in [("raw",rows,sample),("relaxed",json.loads(post.read_text()) if post.exists() else [],small)]:
            if not data:continue
            sub={}
            for tol in TOLS:
                rec={}
                for flag in ["valid","contains","direct","exact","exactops","occupancy"]:
                    col=flag if flag=="valid" else flag+str(tol)
                    x=np.array([int(r.get(col,False)) for r in data]).reshape(len(boot[0]),-1)
                    rec[flag]=int(x.sum());rec[flag+"_ci"]=interval(x.mean(1)[boot].mean(1))
                rec["n"]=len(data)
                for metric in ["contains","direct","exact","exactops","occupancy","valid"]:
                    rates.append(dict(method=a,stage=stage,tolerance=tol,metric=metric,passed=rec[metric],
                                      attempts=len(data),rate=rec[metric]/len(data),low=rec[metric+"_ci"][0],high=rec[metric+"_ci"][1]))
                sub[str(tol)]=rec
            arm["retention"][stage]=sub
            if stage=="relaxed":arm["relaxation"]=dict(statuses=Counter(r["status"] for r in data),wall=sum(r.get("wall",0) for r in data))
            strata=defaultdict(list)
            for r in data:
                tar=selected[r["id"]]
                strata["system",system(tar["group"])].append(r)
                dofs={int(chart(tar["group"],l)[2].sum()) for e,l in tar["template"]}
                for d in dofs:strata["contains_dof",str(d)].append(r)
            for (typ,g),v in strata.items():
                groups.append(dict(method=a,stage=stage,category=typ,stratum=g,n=len(v),valid=sum(r.get("valid",False) for r in v),
                        subgroup=sum(r.get("contains0.01",False) for r in v),exact=sum(r.get("exact0.01",False) for r in v),
                        exact_operations=sum(r.get("exactops0.01",False) for r in v),
                        occupancy=sum(r.get("occupancy0.01",False) for r in v)))
    paired={}
    if "angular-C" in hits:
        for a in hits:
            for k in hits[a]:
                d=hits[a][k].astype(float)-hits["angular-C"][k].astype(float)
                paired[f"{a}-{k}"]=dict(MR_difference=float(d.mean()),ci=interval(d[sample].mean(1)))
    audited=group_audit()
    label_sensitivity={split:{str(t):sum(r["symprec"][str(t)]!=r["symprec"]["0.01"] for r in cohort(split))
                                      for t in TOLS} for split in ("val","test")}
    outjson(HERE/"summary.json",dict(arms=summary,paired=paired,labels=audited,tolerance=label_sensitivity))
    if groups:csvfile(HERE/"strata.csv",groups)
    if rates:csvfile(HERE/"retention.csv",rates)
    if not partial:
        chart_plot(rates)
    print({a:v["recovery"] for a,v in summary.items()},flush=True)

def chart_plot(rates):
    from matplotlib.lines import Line2D
    plt.rcParams.update({"svg.fonttype":"none","font.size":8,"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})
    arms=[a for a in LABELS if a!="DiffCSPpp-S999-test-128x1"];xs=np.arange(len(arms))
    fig,ax=plt.subplots(1,2,figsize=(7.1,2.9))
    for axis,stage in zip(ax,("raw","relaxed")):
        for shift,met,color,mark in [(-.18,"contains","#156D86","o"),(0,"exactops","#7653A2","D"),(.18,"exact","#B54C35","s")]:
            v=[next(t for t in rates if t["method"]==a and t["stage"]==stage and t["tolerance"]==.01 and t["metric"]==met) for a in arms]
            y=100*np.array([u["rate"] for u in v]);ci=100*np.array([[u["low"],u["high"]] for u in v])
            axis.errorbar(xs+shift,y,yerr=np.maximum(0,np.stack([y-ci[:,0],ci[:,1]-y])),fmt=mark,color=color,ms=4,capsize=2)
        axis.set_xticks(xs,["Prior","A drift","C drift","+ orbit","DiffCSP++"],rotation=27,ha="right")
        axis.set_xlim(-.6,len(xs)-.5);axis.set_title("Raw, 20 draws per target" if stage=="raw" else "After relaxation, 16 draws")
        axis.grid(axis="y",alpha=.18)
        if stage=="raw":
            axis.set_ylim(99.58,100.04);axis.set_yticks([99.6,99.7,99.8,99.9,100])
        else:
            axis.set_ylim(-1,104);axis.set_yticks([0,25,50,75,100])
    ax[0].set_ylabel("Recovered symmetry (%)")
    fig.legend(handles=[Line2D([],[],marker="o",ls="",color="#156D86",label="Requested subgroup"),
              Line2D([],[],marker="D",ls="",color="#7653A2",label="Exact operations"),
              Line2D([],[],marker="s",ls="",color="#B54C35",label="Exact group type")],
              loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.5,1.02))
    fig.subplots_adjust(top=.78,left=.10,bottom=.26,right=.985,wspace=.20)
    fig.savefig(PAPER/"wyckoff-retention.pdf");fig.savefig(HERE/"wyckoff-retention.svg");plt.close(fig)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--partial",action="store_true");main(p.parse_args().partial)
