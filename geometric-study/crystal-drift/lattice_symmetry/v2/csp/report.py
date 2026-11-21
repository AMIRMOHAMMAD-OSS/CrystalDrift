"""Recompute fixed-draw and paired fixed-time endpoints from identified case records."""
import os
for v in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[v]="1"
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];PAPER=HERE.parent/"paper"
SEEDS=(11,23,37);K=(1,5,20)
NEW=("prior","radial","power0","angular","fixed-angular","flow16")
LEGACY=("old-drift","old-flow16","meanflow","uflow","retrieval")
LABEL={"prior":"Fitted prior alone","radial":"Physical, pair shells","power0":"Physical, center power 0","angular":"Physical, angular 0--2",
       "fixed-angular":"Fixed basis, angular 0--2","flow16":"Compact flow (16, matched)",
       "old-drift":"Released fixed-basis drift","old-flow16":"Released compact flow (16)","meanflow":"Compact MeanFlow",
       "retrieval":"Train-only retrieval","uflow":"Released uFlowCSP"}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def table(name,rows):
    with (HERE/name).open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def pools(ids):
    for arm in NEW:
        rows=[];deps=[]
        for seed in SEEDS:
            folder=HERE/f"test/{arm}-seed{seed}"
            deps.append(folder/"outcomes.npz")
            ready=json.loads((folder/"ready.json").read_text())
            if arm=="prior":assert sha(HERE/"prior.npz")==ready["checkpoint_sha256"]
            else:
                lock=json.loads((HERE/"selection.json").read_text())[arm]
                assert lock["selected_checkpoint_sha256"][str(seed)]==ready["checkpoint_sha256"]
            with np.load(deps[-1]) as d:
                np.testing.assert_equal(d["ids"],ids);rows.append(d["rms"])
                assert d["error"].shape==d["rms"].shape and len(rows[-1])==9046
        yield arm,rows,deps
    for arm in ("old-drift","old-flow16","meanflow"):
        method={"old-drift":"drift","old-flow16":"flow16","meanflow":"meanflow"}[arm]
        deps=[ROOT/f"expansion/training/test/{method}-seed{s}/outcomes.npz" for s in SEEDS]
        rows=[]
        for p in deps:
            with np.load(p) as d:np.testing.assert_equal(d["ids"],ids);rows.append(d["rms"])
        yield arm,rows,deps
    path=ROOT/"revision/full/external/mp20/complete-crosswalk-k20.npz"
    with np.load(path) as d:np.testing.assert_equal(d["ids"],ids);uf=d["rms"]
    yield "uflow",[uf],[path]
    path=ROOT/"revision/full/mp20/complete-crosswalk-k20.npz"
    with np.load(path) as d:
        np.testing.assert_equal(d["ids"],ids)
        rows=[d["rms"][:,d["arms"].tolist().index(f"retrieval-seed{s}")] for s in SEEDS]
    yield "retrieval",rows,[path]


def fixed_draw():
    with np.load(ROOT/"revision/full/mp20/cohort.npz") as r:ids=r["ids"]
    cases=[];seeds=[];evidence={};methods=[]
    for method,rows,files in pools(ids):
        methods.append(method);rep=[]
        evidence[method]={x.relative_to(ROOT).as_posix():sha(x) for x in files}
        for s,r in enumerate(rows):
            seed=[]
            assert r.shape==(9046,20)
            for k in K:
                best=np.where(np.isfinite(r[:,:k]),r[:,:k],np.inf).min(-1)
                hit=np.isfinite(best)
                seed.append(np.stack((hit,np.where(hit,best,0)),axis=-1))
                seeds.append(dict(method=method,seed=SEEDS[s] if len(rows)==3 else "released",k=k,
                                  hits=hit.sum(),targets=len(ids),mr=hit.mean(),mean_normalized_rmse=best[hit].mean()))
            rep.append(np.stack(seed,1))
        if len(rep)==1:rep*=3
        cases.append(np.stack(rep,1))
    a=np.stack(cases,1)
    def reduce(x):
        h=x[...,0];r=x[...,1]/np.maximum(h,1e-30)
        return np.stack((h.mean(-2),r.mean(-2)),axis=-1)
    result=reduce(a.mean(0));rng=np.random.default_rng(772501);boot=[];n=len(a)
    for i in range(0,5000,25):
        w=rng.multinomial(n,np.full(n,1/n),size=25)
        boot.extend(reduce(((w@a.reshape(n,-1))/n).reshape(-1,*a.shape[1:])))
    boot=np.array(boot);rows=[];pairs=[]
    for j,m in enumerate(methods):
        for k,b in enumerate(K):
            ci=np.quantile(boot[:,j,k],(.025,.975),axis=0)
            rows.append(dict(method=m,k=b,targets=n,mr=result[j,k,0],mr_low=ci[0,0],mr_high=ci[1,0],
                             mean_normalized_rmse=result[j,k,1],rms_low=ci[0,1],rms_high=ci[1,1]))
            d=boot[:,methods.index("angular"),k]-boot[:,j,k];lo,hi=np.quantile(d,(.025,.975),axis=0)
            exact=result[methods.index("angular"),k]-result[j,k]
            pairs.append(dict(first="angular",second=m,k=b,mr_delta=exact[0],mr_low=lo[0],mr_high=hi[0],
                              rms_delta=exact[1],rms_low=lo[1],rms_high=hi[1]))
    table("quality.csv",rows);table("paired-draws.csv",pairs);table("seeds.csv",seeds)
    evidence["released-one-euler-diagnostic"]={}
    for seed in SEEDS:
        path=ROOT/f"expansion/training/test/flow1-seed{seed}/outcomes.npz"
        evidence["released-one-euler-diagnostic"][path.relative_to(ROOT).as_posix()]=sha(path)
    return rows,pairs,evidence


def sensitivities():
    rows=[]
    for method in NEW:
        for seed in SEEDS:
            with np.load(HERE/f"test/{method}-seed{seed}/outcomes.npz") as d:
                for k in K:
                    rms=d["rms"][:,:k];largest=d["strict_fit"][:,:k].any(1)
                    native=(np.isfinite(rms)&(np.abs(d["log_volume_ratio"][:,:k])<=np.log(1.10))).any(1)
                    rows.append(dict(method=method,seed=seed,k=k,targets=len(rms),
                                     max_site_hits=largest.sum(),max_site_mr=largest.mean(),
                                     native_volume_hits=native.sum(),native_volume_mr=native.mean()))
    table("sensitivities.csv",rows)


def timing():
    methods=(*NEW,*LEGACY)
    grid=(1,2,5,10,20,30,50,80,100,150,250,500,1000,2500,5000)
    rng=np.random.default_rng(257841);ix=rng.integers(0,128,(5000,128));curves=[];cost=[];paired=[];hits={}
    for m in methods:
        p=HERE/"gpu"
        with np.load(p/f"{m}-episodes.npz") as t,np.load(p/f"{m}-rms.npz") as a,np.load(p/f"{m}-10000.npz") as w:
            np.testing.assert_equal(t["ids"],a["ids"])
            ts=t["elapsed_ms"];r=a["rms"];lat=np.diff(ts,axis=1,prepend=0)
            assert r.shape==ts.shape==(128,20) and (lat>0).all() and w["counts"].sum()==10000
            cfg=json.loads((p/f"{m}-settings.json").read_text())
            cost.append(dict(method=m,b1_ms=np.median(lat),seconds_per_10000=float(w["elapsed_s"]),
                             per_second=float(10000/w["elapsed_s"]),peak_cuda_mib=cfg["max_cuda_allocated_bytes"]/2**20))
            for axis,thresholds in (("draws",range(1,21)),("ms",grid)):
                for b in thresholds:
                    mask=ts<=b if axis=="ms" else np.broadcast_to(np.arange(20)<b,ts.shape)
                    got=(np.isfinite(r)&mask).any(1)
                    lo,hi=np.quantile(got[ix].mean(1),(.025,.975))
                    curves.append(dict(method=m,axis=axis,budget=b,mean_completed=mask.sum(1).mean(),hits=got.sum(),targets=128,mr=got.mean(),low=lo,high=hi))
                    hits[m,axis,b]=got.astype(float)
    for m in methods:
        for b in grid:
            d=hits["angular","ms",b]-hits[m,"ms",b];lo,hi=np.quantile(d[ix].mean(1),(.025,.975))
            paired.append(dict(first="angular",second=m,ms=b,delta=d.mean(),low=lo,high=hi))
    table("timing-curves.csv",curves);table("timing-paired.csv",paired);table("speed.csv",cost)
    return curves,cost


def plot(curves,filename="csp-recovery"):
    from scipy.stats import binom
    dataset=HERE/("interleaved" if filename.endswith("interleaved") else "gpu")
    events=[];event_arrays={}
    for m in sorted({r["method"] for r in curves}):
        with np.load(dataset/f"{m}-episodes.npz") as t,np.load(dataset/f"{m}-rms.npz") as s:
            ts=t["elapsed_ms"];hit=np.isfinite(s["rms"])
            arrival=np.where(hit,ts,np.inf).min(1);event_arrays[m]=np.sort(arrival)
            for i,mid in enumerate(t["ids"]):
                events.append(dict(method=m,target=mid,cap=20,first_hit_draw=int(hit[i].argmax()+1) if hit[i].any() else "",
                              first_hit_ms=float(arrival[i]) if np.isfinite(arrival[i]) else "",last_completion_ms=ts[i,-1]))
    table(("interleaved/" if filename.endswith("interleaved") else "")+"first-hit-events.csv",events)
    plt.rcParams.update({"svg.fonttype":"none","font.size":8,"axes.spines.top":False,"axes.spines.right":False})
    balanced=filename.endswith("interleaved")
    fig,axs=plt.subplots(1,2,figsize=(7.8,3.6 if balanced else 4.05))
    styles={"angular":("#006759","-"),"radial":("#bb5b15","--"),"prior":("#52525b","-."),"power0":("#6d579a",":"),"old-drift":("#245e96","-."),
            "flow16":("#86394f","-"),"uflow":("#385465","--"),"retrieval":("#697836",":")}
    if not balanced:
        styles["fixed-angular"]=("#806423",(0,(4,2,1,2)))
        styles["meanflow"]=("#685c64",(0,(1.5,2,4,2)))
    styles["old-flow16"]=("#7a4a75",":")
    present={r["method"] for r in curves}
    styles={m:s for m,s in styles.items() if m in present}
    for axis,ax in zip(("draws","ms"),axs):
        for m,(color,ls) in styles.items():
            data=[r for r in curves if r["axis"]==axis and r["method"]==m]
            label=LABEL[m].replace("--","–")
            x=[r["budget"] for r in data]
            y=np.asarray([r["mr"] for r in data]);lo=np.asarray([r["low"] for r in data]);hi=np.asarray([r["high"] for r in data])
            if axis=="ms":
                arrival=event_arrays[m]
                x=np.unique(np.r_[.03,arrival[(arrival>.03)&(arrival<=5000)],5000.])
                y=np.searchsorted(arrival,x,side="right")/len(arrival)
                lo,hi=(binom.ppf(p,len(arrival),y)/len(arrival) for p in (.025,.975))
            ax.plot(x,100*y,drawstyle="steps-post",lw=1.6 if m=="angular" else 1.1,ls=ls,color=color,label=label)
            if m=="angular":ax.fill_between(x,100*lo,100*hi,step="post",alpha=.16,color=color,lw=0)
        if axis=="ms":
            ax.set_xscale("log");ax.axvline(100,color="#888888",lw=.7,zorder=0)
        else:ax.set_xticks([1,5,10,15,20])
        ax.set_xlabel("Ordered candidates per target" if axis=="draws" else "Elapsed proposal time per target (ms)")
        ax.set_ylim(-1,52 if filename.endswith("interleaved") else 102);ax.grid(axis="y",alpha=.15)
    axs[0].set_ylabel("Recovered targets (%)")
    h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,ncol=3,loc="lower center",frameon=False,fontsize=7)
    fig.subplots_adjust(bottom=.31 if balanced else .35,wspace=.24,top=.96,left=.085,right=.98)
    for ext in ("svg","pdf"):fig.savefig(PAPER/f"{filename}.{ext}")
    plt.close(fig)


def balanced_figure():
    base=HERE/"interleaved";cfg=json.loads((base/"settings.json").read_text())
    ix=np.random.default_rng(381075).integers(0,128,(5000,128));rows=[]
    for m in cfg["methods"]:
        with np.load(base/f"{m}-episodes.npz") as t,np.load(base/f"{m}-rms.npz") as a:
            np.testing.assert_equal(t["ids"],a["ids"]);ts=t["elapsed_ms"];r=a["rms"]
            for axis,thresholds in (("draws",range(1,21)),("ms",(1,2,5,10,20,30,50,80,100,150,250,500,1000,2500,5000))):
                for b in thresholds:
                    mask=ts<=b if axis=="ms" else np.broadcast_to(np.arange(20)<b,ts.shape)
                    h=(np.isfinite(r)&mask).any(1)
                    lo,hi=np.quantile(h[ix].mean(1),(.025,.975))
                    rows.append(dict(method=m,axis=axis,budget=b,mean_completed=mask.sum(1).mean(),hits=h.sum(),targets=128,mr=h.mean(),low=lo,high=hi))
    table("interleaved/figure-data.csv",rows)
    plot(rows,"csp-interleaved")


if __name__=="__main__":
    rows,pairs,evidence=fixed_draw();sensitivities();curves,cost=timing();plot(curves);balanced_figure()
    (HERE/"evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
