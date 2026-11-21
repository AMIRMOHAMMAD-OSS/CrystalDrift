"""Editable validation curves and paired GPU-time versus count curves from measured episodes."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,ScalarFormatter


HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]/"paper"
PALETTE=dict(drift="#087E7C",iid="#DC8237",meanflow="#8A59A2",flow1="#727780",flow16="#375F99",
             retrieval="#705132",uflow="#B33650",crystalflow="#5A752D")
NAMES=dict(drift="CrytalDrift",iid="IID-positive drift",meanflow="Compact MeanFlow",flow1="Flow, 1 Euler step",
           flow16="Flow, 16 Euler steps",retrieval="Retrieval",uflow="Released uFlowCSP",crystalflow="CrystalFlow, 5 steps")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.labelsize":8.5,"axes.titlesize":9,
    "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#656A70","pdf.fonttype":42,
    "svg.fonttype":"none","svg.hashsalt":"CrytalDrift-expansion"})


def rows(path):
    with path.open(newline="") as f:return list(csv.DictReader(f))


def export(fig,name):
    fig.savefig(HERE/(name+".svg"))
    fig.savefig(PAPER/(name+".pdf"))
    plt.close(fig)


def training():
    records=rows(HERE/"validation-curves.csv")
    fig,ax=plt.subplots(figsize=(6.15,2.75))
    for m,label in (("drift","CrytalDrift"),("iid","IID-positive drift"),("meanflow","Compact MeanFlow"),("flow","Flow, 16 Euler steps")):
        color=PALETTE["flow16" if m=="flow" else m]
        grp=[r for r in records if r["method"]==m]
        if not grp:continue
        seeds=[]
        for s in (11,23,37):
            pts=sorted((int(r["passes"]),float(r["energy"])) for r in grp if int(r["seed"])==s)
            if len(pts)==3:
                y=np.array([p[1] for p in pts])
                seeds.append(y);ax.plot([1,3,10],y,color=color,lw=.6,alpha=.24)
        if seeds:
            ax.plot([1,3,10],np.mean(seeds,axis=0),marker="o",ms=4,mew=.5,
                    color=color,label=label,linewidth=1.7)
    ax.set_xscale("log");ax.set_xticks([1,3,10],["1","3","10"])
    ax.set(xlabel="Passes through the MP-20 training schedule",ylabel="Validation geometry energy score")
    ax.grid(axis="y",color="#CAD4D5",linewidth=.6,alpha=.7)
    ax.legend(frameon=False,fontsize=7,loc="upper left",bbox_to_anchor=(1.01,1))
    fig.subplots_adjust(left=.11,right=.72,bottom=.21,top=.96)
    export(fig,"exp-training-curves")


def gpu():
    records=rows(HERE/"gpu/recovery-curves.csv")
    compare=HERE.parent/"external/timing-s5"
    if (compare/"serial-scores.npz").exists():
        with np.load(compare/"serial.npz") as a,np.load(compare/"serial-scores.npz") as b,np.load(HERE/"gpu/drift-episodes.npz") as c:
            np.testing.assert_array_equal(a["ids"],b["ids"])
            np.testing.assert_array_equal(a["ids"],c["ids"])
            t=a["elapsed_ms"];hit=np.isfinite(b["rms"])
        for r in [r for r in records if r["method"]=="drift"]:
            k=float(r["budget"])
            mask=t<=k if r["axis"]=="time" else np.arange(20)<k
            records.append(dict(method="crystalflow",axis=r["axis"],budget=k,mr=(hit&mask).any(1).mean()))
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.65),sharey=True)
    for mode,ax,title in zip(("draws","time"),axes,("a    Fixed candidate budget","b    Fixed elapsed time")):
        for method in NAMES:
            entries=[r for r in records if r["method"]==method and r["axis"]==mode]
            if not entries:continue
            x=np.array([float(r["budget"]) for r in entries])
            y=np.array([float(r["mr"])*100 for r in entries])
            if mode=="time":
                serial=compare/"serial.npz" if method=="crystalflow" else HERE/f"gpu/{method}-episodes.npz"
                scored=compare/"serial-scores.npz" if method=="crystalflow" else HERE/f"gpu/{method}-rms.npz"
                with np.load(serial) as a,np.load(scored) as b:
                    np.testing.assert_array_equal(a["ids"],b["ids"])
                    first=np.where(np.isfinite(b["rms"]),a["elapsed_ms"],np.inf).min(1)
                t=np.sort(first)
                x=np.r_[.2,np.unique(t[(t>.2)&(t<10000)]),10000.]
                y=np.searchsorted(t,x,side="right")*100/len(t)
            ax.plot(x,y,color=PALETTE[method],label=NAMES[method],linewidth=1.6,
                    linestyle="--" if method=="flow1" else ":" if method=="iid" else "-",drawstyle="steps-post")
        ax.set_title(title,loc="left",fontweight="bold");ax.grid(axis="y",color="#BBCACC",lw=.6,alpha=.7)
        ax.set_ylim(0,100)
        if mode=="time":
            ax.set_xscale("log");ax.set_xlim(.2,10000)
            ax.set_xticks([1,10,100,1000,10000],["1","10","100","1,000","10,000"])
            ax.set_xlabel("Completed proposal time per target (ms)")
        else:
            ax.set_xticks([1,5,10,15,20]);ax.set_xlim(1,20)
            ax.set(xlabel="Completed candidates per target",ylabel="Targets recovered (%)")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=4,frameon=False,fontsize=7)
    fig.subplots_adjust(left=.09,right=.955,bottom=.28,top=.90,wspace=.14)
    export(fig,"exp-gpu-curves")


if __name__=="__main__":
    training()
    if (HERE/"gpu/recovery-curves.csv").exists():gpu()
