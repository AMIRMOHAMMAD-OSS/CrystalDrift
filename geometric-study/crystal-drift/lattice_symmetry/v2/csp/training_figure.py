"""Plot every matched seed and objective on a common validation scale and update budget."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;PAPER=HERE.parent/"paper"
STYLES={"angular":("Angular orders 0–2","#006759"),"power0":("Center power 0","#6d579a"),"radial":("Pair shells","#bb5b15"),
        "fixed-angular":("Fixed basis, angular field","#245e96"),"flow16":("Matched 16-step flow","#86394f")}

def main():
    budgets=[256,1024,3392];seeds=[11,23,37]
    lock=json.loads((HERE/"selection.json").read_text())
    plt.rcParams.update({"font.size":8,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(7.7,3.35),sharey=True);rows=[]
    for panel,methods in zip(axes,(("angular","power0","radial"),("fixed-angular","flow16"))):
        for m in methods:
            label,color=STYLES[m];se=[]
            for seed in seeds:
                values=[]
                for u in budgets:
                    with np.load(HERE/f"fits/{m}-{seed}/val-{u:05}.npz") as vt:
                        assert vt["energy"].shape==(128,)
                        mean=vt["energy"].mean()
                        np.testing.assert_allclose(mean,lock[m]["validation_energy_by_seed"][str(u)][seeds.index(seed)])
                        values.append(mean)
                        rows.append(dict(method=m,seed=seed,updates=u,positive_exposures=u*32,mean_fair_geometry_score=mean,validation_targets=128,draws=5))
                panel.plot(budgets,values,color=color,alpha=.33,lw=.8,marker=".",ms=3)
                se.append(values)
            panel.plot(budgets,np.mean(se,axis=0),color=color,lw=2,label=label)
            selected=lock[m]["updates"];j=budgets.index(selected)
            panel.scatter([selected],[np.mean(se,axis=0)[j]],facecolors="white",edgecolors=color,zorder=5,s=28,lw=1.4)
        panel.set_xscale("log");panel.set_xticks(budgets,["256","1,024","3,392"]);panel.minorticks_off()
        panel.set_xlim(220,3800);panel.grid(axis="y",alpha=.15)
        panel.set_xlabel("Optimizer updates")
        panel.legend(loc="lower left",fontsize=7,frameon=False)
    axes[0].set_ylabel("Validation geometry score (lower is better)")
    axes[0].set_title("Physical generator",loc="left",fontweight="bold",fontsize=9)
    axes[1].set_title("Fixed-basis controls",loc="left",fontweight="bold",fontsize=9)
    axes[0].set_ylim(1.55,3.25)
    fig.subplots_adjust(bottom=.18,wspace=.12,left=.092,right=.985,top=.89)
    PAPER.mkdir(exist_ok=True)
    for ext in ("svg","pdf"):fig.savefig(PAPER/f"csp-validation.{ext}")
    plt.close(fig)
    with (HERE/"validation-curves.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    rng=np.random.default_rng(172091)
    ix=rng.integers(0,128,(5000,128))
    paired=[]
    for u in budgets:
        for m in ("radial","power0","fixed-angular","flow16"):
            delta=[]
            for s in seeds:
                with np.load(HERE/f"fits/angular-{s}/val-{u:05}.npz") as a,np.load(HERE/f"fits/{m}-{s}/val-{u:05}.npz") as b:
                    np.testing.assert_equal(a["ids"],b["ids"])
                    delta.append(a["energy"]-b["energy"])
            by_target=np.mean(delta,0)
            lo,hi=np.quantile(by_target[ix].mean(1),(.025,.975))
            paired.append(dict(first="angular",second=m,updates=u,targets=128,
                               energy_delta=by_target.mean(),lower=lo,upper=hi))
    with (HERE/"paired-validation.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    print("Plotted",len(rows),"checkpoint values.")

if __name__=="__main__":main()
