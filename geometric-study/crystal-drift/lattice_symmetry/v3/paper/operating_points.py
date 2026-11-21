"""Plot same-device time, ordered draw prefixes and measured cost at capacity."""
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from figures import HERE,V3,BLUE,TEAL,ORANGE,VIOLET,GRAY,save

def main():
    folder=V3/"evaluation/timing"
    names=json.loads((folder/"settings.json").read_text())["method_order"]
    colors=plt.get_cmap("tab10").colors
    labels={"prior":"Symbolic prior","geometry-donor":"Geometry donor","released-uflow":"Released uFlowCSP",
        "angular-small-long":"Angular 0.2M","angular-medium-long":"Angular 2M",
        "contacts-medium-long":"Contact 2M","meanflow-medium-long":"Orbit MeanFlow 2M",
        "uflow-small-11-long":"Native uFlow 0.2M","uflow-medium-11-long":"Native uFlow 2M"}
    fig,ax=plt.subplots(1,2,figsize=(8,3.2),sharey=True)
    for idx,file in enumerate(("draws.csv","elapsed.csv")):
        data=list(csv.DictReader((folder/file).open()))
        for j,m in enumerate(names):
            rows=[r for r in data if r["method"]==m]
            x=np.array([float(r["k" if idx==0 else "ms"]) for r in rows])
            y=np.array([100*float(r["mr"]) for r in rows])
            ax[idx].plot(x,y,color=colors[j%10],lw=1.8,marker="o" if idx else None,ms=2.7,label=labels.get(m,m))
            if m=="contacts-medium-long":
                ax[idx].fill_between(x,[100*float(r["low"]) for r in rows],[100*float(r["high"]) for r in rows],color=colors[j%10],alpha=.13,linewidth=0)
        ax[idx].grid(axis="y",alpha=.15)
    ax[0].set(xlabel="Completed candidates per target",ylabel="Recovered targets (%)",xticks=[1,5,10,15,20],ylim=(-2,104))
    ax[1].set(xlabel="Elapsed proposal time (ms)",xscale="log",xticks=[5,20,100,500,1000])
    ax[1].set_xticklabels(["5","20","100","500","1,000"])
    h,l=ax[0].get_legend_handles_labels();fig.legend(h,l,loc="lower center",bbox_to_anchor=(.5,-.12),ncol=3,frameon=False,fontsize=7)
    fig.tight_layout();save(fig,"operating-points")
if __name__=="__main__":main()
