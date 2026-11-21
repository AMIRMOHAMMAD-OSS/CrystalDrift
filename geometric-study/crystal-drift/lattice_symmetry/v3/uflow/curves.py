"""Plot the observed squared velocity residuals and the four frozen seed-11 development looks."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
def main():
    matplotlib.rcParams.update({"svg.fonttype":"none","pdf.fonttype":42,"font.size":9,
                                "axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(7.3,2.7),layout="constrained")
    dat=[]
    for tier,label,c in (("small","0.234 M","#006f78"),("medium","2.135 M","#ae4c2c")):
        long=HERE/f"fits/{tier}-11-long"
        selected=json.loads((long/"selection.json").read_text())
        curve=np.array(json.loads((long/"curve.json").read_text()))
        seen=[];values=[]
        for i in range(0,len(curve),48):
            part=curve[i:i+48];seen.append(part[:,1].mean());values.append(np.median(part[:,4]))
        axes[0].plot(np.array(seen)/26614,values,color=c,lw=1.25,label=label)
        passes=[.5,1,4,8]
        keys=["ema-00416","ema-00832","ema-03328","ema-06656"]
        obs=np.array([selected["results"][k] for k in keys])
        axes[1].plot(passes,100*obs/96,"o-",color=c,lw=1.3,ms=4,label=label)
        for e,k,h in zip(passes,keys,obs):
            dat.append(dict(tier=tier,exposures=e,checkpoint=k,hits=int(h),targets=96,
                            selected=selected["checkpoint"].endswith(k+".pt")))
    axes[0].set(ylabel="Masked velocity MSE",xlabel="Training traversals",yscale="log")
    axes[1].set_xscale("log",base=2)
    axes[1].set(ylabel="Development MR@5 (%)",xlabel="Training traversals (log scale)")
    axes[1].set_xticks([.5,1,2,4,8],labels=["0.5","1","2","4","8"])
    for ax in axes:
        ax.grid(axis="y",alpha=.16)
    axes[0].legend(frameon=False,title="uFlowCSP",loc="upper right")
    for ext in ("pdf","svg"):
        fig.savefig(HERE/f"uflow-learning.{ext}",bbox_inches="tight")
    plt.close(fig)
    with (HERE/"validation-curves.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=dat[0]);wr.writeheader();wr.writerows(dat)
if __name__=="__main__":main()
