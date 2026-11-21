"""Vector plot of the actual chemical shortlist's target occupancy coverage."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
def main():
    matplotlib.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.spines.top":False,
        "axes.spines.right":False,"pdf.fonttype":42,"svg.fonttype":"none"})
    f,(left,right)=plt.subplots(1,2,figsize=(7.1,3.1),sharey=True)
    colors=["#243e68","#d18526","#087d78","#93477f","#6c737d"]
    k=np.arange(20)+1
    def draw(ax,panel,key,label,color,marker,ls="-"):
        with np.load(HERE/f"audits/reachable-{panel}.npz") as d:
            y=np.maximum.accumulate(d[key],axis=1).mean(0)*100
        ax.plot(k,y,label=label,color=color,lw=1.6,marker=marker,markevery=[0,4,9,14,19],ms=3.2,ls=ls)
    draw(left,"test","group","Space-group type",colors[2],"^")
    draw(left,"test","equivalent","Equivalent occupied pattern",colors[0],"o")
    draw(left,"test","canonical","Literal Wyckoff letters",colors[1],"s","--")
    draw(right,"shared","equivalent","Shared families (96)",colors[2],"^")
    draw(right,"arrangement-control","equivalent","Covered controls (96)",colors[1],"s","--")
    draw(right,"arrangement","equivalent","Uncovered arrangements (25)",colors[3],"o")
    draw(right,"family","equivalent","Excluded families (228)",colors[4],"x",":")
    left.set_title("Official MP-20 test (9,046)",loc="left",fontweight="semibold",fontsize=10)
    right.set_title("Structural challenges",loc="left",fontweight="semibold",fontsize=10)
    for ax in (left,right):
        ax.set_ylim(-2,103);ax.set_xlim(.65,20.35);ax.set_yticks([0,25,50,75,100]);ax.set_xticks([1,5,10,15,20])
        ax.set_xlabel("Ordered symbolic proposals");ax.grid(axis="y",lw=.45,color="#d6dbe0");ax.set_axisbelow(True)
        ax.legend(loc="lower left" if ax is left else "lower right",
                  bbox_to_anchor=(0,.1) if ax is left else (.998,.09),
                  frameon=True,facecolor="white",edgecolor="none",
                  handlelength=1.6,fontsize=7,labelspacing=.42,borderaxespad=0.1)
    left.set_ylabel("Targets reachable (%)")
    f.subplots_adjust(left=.085,right=.983,top=.9,bottom=.17,wspace=.13)
    for suffix in ("pdf","svg"):f.savefig(HERE/f"template-coverage.{suffix}")
    plt.close(f)

if __name__=="__main__":main()
