"""Vector explanation of a verified Fourier collision and its finite-cutoff angular counterpart."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
m=np.array([[0,1,4,6],[0,1,3,7]])
table=list(csv.DictReader((HERE/"decorated-cutoff-sweep.csv").open()))
vec=lambda k:np.array([float(r[k]) for r in table])
plt.rcParams.update({"svg.fonttype":"none","pdf.fonttype":42,"font.size":8.2,"font.family":"DejaVu Sans",
    "axes.spines.top":False,"axes.spines.right":False,"axes.labelcolor":"#39494c","axes.edgecolor":"#a5b3b0",
    "xtick.color":"#465351","ytick.color":"#465351","axes.titlelocation":"left"})
fig=plt.figure(figsize=(7.85,3.9))
grid=fig.add_gridspec(2,2,width_ratios=(.91,1.18),height_ratios=(1.1,.83),left=.057,right=.99,
                      bottom=.17,top=.90,wspace=.31,hspace=.56)
ax=fig.add_subplot(grid[0,0]);ax.axis("off");ax.set_xlim(-2.2,11.6);ax.set_ylim(-.6,1.66)
ax.set_title("A   Decorated twelve-period motifs",fontweight="semibold",fontsize=9.7,pad=6)
for k,indices in enumerate(m):
    y=1.15-1.1*k
    ax.text(-.9,y-.15,"A" if k==0 else "B",ha="right",va="center",weight="bold")
    for j,(off,c,mk) in enumerate(((0,"#148375","o"),(-.32,"#b6662a","s"))):
        ax.plot([0,11],[y+off]*2,color="#d1dad8",lw=.8,zorder=0)
        ax.scatter(range(12),np.full(12,y+off),s=13,facecolors="white",edgecolors="#b4c2c1",lw=.6,marker=mk)
        ax.scatter(indices,np.full(4,y+off),s=33,facecolors=c,edgecolors="white",lw=.4,marker=mk)
for x in (0,3,6,9):ax.text(x,-.53,str(x),ha="center",fontsize=7,color="#5b6766")
ax.scatter([],[],s=26,c="#148375",label="O");ax.scatter([],[],s=26,c="#b6662a",marker="s",label="Si")
ax.legend(loc="upper left",bbox_to_anchor=(-.03,.04),ncol=2,frameon=False,columnspacing=.6,handletextpad=.1,fontsize=7)
ax=fig.add_subplot(grid[1,0]);ax.set_title("B   One phase-sensitive channel",fontweight="semibold",fontsize=9.7,pad=9)
val=np.array([np.mean(np.exp(2j*np.pi*s/12))**2*np.conj(np.mean(np.exp(4j*np.pi*s/12))) for s in m]).real/np.sqrt(6)
ax.axvline(0,c="#879592",lw=.7);ax.barh([1,0],val,height=.42,color=("#386b83","#99596f"))
ax.set_yticks([1,0],["A","B"]);ax.set_xlim(-.028,.028);ax.set_xticks([-.02,0,.02],["−0.02","0","+0.02"])
ax.set_xlabel("Real signed triple / √6",labelpad=2.5);ax.tick_params(length=2);ax.spines["left"].set_visible(False)
ax=fig.add_subplot(grid[:,1]);ax.set_title("C   What the angular channels resolve",fontweight="semibold",fontsize=9.7,pad=9)
ax.axvspan(48,60,color="#e7efec",zorder=0)
for label,k,c,style in [("Order 0","power0","#1a776b","-"),("Order 1","power1","#52659a","--"),
                        ("Order 2","power2","#b5672f",":")]:
    ax.plot(vec("a"),vec(k),lw=2,ls=style,color=c,label=label)
ax.axvline(48,color="#9daea6",ls=(0,(3,3)),lw=.8);ax.set_xlim(4,60)
ax.set_ylim(-.02,1.04);ax.set_xticks([6,18,30,42,54]);ax.set_xlabel("Axial cell length a (Å)")
ax.set_ylabel("Unscaled block difference (L2 norm)",labelpad=2)
ax.grid(axis="y",color="#dce4e1",alpha=.8,linewidth=.65)
ax.legend(loc="upper right",frameon=False,fontsize=7.4)
ax.text(54,.43,"No edges\nbetween\ndistinct\naxial sites",ha="center",va="center",color="#45594e",fontsize=7.)
for suffix in ("pdf","svg"):fig.savefig(HERE/f"fourier-cutoff.{suffix}",bbox_inches="tight",pad_inches=.08)
