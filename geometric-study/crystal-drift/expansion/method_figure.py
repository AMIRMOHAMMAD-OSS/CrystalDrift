"""Editable full-chemistry sampling and training diagram with a real ordered draw."""
from pathlib import Path
from itertools import product
import sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
# Cell reconstruction below uses the same unpacker as structure matching.

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
mpl.rcParams.update({"svg.fonttype":"none","pdf.fonttype":42,"font.family":"DejaVu Sans"})
fig, ax = plt.subplots(figsize=(9.2,4.45))
fig.subplots_adjust(left=.015,right=.985,bottom=.025,top=.99)
ax.set(xlim=(0,10),ylim=(0,4.7));ax.axis("off")
ink="#1c3440";blue="#246687";orange="#9b4f28";teal="#20756b"

def label(x,y,msg,size=10,weight="normal",color=ink,align="left"):
    ax.text(x,y,msg,ha=align,va="center",color=color,fontsize=size,fontweight=weight)

def panel(x,y,w,h,fill,edge):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=0.10",
                                facecolor=fill,edgecolor=edge,lw=1))

def arrow(x1,y1,x2,y2,color=blue):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=11,lw=1.4,color=color))

label(.15,4.46,"Sampling",13,"bold",blue)
label(1.38,4.46,"One network evaluation given the cell composition",10)
panel(.18,2.37,2.53,1.72,"#f0f7fa","#9cbecc")
label(.34,3.78,"Unit-cell composition",10,"bold");label(.34,3.42,"Supplied elements and counts",8.6)
label(.34,3.09,"or a draw from a fitted prior",8.6)
label(.34,2.65,"Cell noise + uniform sites",8.8,"bold",blue)
arrow(2.77,3.25,3.13,3.25)
panel(3.19,2.37,2.60,1.72,"#e6f2ef",teal)
label(3.37,3.78,"Periodic set network",10,"bold")
label(3.37,3.42,"Embed elements and site phases",8.6)
label(3.37,3.09,"Pool occupied sites for the cell",8.6)
label(3.37,2.65,"Update sites together",8.8,"bold",teal)
arrow(5.84,3.25,6.19,3.25)
panel(6.26,2.37,3.48,1.72,"#f7faf9","#9cbecc")
label(6.42,3.77,"Generated crystal",10,"bold")
label(6.42,3.40,"Log lattice metric",8.6)
label(6.42,3.09,"and fractional sites",8.6)
label(6.42,2.64,r"K$_2$NaRhCl$_6$ · draw 2",8.3,"bold",blue)
with np.load(HERE/"denovo/pools/learned-drift.npz") as bank:
    z,k,f=(bank[key][1] for key in ("z","k","f"))
sys.path.insert(0,str(ROOT))
from data import unpack_gram
lattice=np.linalg.cholesky(unpack_gram(k))
corners=np.array(list(product((0.,1.),repeat=3)))
proj=np.array([[.89,-.30],[.45,.41],[0,.86]])
wire=corners@lattice@proj;origin=(wire.max(0)+wire.min(0))/2
scale=1.25/max(np.ptp(wire,axis=0))
def project(r):return (r@lattice@proj-origin)*scale+np.array([8.98,3.22])
for a in corners:
    for dim in np.flatnonzero(a==0):
        b=a.copy();b[dim]=1
        p=project(np.stack([a,b]));ax.plot(*p.T,color="#74888d",lw=.75,zorder=2)
palette={11:"#ab5bd2",17:"#57aa59",19:"#8770b7",45:"#64799a"}
pts=project(f[z>0]);ax.scatter(*pts.T,s=30,c=[palette[int(e)] for e in z[z>0]],edgecolors="white",linewidths=.45,zorder=4)
ax.plot([.15,9.82],[2.12,2.12],color="#c1cbd1",lw=.8)
label(.15,1.88,"Training",13,"bold",orange)
label(1.38,1.88,"Attraction and repulsion act on differentiable features",10)
panel(.18,.19,2.53,1.38,"#fff6ef","#d6ac8e")
label(.34,1.29,"Reference banks",10,"bold")
label(.34,.94,"Observed structures (+)",8.6)
label(.34,.59,"Independent generated draws (−)",8.6)
arrow(2.77,.89,3.13,.89,orange)
panel(3.19,.19,2.60,1.38,"#fff6ef","#d6ac8e")
label(3.37,1.29,"Periodic fingerprint",10,"bold")
label(3.37,.94,"Six lattice coordinates",8.6)
label(3.37,.59,"Species-resolved Fourier products",8.6)
arrow(5.84,.89,6.19,.89,orange)
panel(6.26,.19,3.48,1.38,"#fff6ef","#d6ac8e")
label(6.42,1.29,"Detached feature target",10,"bold")
label(6.42,.94,"Compare live features with both banks",8.6)
label(6.42,.59,"Backpropagate through the live query",8.6)
fig.savefig(ROOT/"paper/exp-method.pdf");fig.savefig(HERE/"exp-method.svg");plt.close(fig)
