"""Vector figure of as-generated space-group distributions with all failures retained."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import numpy as np

HERE=Path(__file__).resolve().parent/"catalog"
table=list(csv.DictReader((HERE/"spacegroup-counts.csv").open()))
assert [int(r["international_number_0_is_failure"]) for r in table]==list(range(231))
names={"held_out_reference":"MP-20 test","source_structures":"Formula sources",
       "prior":"Symbolic prior","contacts-small":"Small contacts",
       "angular-small":"Small angular","angular-medium-long":"Extended medium angular",
       "contacts-medium-long":"Extended medium contacts","released-angular":"Released angular","released-uflow":"Released uFlowCSP"}
labels=[("No group",0,0),("P1",1,1),("P−1",2,2),("Mono.",3,15),("Ortho.",16,74),("Tetra.",75,142),
        ("Trig.",143,167),("Hex.",168,194),("Cubic",195,230)]
arms=[k for k in table[0] if k!="international_number_0_is_failure"]
total=np.array([[int(r[a]) for r in table] for a in arms])
X=100*np.stack([total[:,lo:hi+1].sum(1) for _,lo,hi in labels],axis=1)/total.sum(1)[:,None]
assert np.allclose(X.sum(1),100)
plt.rcParams.update({"svg.fonttype":"none","pdf.fonttype":42,"font.size":8.7,"axes.spines.top":False,
    "axes.spines.right":False,"axes.spines.left":False,"axes.spines.bottom":False})
fig,ax=plt.subplots(figsize=(7.9,.4*len(arms)+1.0))
cmap=LinearSegmentedColormap.from_list("groups",["#f6f6f0","#b6d6cc","#58a69b","#17655f","#123e3e"])
norm=PowerNorm(.5,vmin=0,vmax=100)
xedges=np.arange(len(labels)+1)-.5
yedges=np.arange(len(arms)+1)-.5
ax.pcolormesh(xedges,yedges,X,cmap=cmap,norm=norm,shading="flat",rasterized=False)
ax.set_xlim(xedges[0],xedges[-1]);ax.set_ylim(yedges[-1],yedges[0])
for r in range(len(arms)):
    for c,val in enumerate(X[r]):
        valtext="0" if val==0 else "<0.1" if val<.05 else f"{val:.1f}"
        ax.text(c,r,valtext,ha="center",va="center",color="white" if norm(val)>.65 else "#172d2b")
ax.set_xticks(range(len(labels)),[k for k,_,_ in labels]);ax.xaxis.tick_top()
ax.set_yticks(range(len(arms)),[names.get(a,a) for a in arms])
ax.tick_params(length=0,pad=5)
ax.set_xticks(np.arange(-.5,len(labels),1),minor=True);ax.set_yticks(np.arange(-.5,len(arms),1),minor=True)
ax.tick_params(which="minor",length=0);ax.grid(which="minor",linewidth=.7,color="white")
ax.axhline(1.5,color="#506663",lw=1)
ax.set_xlabel("Share of ordered attempts (%)     ·     Detection tolerance 0.01 Å",labelpad=7)
fig.tight_layout(pad=.65)
for suffix in ("svg","pdf"):fig.savefig(HERE/f"spacegroups.{suffix}",bbox_inches="tight",pad_inches=.04)
