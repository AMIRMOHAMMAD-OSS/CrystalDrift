"""Editable vector drawing of an exact rebasing and the homometric chord example."""
import json
from pathlib import Path
from itertools import combinations
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({"svg.fonttype":"none","font.size":9,"font.family":"DejaVu Sans","pdf.fonttype":42})
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
HERE=Path(__file__).resolve().parent


def main():
    test=json.loads((HERE/"results/physics-acceptance.json").read_text())
    w=test["training_witness"]["twelve_gon"]
    fig=plt.figure(figsize=(7.75,3.2),layout="constrained")
    gs=fig.add_gridspec(2,3,width_ratios=(1.42,1,1),height_ratios=(1,.34))
    a=fig.add_subplot(gs[0,0]);ax1=fig.add_subplot(gs[0,1]);ax2=fig.add_subplot(gs[0,2])
    for ax in (a,ax1,ax2):
        ax.set_aspect("equal");ax.set_axis_off()
    graphite="#24364a";teal="#008483";red="#c45b23";line="#6f8192"
    grid=np.array([(x,y) for x in range(-1,3) for y in range(-1,4)])
    a.scatter(*grid.T,color="#9aa7b0",s=14,zorder=3)
    square=np.array([[0,0],[1,0],[1,1],[0,1]])
    shear=np.array([[0,0],[1,1],[1,2],[0,1]])
    for v,c in ((square,teal),(shear,red)):
        a.add_patch(Polygon(v,closed=True,facecolor=c,alpha=.12,zorder=1))
        a.add_patch(Polygon(v,closed=True,fill=False,edgecolor=c,lw=1.9,zorder=2))
    a.set(xlim=(-.63,1.64),ylim=(-.34,2.23))
    a.set_title("A   One lattice, two bases",loc="left",weight="bold",color=graphite,fontsize=9)
    t=2*np.pi*np.arange(12)/12; pts=np.column_stack((np.cos(t),np.sin(t)))
    for ax,V,title in zip((ax1,ax2),w["vertices"],("B   Motif A","C   Motif B")):
        theta=np.linspace(0,2*np.pi,400)
        ax.plot(np.cos(theta),np.sin(theta),lw=.8,color="#acb7c2")
        ax.scatter(*pts.T,s=6,color="#a3b1bd")
        pairs=sorted(combinations(V,2),key=lambda ij:np.linalg.norm(pts[ij[0]]-pts[ij[1]]))
        for i,(x,y) in enumerate(pairs):
            c=(teal,red)[i] if i<2 else "#9aa8b3"
            ax.plot(*pts[[x,y]].T,color=c,lw=2.3 if i<2 else .7,zorder=4 if i<2 else 1)
        ax.scatter(*pts[V].T,s=28,color=graphite,zorder=5)
        for k in V:
            x,y=1.19*pts[k]
            ax.text(x,y,str(k),ha="center",va="center",fontsize=8,color=graphite)
        ax.set_title(title,weight="bold",loc="left",color=graphite,fontsize=9)
        ax.set(xlim=(-1.36,1.36),ylim=(-1.32,1.32))
    bottom=fig.add_subplot(gs[1,0]);bottom.set_axis_off()
    bottom.text(.02,.84,"Original basis",color=teal,weight="bold",transform=bottom.transAxes)
    bottom.text(.02,.48,"Integer shear",color=red,weight="bold",transform=bottom.transAxes)
    bottom.text(.02,.10,"Normalized area = 1 for both cells",color=graphite,fontsize=8,transform=bottom.transAxes)
    b=fig.add_subplot(gs[1,1:]);b.set_axis_off()
    stats=w["norms"]
    b.text(.015,.84,"Both motifs have the same six chord lengths.",color=graphite,fontsize=8.5)
    b.text(.015,.48,"The two shortest chords meet only in motif B.",color=graphite,fontsize=8.5)
    b.text(.015,.10,f"Pair-shell difference < 10⁻¹⁴.  Order-2 power difference = {stats['power2']:.5f}.",color=graphite,fontsize=8)
    for ext in ("svg","pdf"):
        fig.savefig(HERE/"paper"/f"equivalence.{ext}",bbox_inches="tight",pad_inches=.03)
    plt.close(fig)


if __name__=="__main__":main()
