"""Editable vector manuscript figures from the actual algorithm and paired score records."""
import argparse
import csv
import itertools
import json
import pathlib
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle
from matplotlib.collections import LineCollection
from matplotlib import colors
HERE=pathlib.Path(__file__).resolve().parent;V3=HERE.parent;ROOT=V3.parents[1]
sys.path.insert(0,str(ROOT))
from data import unpack_gram,ELEMENTS
plt.rcParams.update({"font.size":8,"font.family":"DejaVu Sans","svg.fonttype":"none","pdf.fonttype":42,
                    "axes.spines.top":False,"axes.spines.right":False,"axes.labelcolor":"#253447","text.color":"#253447"})
BLUE="#226a96";ORANGE="#bc5c2d";TEAL="#167768";GRAY="#65717e";VIOLET="#765193"
def save(fig,name):
    for ext in ("pdf","svg"):
        fig.savefig(HERE/f"{name}.{ext}",bbox_inches="tight",pad_inches=.07)
    plt.close(fig)
def workflow():
    fig,ax=plt.subplots(figsize=(8,3.5));ax.set(xlim=(0,10),ylim=(0,4.3));ax.axis("off")
    def node(x,y,w,h,title,detail,color):
        box=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.035,rounding_size=.10",
             facecolor=colors.to_rgba(color,.075),edgecolor=color,linewidth=1.0)
        ax.add_patch(box);ax.text(x+.13,y+h*.69,title,weight="bold",fontsize=8.3,color=color,va="center")
        ax.text(x+.13,y+h*.28,detail,fontsize=7.0,va="center",linespacing=1.15)
    def arrow(a,b,c="#6b7886",bend=0):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=11,lw=1.15,color=c,
                    connectionstyle=f"arc3,rad={bend}"))
    ax.text(0,3.99,"INFERENCE",weight="bold",fontsize=8,color=BLUE)
    node(.06,2.65,1.48,.94,"Unit cell","Species and\nexact counts",BLUE)
    node(1.94,2.65,2.03,.94,"Train-only lookup","Feasible occupations\nranked by chemistry",BLUE)
    node(4.40,2.65,2.12,.94,"Symbolic seed","Sample cell and orbits,\nexpand sites",TEAL)
    node(7.02,2.65,2.83,.94,"One orbit-network call","Global context, allowed orbit\nshifts, cell shape and volume",ORANGE)
    for a,b in (((1.58,3.12),(1.9,3.12)),((4.01,3.12),(4.36,3.12)),((6.56,3.12),(6.98,3.12))):arrow(a,b)
    ax.text(5.46,2.22,"Zero-call output",ha="center",color=TEAL,weight="bold",fontsize=8)
    ax.text(8.44,2.22,"One-call output",ha="center",color=ORANGE,weight="bold",fontsize=8)
    arrow((5.46,2.65),(5.46,2.36),TEAL);arrow((8.44,2.65),(8.44,2.36),ORANGE)
    ax.plot([0,9.92],[1.93,1.93],color="#cad3dc",lw=.8)
    ax.text(0,1.64,"TRAINING",weight="bold",fontsize=8,color=VIOLET)
    node(.06,.24,2.78,.95,"Exact-formula banks","Observed positives, generated\nnegatives and queries",VIOLET)
    node(3.34,.24,2.80,.95,"Periodic descriptors","Angular powers with optional\ninter-orbit or reciprocal terms",VIOLET)
    node(6.68,.24,3.17,.95,"Stopped regression target","Affinity-weighted attraction and\nrepulsion update the generator",VIOLET)
    arrow((2.88,.72),(3.30,.72));arrow((6.18,.72),(6.64,.72))
    save(fig,"workflow")
def results(modeltag,maindesc):
    rep=json.loads((V3/"evaluation/reports/test.json").read_text())
    table=rep["quality"]
    labels={"prior":"Symbolic prior","angular-small-11":"Angular, 0.2M",modeltag:maindesc,"released-uflow":"Released uFlowCSP"}
    palette=[GRAY,TEAL,ORANGE,BLUE]
    fig,ax=plt.subplots(figsize=(6.6,2.8));bins=["1-4","5-8","9-12","13-20"]
    for (tag,name),color in zip(labels.items(),palette):
        pts=[table.get(f"{b}/20/{tag}") for b in bins]
        if all(pts):
            ax.plot(range(4),[100*p["mr"] for p in pts],"o-",label=name,color=color,lw=1.9,ms=4.3)
    ax.set(xticks=range(4),xticklabels=["1–4","5–8","9–12","13–20"],xlabel="Sites per unit cell",ylabel="Recovered at 20 draws (%)",ylim=(-2,103))
    ax.grid(axis="y",alpha=.16);ax.legend(frameon=False,ncol=2,fontsize=8);save(fig,"sites")
    fig,ax=plt.subplots(figsize=(6.7,3.05))
    tags=[("angular-small-3seeds","Angular, 0.2M (three fits)"),("contacts-small-3seeds","Contact, 0.2M (three fits)"),
          ("reciprocal-small-11","Reciprocal, 0.2M"),("meanflow-small-11","MeanFlow, 0.2M"),
          ("angular-medium-11","Angular, 2M"),("contacts-medium-11","Contact, 2M"),("meanflow-medium-11","MeanFlow, 2M"),
          (modeltag,maindesc)]
    tags=list(dict(tags).items());points=[]
    for tag,label in tags:
        m=rep["paired"].get(f"all/5/{tag}--prior")
        if m:points.append((label,m))
    for y,(label,v) in enumerate(points):
        lo,hi=v["ci95_pp"];ax.plot([lo,hi],[y,y],color=BLUE,lw=2);ax.scatter(v["pp"],y,s=27,color=TEAL,zorder=3)
    ax.axvline(0,color=GRAY,lw=.9);ax.axvline(3,color=ORANGE,ls="--",lw=.9)
    ax.set(yticks=range(len(points)),yticklabels=[x[0] for x in points],xlabel="Gain over the same symbolic seeds (percentage points)")
    ax.invert_yaxis();ax.grid(axis="x",alpha=.16);save(fig,"increments")
def crystal():
    panel=json.loads((V3/"denovo/screen/panel.json").read_text())["indices_zero_based"]
    draws=sorted((V3/"denovo/screen/inputs").glob("*.json"))
    records=[json.loads(p.read_text()) for p in draws]
    arms=[a for a in sorted({r.get("arm") for r in records if r.get("arm")}) if a!="prior"]
    if not arms:return
    arm=arms[0]; raw={r["index"]:r for r in records if r.get("arm")==arm}
    ix=next(j for j in panel if j in raw and len(set(raw[j]["z"]))>=2)
    fig,axs=plt.subplots(2,2,figsize=(6.6,4.7))
    theta=np.deg2rad(25);phi=np.deg2rad(24)
    basis=np.array([[np.cos(theta),-np.sin(theta)*np.sin(phi),np.sin(theta)*np.cos(phi)],
                    [-np.sin(theta),-np.cos(theta)*np.sin(phi),np.cos(theta)*np.cos(phi)],[0,np.cos(phi),np.sin(phi)]])
    for rr,name in enumerate(("prior",arm)):
        row=next(r for r in records if r.get("arm")==name and r["index"]==ix)
        opt=json.loads((V3/"denovo/screen/records"/(row["id"]+".json")).read_text())
        for cc,(k,f) in enumerate([(row["k"],row["f"]),(opt.get("final_k"),opt.get("final_f"))]):
            ax=axs[rr,cc];ax.axis("off");ax.set_aspect("equal")
            if k is None:ax.text(.1,.5,"Relaxation failed");continue
            L=np.linalg.cholesky(unpack_gram(np.array(k)));v=np.array(list(itertools.product((0,1),repeat=3)))
            u=v@L@basis;p=np.array(f)@L@basis
            for i,j in itertools.combinations(range(8),2):
                if np.sum(v[i]!=v[j])==1:ax.plot(u[[i,j],0],u[[i,j],1],lw=.75,color="#8293a3",zorder=0)
            zs=np.array(row["z"]);zvals=sorted(set(zs));pal=[BLUE,ORANGE,TEAL,VIOLET,"#ae9528","#687737"]
            for j in np.argsort(p[:,2]):
                col=pal[zvals.index(zs[j])%len(pal)]
                ax.scatter(*p[j,:2],s=87,c=col,edgecolors="white",linewidths=.65,zorder=2)
            ax.set_title(("Zero-call prior" if rr==0 else "One-call update")+(" · raw" if cc==0 else " · relaxed"),fontsize=9)
            if rr==0 and cc==0:
                for z,c in zip(zvals,pal):ax.scatter([],[],s=36,c=c,label=ELEMENTS[z])
                ax.legend(frameon=False,ncol=2,loc="lower left",fontsize=7)
    fig.tight_layout();save(fig,"crystals")
    (HERE/"crystal-example.json").write_text(json.dumps(dict(index=ix,arm=arm,rule="first preselected panel index with at least two elements",source=raw[ix]["id"])))
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--results",nargs=2);ap.add_argument("--crystals",action="store_true")
    a=ap.parse_args();workflow()
    if a.results:results(*a.results)
    if a.crystals:crystal()
