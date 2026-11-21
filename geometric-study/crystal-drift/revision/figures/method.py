"""Editable method schematic and paired CHGNet outcomes from the deposited observations."""
from pathlib import Path
import csv
import itertools
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
INK="#132b3a"; TEAL="#11746e"; GOLD="#b57518"; PLUM="#6d4c83"; GREY="#536976"
plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"svg.fonttype":"none",
                     "axes.spines.top":False,"axes.spines.right":False,"font.size":9})


def save(fig,name):
    for ext in ("pdf","svg","png"):
        fig.savefig(OUT/(name+"."+ext),dpi=185,facecolor="white",bbox_inches="tight",pad_inches=.07)
    shutil.copyfile(OUT/(name+".pdf"),ROOT/"paper"/(name+".pdf"))
    plt.close(fig)


def workflow():
    fig,ax=plt.subplots(figsize=(8.4,3.63))
    fig.subplots_adjust(left=.01,right=.995,bottom=.03,top=.99)
    ax.set(xlim=(0,100),ylim=(0,45),aspect="equal");ax.axis("off")
    def label(x,y,t,fs=9,color=INK,weight="normal",ha="left"):
        ax.text(x,y,t,fontsize=fs,color=color,fontweight=weight,ha=ha,va="center")
    def box(x,y,w,h,fill,stroke):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.12,rounding_size=.7",
                                   facecolor=fill,edgecolor=stroke,linewidth=.9))
    def arrow(a,b,color=GREY,style="-"):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=11,linewidth=1.1,
                                     color=color,linestyle=style,shrinkA=1,shrinkB=1))
    label(0,43,"a   Sampling",12,TEAL,"bold")
    box(0,25.6,26,12.5,"#f2f7f8","#acbfca")
    label(1.5,34.7,"Composition and noise",9.5,weight="bold")
    label(1.5,31.2,"Elements and site counts",8.6)
    label(1.5,28.2,"Gaussian cell, uniform sites",8.1)
    arrow((26.5,31.9),(30,31.9),TEAL)
    box(30.3,25.6,25,12.5,"#e8f4ee","#79aaa0")
    label(32,34.7,"One-pass generator",9.5,TEAL,"bold")
    label(32,31.1,"Cell and site heads",8.5)
    label(32,28.1,"Shared site network",8.5)
    arrow((55.8,31.9),(60,31.9),TEAL)
    box(60.3,25.6,39,12.5,"#f3f4fa","#a5afc4")
    label(62.1,34.7,"Periodic crystal",9.5,weight="bold")
    label(62.1,31.2,"Positive-definite metric",8.6)
    label(62.1,28.2,"All fractional sites",8.6)
    origin=np.array([87.1,28.0]);v=np.array([[6,0],[1.9,2.3],[-.65,4.6]])
    for a in itertools.product([0,1],repeat=3):
        for j in range(3):
            if a[j]==0:
                q=origin+np.array(a)@v
                ax.plot([q[0],q[0]+v[j,0]],[q[1],q[1]+v[j,1]],color="#7c909f",lw=.75)
    for a,c in [([.1,.1,.12],TEAL),([.75,.29,.57],GOLD),([.30,.7,.85],TEAL)]:
        q=origin+np.array(a)@v;ax.scatter(*q,s=28,c=c,edgecolors="white",linewidths=.5,zorder=5)
    ax.plot([0,99],[23.3,23.3],lw=.65,color="#bdc9cb")
    label(0,20.6,"b   Training",12,PLUM,"bold")
    box(0,5.3,23,11.6,"#f9f6ed","#b99e63")
    label(1.4,13.5,"Reference bank",9.3,weight="bold")
    label(1.4,10.3,"Real (+), generated (−)",8.5)
    label(1.4,7.5,"Same cell composition",8.3)
    arrow((23.3,11.1),(28.8,11.1),GOLD)
    box(29,5.3,26,11.6,"#f5f1f8","#ad9abb")
    label(30.4,13.5,"Detached drift target",9.3,PLUM,"bold")
    label(30.4,10.3,"Attraction and repulsion",8.5)
    label(30.4,7.5,"In periodic features",8.0)
    arrow((55.4,11.1),(60,11.1),PLUM)
    box(60.3,5.3,39,11.6,"#f1f7f6","#90b2ad")
    label(62,13.5,"Regress the live features",9.3,TEAL,"bold")
    label(62,9.5,"Gradient through query and generator",8.5)
    arrow((77.9,25.3),(77.9,17.2),TEAL)
    label(79.5,20.3,"Feature map Φ",8.7,TEAL)
    ax.plot([77.8,42.1],[19.0,19.0],color=PLUM,ls=":",lw=.9)
    arrow((42.1,19.0),(42.1,17.0),PLUM,":")
    label(0,2.0,"Only the upper row runs when sampling a new crystal.",8.5,GREY)
    save(fig,"revision-method")


def screen():
    with (ROOT/"de-novo/relaxation/results.csv").open(encoding="utf8") as f:
        rows=[r for r in csv.DictReader(f) if r["role"]=="generated"]
    fig,axes=plt.subplots(1,3,figsize=(8.6,2.75),gridspec_kw={"width_ratios":[1.05,1.05,1.08]})
    for ax,xkey,ykey,title,threshold in [
            (axes[0],"initial_max_force_eV_per_A","final_max_force_eV_per_A","a   Maximum atomic force",.1),
            (axes[1],"initial_min_covalent_radius_ratio","final_min_covalent_radius_ratio","b   Shortest contact",.60)]:
        x=np.array([float(r[xkey]) for r in rows]);y=np.array([float(r[ykey]) for r in rows])
        conv=np.array([r["converged"]=="True" for r in rows])
        for chosen,marker,col,lab in [(conv,"o",TEAL,"Converged"),(~conv,"x",GOLD,"Step limit")]:
            ax.scatter(x[chosen],y[chosen],s=20,marker=marker,color=col,alpha=.9,label=lab,zorder=2)
        ax.set_title(title,fontsize=9,loc="left",fontweight="bold")
        ax.axhline(threshold,color=PLUM,ls=":",lw=1)
        lo,hi=min(x.min(),y.min()),max(x.max(),y.max())
        ax.plot([lo,hi],[lo,hi],ls="--",color="#9aa6a9",lw=.8,zorder=0)
        if ax is axes[0]:
            ax.set(xscale="log",yscale="log",xlabel="Initial (eV/Å)",ylabel="Relaxed (eV/Å)")
        else:ax.set(xlabel="Initial distance / sum of radii",ylabel="Relaxed distance / sum of radii")
    ax=axes[2];measured=sorted((float(r["delta_vs_reference_final_eV_per_atom"]),r["sample_id"]) for r in rows if r["delta_vs_reference_final_eV_per_atom"] not in ("","nan"))
    ax.axhline(0,color=GREY,lw=.7)
    ax.scatter(np.arange(1,len(measured)+1),[x[0] for x in measured],s=15,color=PLUM)
    ax.set_yscale("symlog",linthresh=.03)
    ax.annotate(measured[-1][1],xy=(23,measured[-1][0]),xytext=(5,4.5),fontsize=7.5,
                arrowprops={"arrowstyle":"-","lw":.7,"color":GREY})
    ax.set(xlabel="Rank among 23 references",ylabel="Candidate − reference (eV/atom)")
    ax.set_title("c   Same-formula energy",loc="left",fontsize=9,fontweight="bold")
    axes[0].legend(fontsize=7,frameon=False,loc="upper left")
    fig.tight_layout(w_pad=1)
    save(fig,"revision-relaxation")


if __name__=="__main__":
    workflow()
    screen()
