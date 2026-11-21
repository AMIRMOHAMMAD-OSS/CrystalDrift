"""Make a fully editable orthographic rendering of selected raw and relaxed periodic cells."""
import csv
import itertools
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
import numpy as np
from ase.data import chemical_symbols
from ase.data.colors import jmol_colors
from pymatgen.core import Composition

HERE=Path(__file__).resolve().parent
STORE=HERE/"screen"
V2=HERE.parent
mpl.rcParams.update({"svg.fonttype":"none","pdf.fonttype":42,"font.family":"DejaVu Sans"})


def unpack(k):
    i,j=np.triu_indices(3,1)
    s=np.diag(k[:3]);s[i,j]=s[j,i]=k[3:]/np.sqrt(2)
    lam,v=np.linalg.eigh(s)
    # All deposited physical CIFs use the Cholesky frame.
    return np.linalg.cholesky((v*np.exp(2*lam)[None,:])@v.T)


def render(ax,points,box,z,limit):
    for i,a in enumerate(box):
        for j,b in enumerate(box[i+1:],i+1):
            if (i^j) in (1,2,4):
                ax.plot([a[0],b[0]],[a[1],b[1]],color="#8899a5",lw=.8,alpha=.86,zorder=1)
    for i in np.argsort(points[:,2]):
        xy=points[i,:2]; color=np.asarray(jmol_colors[z[i]])
        radius=.36
        ax.add_patch(Circle(xy,radius,fc=color*.58,ec="#344451",lw=.32,zorder=2+int(np.argsort(np.argsort(points[:,2]))[i])))
        # Nested vector disks approximate a lit sphere while preserving a separate object for each atom.
        for j,f in enumerate(np.linspace(.94,.06,14)):
            offset=radius*(1-f)*np.array([-.34,.38])
            bright=np.clip(color*(1.03-.28*f)+(1-f)*.27,0,1)
            ax.add_patch(Circle(xy+offset,radius*f,fc=bright,ec="none",
                                zorder=2+int(np.argsort(np.argsort(points[:,2]))[i])+.02*j))
    ax.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect="equal");ax.axis("off")


def main():
    rows=list(csv.DictReader((STORE/"per-candidate.csv").open()))
    def data(r,dir):return json.loads((STORE/dir/(r["id"]+".json")).read_text())
    rows=sorted(rows,key=lambda r:(("angular","radial").index(r["arm"]),int(r["draw"])))
    eligible=[r for r in rows if len(set(data(r,"inputs")["z"]))>=3]
    chosen=[]
    for desired in (True,False):
        group=[r for r in eligible if (r["converged"]=="True" and r["contacts_ok"]=="True")==desired]
        if group:chosen.append(group[0])
    assert chosen
    orientation=np.array([[.91,-.19,.36],[.40,.43,-.81],[0,.882,.47]])
    orientation,_=np.linalg.qr(orientation)
    corners=np.asarray(list(itertools.product((0.,1.),repeat=3)))-.5
    scenes={};limit=0;meta=[]
    for n,r in enumerate(chosen):
        raw=data(r,"inputs");final=data(r,"records")
        z=np.asarray(raw["z"])
        meta.append(dict(id=r["id"],arm=r["arm"],draw=int(r["draw"]),source_id=raw["source_id"],
                     status=r["status"],converged=r["converged"]=="True",contact_ok=r["contacts_ok"]=="True",
                     symbols=[chemical_symbols[x] for x in np.unique(z)],screened=r["screened_sun"]=="True"))
        for j,(k,f) in enumerate(((raw["k"],raw["f"]),(final.get("final_k"),final.get("final_f")))):
            if k is None:continue
            proj=unpack(np.asarray(k))@orientation
            points=(np.asarray(f)%1-.5)@proj
            box=corners@proj
            limit=max(limit,np.abs(box[:,:2]).max()+.42)
            scenes[n,j]=(points,box,z)
    fig,axes=plt.subplots(2,len(chosen),figsize=(3.25*len(chosen)+.5,4.70),squeeze=False)
    fig.subplots_adjust(left=.095,right=.985,bottom=.10,top=.85,hspace=.04,wspace=.04)
    limit*=1.02
    for n,(r,info) in enumerate(zip(chosen,meta)):
        z=scenes[n,0][2]
        comp=Composition({chemical_symbols[i]:int((z==i).sum()) for i in np.unique(z)})
        formula=comp.reduced_formula.translate(str.maketrans("0123456789","₀₁₂₃₄₅₆₇₈₉"))
        axes[0,n].set_title(f"{formula}  ·  {info['arm']} draw {info['draw']}",fontsize=10,fontweight="bold",pad=20)
        handles=[Line2D([],[],marker="o",ls="",ms=5,mec="#43535d",mew=.4,mfc=jmol_colors[i],label=chemical_symbols[i]) for i in np.unique(z)]
        axes[0,n].legend(handles=handles,frameon=False,ncol=len(handles),fontsize=7.6,handletextpad=.2,columnspacing=.7,loc="lower center",bbox_to_anchor=(.5,.95))
        for j in (0,1):
            ax=axes[j,n]
            if (n,j) in scenes:render(ax,*scenes[n,j],limit)
            else:
                ax.axis("off");ax.text(.5,.5,"No final geometry",ha="center",transform=ax.transAxes)
        axes[1,n].text(.5,-.01,info["status"].replace("step_cap","step limit"),fontsize=8.3,ha="center",transform=axes[1,n].transAxes)
    for j,name in enumerate(("As generated","After relaxation")):
        axes[j,0].text(-.065,.5,name,rotation=90,ha="right",va="center",fontsize=9,transform=axes[j,0].transAxes)
    axes[1,0].plot([.02,.02+1/limit],[.06,.06],transform=axes[1,0].transAxes,lw=2,color="#203849")
    axes[1,0].text(.02,.075,"2 Å",transform=axes[1,0].transAxes,fontsize=7.5)
    (HERE/"figure-selection.json").write_text(json.dumps(dict(cases=meta,viewport_width_angstrom=2*limit,projection=orientation.tolist()),indent=2)+"\n")
    fig.savefig(V2/"paper/new-crystals.pdf")
    fig.savefig(V2/"figures/new-crystals.svg")


if __name__=="__main__":main()
