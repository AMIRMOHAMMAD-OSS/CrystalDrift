"""Editable orthographic rendering of the first multispecies C-angular draw and its relaxed cell."""
import itertools
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ase.data import chemical_symbols
from charts import HERE
from labels import ROOT
from geometry import cohort
sys.path.insert(0,str(ROOT))
from data import unpack_gram

def main():
    record=next(r for r in cohort("test")[:16] if len({z for z,l in r["template"]})>=2 and r["group"]>1)
    i=next(i for i,r in enumerate(cohort("test")) if r["id"]==record["id"])
    name=f"angular-C-{i:03}"
    src=json.loads((HERE/f"relaxation/inputs/{name}.json").read_text())
    out=json.loads((HERE/f"relaxation/records/{name}.json").read_text())
    z=np.asarray(src["z"]);paper=HERE.parent/"paper"
    stages=[("Generated",src["k"],src["f"]),("After CHGNet",out["final_k"],out["final_f"])]
    plt.rcParams.update({"svg.fonttype":"none","font.family":"DejaVu Sans","font.size":8})
    fig,axes=plt.subplots(1,2,figsize=(6.5,2.75))
    q=np.array([[1,-1,0],[.55,.55,-1],[1,1,1.1]],float);q/=np.linalg.norm(q,axis=1)[:,None]
    elements=np.unique(z);colors=dict(zip(elements,["#187C85","#CF8E33","#8C529D","#3073AE","#BD4852","#598457"]))
    corners=np.array(list(itertools.product([0.,1.],repeat=3)));extent=0.;allpts=[]
    for _,k,f in stages:
        L=np.linalg.cholesky(unpack_gram(np.array(k)));cc=(corners-.5)@L@q.T
        points=(np.asarray(f)%1-.5)@L@q.T;allpts.append((cc,points));extent=max(extent,abs(cc[:,:2]).max())
    for ax,(lab,_,_),(c,p) in zip(axes,stages,allpts):
        for j in range(8):
            for m in range(j+1,8):
                if abs(corners[j]-corners[m]).sum()==1:ax.plot(c[[j,m],0],c[[j,m],1],lw=.8,color="#8396A0",zorder=0)
        ids=np.argsort(p[:,2]);ax.scatter(p[ids,0],p[ids,1],s=58,c=[colors[a] for a in z[ids]],edgecolor="white",linewidth=.8)
        ax.set(xlim=(-extent*1.07,extent*1.07),ylim=(-extent*1.07,extent*1.07),aspect="equal",title=lab);ax.axis("off")
    fig.legend([Line2D([],[],marker="o",ls="",c=colors[a]) for a in elements],[chemical_symbols[a] for a in elements],
               loc="lower center",ncol=len(elements),frameon=False)
    fig.subplots_adjust(left=.02,right=.98,top=.90,bottom=.16,wspace=.04)
    fig.savefig(paper/"wyckoff-example.pdf");fig.savefig(HERE/"wyckoff-example.svg")
    (HERE/"example.json").write_text(json.dumps(dict(target=record["id"],source=name,group=record["group"],status=out["status"],
        raw_min_contact=out["initial"]["min_contact"],final_min_contact=out["final"]["min_contact"]),indent=2)+"\n")
if __name__=="__main__":main()
