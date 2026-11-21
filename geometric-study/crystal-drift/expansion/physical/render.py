"""Render matched before/after multielement CIFs using physical positions at a common scale."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex
from ase.data.colors import jmol_colors
from ase.data import chemical_symbols
from pymatgen.core.composition import Composition

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/"figures-v2"),str(ROOT)]
import crystal_render as render


def main():
    rows=list(csv.DictReader((HERE/"per-candidate.csv").open()))
    def many(r):
        src=json.loads((HERE/f"inputs/{r['id']}.json").read_text())
        return len(set(src["z"]))>=3
    chosen=[]
    for arm in ("learned-drift","learned-flow16","crystalflow-joint"):
        eligible=[r for r in rows if r["arm"]==arm and r["converged"]=="True" and r["contacts_ok"]=="True" and many(r)]
        if eligible:chosen.append(eligible[0])
    failures=[r for r in rows if (r["converged"]!="True" or r["contacts_ok"]!="True") and "final_energy" in r and r["final_energy"] and many(r)]
    if failures:chosen.append(failures[0])
    scenes={};metadata=[]
    for number,r in enumerate(chosen):
        src=json.loads((HERE/f"inputs/{r['id']}.json").read_text())
        rec=json.loads((HERE/f"records/{r['id']}.json").read_text())
        z=np.array(src["z"])
        for e in set(z):
            render.COLOR[chemical_symbols[e]]=to_hex(jmol_colors[e])
        for j,(k,f) in enumerate(((src["k"],src["f"]),(rec["final_k"],rec["final_f"]))):
            scenes[number,j]=render.scene(np.array(k),np.array(f),z)
        comp=Composition({chemical_symbols[e]:int((z==e).sum()) for e in np.unique(z)})
        metadata.append(dict(id=r["id"],arm=r["arm"],draw=r["draw"],formula=comp.reduced_formula,
            symbols=[chemical_symbols[e] for e in np.unique(z)],converged=r["converged"],min_contact=float(r["final_min_contact"]),status=r["status"]))
    extent=max(np.max(s["halfextent"]) for s in scenes.values())+.1
    fig,axes=plt.subplots(2,len(chosen),figsize=(8.4,4.65),squeeze=False)
    mpl.rcParams["pdf.fonttype"]=42
    for i,entry in enumerate(metadata):
        for j in (0,1):
            ax=axes[j,i];ax.imshow(render.render_scene(scenes[i,j],extent));ax.axis("off")
        subscript = str.maketrans("0123456789","₀₁₂₃₄₅₆₇₈₉")
        axes[0,i].set_title(entry["formula"].translate(subscript),fontweight="bold",fontsize=12,pad=17)
        names=entry["symbols"];handles=[Line2D([],[],marker="o",linestyle="",markerfacecolor=render.COLOR[e],markeredgecolor="#65737b",markeredgewidth=.35,label=e,markersize=5) for e in names]
        axes[0,i].legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.95),ncol=len(names),frameon=False,fontsize=7,handlelength=.7,columnspacing=.65)
        armname={"learned-drift":"CrytalDrift","empirical-drift":"CrytalDrift (empirical)","learned-flow16":"16-step flow","crystalflow-joint":"CrystalFlow","learned-uflow":"uFlowCSP"}[entry["arm"]]
        axes[1,i].text(.5,-.02,f"{armname} · draw {entry['draw']}\n{entry['status'].replace('step_cap','step limit')} · {entry['min_contact']:.2f} Å",ha="center",va="top",transform=axes[1,i].transAxes,fontsize=7.2,color="#932e23" if entry["status"]!="converged" else "#274758")
    for j,letter in enumerate(("As generated","CHGNet-relaxed")):axes[j,0].text(-.035,.5,letter,rotation=90,transform=axes[j,0].transAxes,va="center",ha="right",fontsize=9,fontweight="bold")
    fig.subplots_adjust(left=.058,right=.993,top=.87,bottom=.13,wspace=.01,hspace=.07)
    ax=axes[1,0]
    ax.plot([.05,.05+2/(2*extent)],[.07,.07],lw=2.3,color="#27384a",transform=ax.transAxes)
    ax.text(.05,.10,"2 Å",fontsize=7,transform=ax.transAxes)
    fig.savefig(ROOT/"paper/exp-structures.pdf")
    fig.savefig(HERE/"structures.png",dpi=220)
    (HERE/"figure-selection.json").write_text(json.dumps(dict(selection=metadata,viewport_width_angstrom=2*extent),indent=2)+"\n")


if __name__=="__main__":main()
