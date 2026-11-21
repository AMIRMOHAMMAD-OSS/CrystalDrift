"""Draw the inference and training map. All symbols in the diagram are editable SVG text."""
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

HERE=Path(__file__).resolve().parent
mpl.rcParams.update({"svg.fonttype":"none","pdf.fonttype":42,"font.family":"DejaVu Sans",
                     "mathtext.fontset":"dejavusans","font.size":9})
fig,ax=plt.subplots(figsize=(8.2,4.0))
fig.subplots_adjust(left=.014,right=.987,top=.978,bottom=.015)
ax.set(xlim=(0,10),ylim=(0,5.4));ax.axis("off")
edge="#234d66"; ink="#142d3d"; tan="#a55c29"


def box(x,y,w,h,title,lines,bg,color=edge):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.045,rounding_size=0.12",
                                facecolor=bg,edgecolor=color,linewidth=1.05))
    ax.text(x+.12,y+h-.19,title,va="top",fontweight="bold",fontsize=9.2,color=color)
    ax.text(x+.12,y+h-.52,lines,va="top",fontsize=8.1,color=ink,linespacing=1.55)


def arrow(x1,y1,x2,y2,color=edge):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",linewidth=1,
                                 mutation_scale=10,color=color,shrinkA=0,shrinkB=0))


ax.text(.08,5.08,"Sampling",fontsize=12,fontweight="bold",color=edge)
ax.text(1.32,5.08,"one neural evaluation per crystal",color=ink,fontsize=9)
box(.12,3.86,2.50,.91,"A  ·  Composition","Draw a physical cell and sites.","#eff6f9")
box(.12,2.70,2.50,.94,"C  ·  Occupied template","Draw a compatible cell and\nexpand Wyckoff representatives.","#eff6f9")
box(3.13,3.05,2.52,1.51,"Periodic graph","Enumerate physical images.\nPass species and radial messages.\nInclude nonzero self-images.","#ebf5f4")
box(6.13,3.05,3.67,1.51,"Joint geometric update",
    "Edge vectors move sites in the seed cell.\nEdge dyads and isotropic strain deform the cell.\n"+r"$F'=(F+vL^{-1})\,\mathrm{mod}\,1$     $L'=L\exp E$","#e9f2ee")
arrow(2.70,4.29,3.03,3.99);arrow(2.70,3.15,3.03,3.61);arrow(5.72,3.81,6.04,3.81)
ax.plot([.10,9.82],[2.42,2.42],color="#b6c6cd",lw=.8)
ax.text(.08,2.09,"Training",fontsize=12,fontweight="bold",color=tan)
ax.text(1.32,2.09,"references share the query's conditioning information",color=ink,fontsize=9)
box(.12,.34,2.50,1.44,"Finite reference banks","Positive: sampled training cells\nNegative: independent draws\nQuery: differentiable draws","#fcf5eb",tan)
box(3.13,.34,2.52,1.44,"Physical features","Cell volume, species and shells\nCenter-wise harmonic products\nPool by central element.","#fcf5eb",tan)
box(6.13,.34,3.67,1.44,"Stopped feature target","Laplace weights set attraction and repulsion.\nThe query feature receives the gradient.\nInference uses only the trained generator.","#fcf5eb",tan)
arrow(2.70,1.00,3.03,1.00,tan);arrow(5.72,1.00,6.04,1.00,tan)
fig.savefig(HERE/"method.svg",facecolor="white")
fig.savefig(HERE.parent/"paper/method.pdf",facecolor="white")
