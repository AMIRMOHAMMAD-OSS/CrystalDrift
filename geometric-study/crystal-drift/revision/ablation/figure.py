"""Draw the eight-site Fourier collision as editable PDF and SVG vector artwork."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]/"paper"
mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":7.6,"axes.labelsize":8,
                     "axes.titlesize":9,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.linewidth":.64,"axes.edgecolor":"#707477","xtick.color":"#404447",
                     "ytick.color":"#404447","text.color":"#17232a","pdf.fonttype":42,
                     "svg.fonttype":"none","svg.hashsalt":"CrytalDrift-Fourier-v1"})
BLUE="#1c6682"; ORANGE="#bd5b2d"; INK="#25343a"; LIGHT="#c6cdcf"
SETS=((0,1,4,6),(0,1,3,7))


def make():
    fig=plt.figure(figsize=(7.05,3.24))
    grid=fig.add_gridspec(2,2,left=.09,right=.97,top=.88,bottom=.19,
                          width_ratios=[1.55,1],hspace=1.12,wspace=.52)
    for i,s in enumerate(SETS):
        ax=fig.add_subplot(grid[i,0])
        ax.set_title(("a   Motif I","b   Motif II")[i],loc="left",fontweight="semibold",pad=4)
        ax.set_xlim(-.55,11.65);ax.set_ylim(-.4,1.4)
        for y,marker,color in ((0,"o",BLUE),(1,"D",ORANGE)):
            ax.hlines(y,-.35,11.35,color=LIGHT,lw=.7,zorder=0)
            ax.scatter(np.arange(12),np.full(12,y),s=9,facecolor="white",edgecolor=LIGHT,lw=.65,zorder=1)
            ax.scatter(s,np.full(4,y),s=37,marker=marker,color=color,lw=.55,edgecolor="white",zorder=2)
        ax.set_yticks([0,1],["O","Si"]);ax.tick_params(axis="both",length=0,pad=2)
        ax.set_xticks([0,3,6,9],["0","¼","½","¾"]);ax.set_xlabel("Fractional position along the first lattice vector",labelpad=3)
        ax.spines["left"].set_visible(False);ax.spines["bottom"].set_visible(False)
    ax=fig.add_subplot(grid[0,1])
    alphas=[]
    h=np.arange(7)
    for s in SETS:
        alphas.append(np.exp(2j*np.pi*np.outer(s,h)/12).mean(0))
    ax.set_title("c   Identical species powers",loc="left",fontweight="semibold",pad=4)
    ax.plot(h,np.abs(alphas[0])**2,color=INK,lw=.8)
    ax.scatter(h,np.abs(alphas[0])**2,s=29,facecolor="white",edgecolor=INK,lw=.85,label="I")
    ax.scatter(h,np.abs(alphas[1])**2,s=8,marker="x",color=INK,lw=.8,label="II")
    ax.set(xlim=(-.3,6.3),ylim=(-.1,1.18),xticks=(0,2,4,6),yticks=(0,.5,1),xlabel="Axial harmonic",ylabel=r"$|a_{\mathrm{O}}(h)|^2$")
    ax.legend(loc="upper right",ncol=2,handletextpad=.1,columnspacing=.7,borderpad=0,frameon=False,fontsize=7)
    ax.tick_params(length=2,pad=2)
    ax=fig.add_subplot(grid[1,1])
    ax.set_title("d   Signed collinear bispectrum",loc="left",fontweight="semibold",pad=4)
    ax.axvline(0,color=LIGHT,lw=.7)
    values=np.array([-1,1])*np.sqrt(3)/32
    for i,v in enumerate(values):
        ax.hlines(1-i,0,v,color=INK,lw=.9)
        ax.scatter(v,1-i,s=31,facecolor=INK if i else "white",edgecolor=INK,lw=.9,zorder=3)
    ax.set(xlim=(-.074,.074),ylim=(-.5,1.6),yticks=(0,1),yticklabels=("II","I"),
           xticks=(-.05,0,.05),xticklabels=("−0.05","0","+0.05"))
    ax.set_xlabel(r"$\mathrm{Re}\,[a_{\mathrm{O}}(e_1)^2\,\overline{a_{\mathrm{Si}}(2e_1)}]$",labelpad=3)
    ax.tick_params(length=2,pad=2);ax.spines["left"].set_visible(False)
    for ext in ("pdf","svg"):
        fig.savefig(PAPER/("revision-fourier-collision."+ext),bbox_inches="tight",pad_inches=.045)
    plt.close(fig)


if __name__=="__main__":
    make()
