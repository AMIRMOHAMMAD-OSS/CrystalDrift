from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.6,"axes.labelsize":8.5,"xtick.labelsize":7.5,"ytick.labelsize":8,
                     "axes.linewidth":.65,"axes.edgecolor":"#a7b3bb","axes.labelcolor":"#233844",
                     "xtick.color":"#425564","ytick.color":"#425564","text.color":"#162e3c",
                     "svg.fonttype":"none","pdf.fonttype":42,"figure.facecolor":"white","savefig.facecolor":"white"})
colors={"drift":"#155579","flow1":"#bd6533","flow16":"#348678","retrieval":"#665581","uflow":"#43484d"}
names={"drift":"Periodic drift","flow1":"Euler (1)","flow16":"Euler (16)","retrieval":"Retrieval","uflow":"uFlowCSP"}
markers={"drift":"o","flow1":"s","flow16":"^","retrieval":"D","uflow":"x"}
def save(fig,name):
    for typ in ("pdf","svg","png"):
        fig.savefig(HERE/(name+"."+typ),dpi=220)
    plt.close(fig)
def tidy(ax):
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.tick_params(length=3,pad=3)
def heading(ax,letter,text):
    ax.set_title(letter+"   "+text,loc="left",fontweight="bold",fontsize=9.4,pad=9)
def table(name):
    with (HERE/name).open(encoding="utf8",newline="") as f: return list(csv.DictReader(f))
def motifs():
    fig=plt.figure(figsize=(7.0,3.25))
    grid=fig.add_gridspec(2,3,width_ratios=(1.42,.98,1),hspace=.64,wspace=.53,left=.09,right=.97,bottom=.22,top=.82)
    sets=([0,1,4,6],[0,1,3,7])
    primary="#1c6689"; second="#bc6835"
    for i,s in enumerate(sets):
        ax=fig.add_subplot(grid[i,0]); ax.set(xlim=(-.45,12.45),ylim=(-.15,.48))
        for y,mark,c in [(0,"o",primary),(1/3,"s",second)]:
            ax.plot([0,12],[y,y],c="#d1dbe0",lw=.9,zorder=1)
            ax.scatter(s,[y]*4,marker=mark,s=29,c=c,edgecolors='white',linewidths=.4,zorder=3)
            ax.scatter([12],[y],marker=mark,s=25,facecolors="white",edgecolors=c,linewidths=.85,zorder=3)
        for x in (0,12):ax.axvline(x,lw=.6,c="#a9b6bf",ls=":",zorder=0)
        ax.set_yticks([0,1/3],["A","B"]);ax.set_xticks([0,3,6,9,12]); ax.tick_params(left=False,bottom=True,length=2)
        for spine in ax.spines.values():spine.set_visible(False)
        if not i:
            heading(ax,"A","Decorated motifs");ax.tick_params(labelbottom=False)
        else: ax.set_xlabel("Position along cell, 12$x$")
        ax.text(.04,.5,"$S_"+str(i+1)+"$",transform=ax.transAxes,fontsize=8,ha="left",va="center")
    ax=fig.add_subplot(grid[:,1]);tidy(ax);heading(ax,"B","Equal powers")
    freq=np.arange(1,7)
    powers=[]
    for s in sets:
        a=np.exp(2j*np.pi*np.outer(freq,np.array(s))/12).mean(1)
        powers.append(abs(a)**2)
    ax.vlines(freq,0,powers[0],color="#bdcbd2",lw=1.2)
    ax.scatter(freq,powers[0],s=43,facecolors='white',edgecolors=primary,label='$S_1$',lw=1.25)
    ax.scatter(freq,powers[1],s=14,marker='x',c=second,label='$S_2$',lw=1)
    ax.set(xlim=(.55,6.45),ylim=(0,.32),xlabel='Frequency $k$',ylabel='$|a_A(k e_1)|^2$');ax.set_xticks([1,2,3,4,5,6]);ax.set_yticks([0,.1,.2,.3])
    ax.legend(loc="upper center",ncol=2,frameon=False,fontsize=7,handlelength=1,columnspacing=.8)
    ax=fig.add_subplot(grid[:,2]);tidy(ax);heading(ax,"C","Omitted triple")
    vals=[]
    for s in sets:
        sites=np.array(s)/12
        vals.append(np.real(np.exp(2j*np.pi*sites).mean()**2*np.conj(np.exp(4j*np.pi*sites).mean())))
    ax.axhline(0,color="#96a7ae",lw=.85)
    ax.vlines([0,1],0,vals,color=primary,lw=2)
    ax.scatter([0,1],vals,s=31,c=primary,zorder=3)
    ax.set(xlim=(-.5,1.5),ylim=(-.083,.083),ylabel='$\\mathrm{Re}\\, B_{AAB}(e_1)$');ax.set_xticks([0,1],["$S_1$","$S_2$"]);ax.set_yticks([-.05,0,.05],["−0.05","0","0.05"])
    for j,v in enumerate(vals):ax.annotate(('$-' if j==0 else '$+')+r'\sqrt{3}/32$',(j,v),xytext=(0,-14 if j==0 else 7),textcoords="offset points",ha='center',fontsize=7.3)
    save(fig,"01_motif_and_fourier_features")
def budgets():
    sr=table("binary_seed_prefixes.csv"); records={}
    for row in sr:
        records.setdefault((row['method'],int(row['seed'])),[]).append(row)
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.55),gridspec_kw=dict(width_ratios=[.88,1.2]))
    fig.subplots_adjust(left=.082,right=.98,bottom=.24,top=.77,wspace=.32)
    ts=sorted({int(r['seed']) for r in sr if r['method']=='drift'})
    for i,ax in enumerate(axes):
        tidy(ax);ax.set_ylim(-2,86);ax.set_yticks([0,20,40,60,80]);ax.set_axisbelow(True);ax.grid(axis='y',color='#e2e9ec',lw=.6)
    ax=axes[0];heading(ax,'A','Equal numbers of proposals')
    for m in names:
        fits=[sorted(rows,key=lambda r:int(r['k'])) for (mm,_),rows in records.items() if mm==m]
        xs=np.arange(1,6); ys=np.mean([[float(r['percent']) for r in rows] for rows in fits],0)
        ax.plot(xs,ys,marker=markers[m],ms=3.9,color=colors[m],lw=1.4,label=names[m])
    ax.set(xlabel="Proposals per target",ylabel="Targets recovered (%)",xticks=[1,2,3,4,5])
    ax=axes[1];heading(ax,'B','Estimated serial CPU budget')
    for m in ['drift','flow1','flow16','retrieval']:
        prof=[]
        for s in ts:
            rs=sorted(records[m,s],key=lambda r:int(r['k']))
            prof.append((float(rs[0]['median_ms']),np.r_[0,[float(r['percent']) for r in rs]]))
        events=sorted({v*k for v,ys in prof for k in range(1,6)} | {.045,30.})
        ys=[np.mean([rr[min(5,int(np.floor(t/v+1e-11)))] for v,rr in prof]) for t in events]
        ax.step(events,ys,where='post',color=colors[m],lw=1.4)
        if m=='flow16':
            y5=np.mean([rr[min(5,int(5/v))] for v,rr in prof])
            ax.scatter([5],[y5],c=colors[m],s=20,zorder=4,edgecolors='white',linewidths=.6)
            ax.annotate(f'{y5:.1f}%',(5,y5),xytext=(9,-7),textcoords='offset points',color=colors[m],fontsize=7.2)
    ax.axvline(5,ls='--',lw=.9,color='#66747c')
    ax.text(5.25,68,'5 ms',va='bottom',ha='left',fontsize=7.5,color='#415460')
    ax.set_xscale('log');ax.set_xlim(.045,30);ax.xaxis.set_major_locator(FixedLocator([.05,.2,1,5,20]));ax.xaxis.set_major_formatter(FixedFormatter(["0.05","0.2","1","5","20"]));ax.set_xlabel("Time budget (ms, at most five proposals)")
    handles=[Line2D([],[],color=colors[m],marker=markers[m],lw=1.35,ms=4,label=names[m]) for m in names]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.50,-.003),frameon=False,ncol=5,handlelength=1.35,columnspacing=1.2,fontsize=7.8)
    save(fig,"02_recovery_and_time")
def recovery():
    data=table("native_recovery.csv")
    fig,axes=plt.subplots(1,2,figsize=(7.25,3.62),sharey=True)
    fig.subplots_adjust(left=.17,right=.97,top=.79,bottom=.26,wspace=.20)
    for j,(ax,cohort,title) in enumerate(zip(axes,('mp20','wbm'),('MP-20','WBM'))):
        tidy(ax);heading(ax,chr(65+j),title+'   ·   128 targets')
        for i,m in enumerate(names):
            row=next(r for r in data if r['cohort']==cohort and r['method']==m)
            original,volume,strict=map(float,(row['matcher_pct'],row['volume_pct'],row['max_site_pct']))
            if i:ax.axhline(i-.5,color='#e2e8ec',lw=.55,zorder=0)
            ax.scatter([original],[i-.20],s=32,facecolors='white',edgecolors=colors[m],lw=1.3,zorder=3)
            ax.scatter([strict],[i],s=27,c=colors[m],marker='|',linewidths=2,zorder=4)
            ax.scatter([volume],[i+.20],s=24,c=colors[m],marker='D',zorder=4)
        ax.set(xlim=(-2,101),ylim=(4.55,-.55),xticks=[0,25,50,75,100],xlabel="Targets recovered in five draws (%)")
        ax.set_yticks(np.arange(5),[names[m] for m in names]);ax.tick_params(axis='y',length=0)
        ax.grid(axis='x',color='#e1e8eb',lw=.55);ax.set_axisbelow(True);ax.spines['left'].set_visible(False)
    handles=[Line2D([],[],marker='o',linestyle='',markeredgecolor='#284450',markerfacecolor='white',label='RMS match'),
             Line2D([],[],marker='|',markersize=7,markeredgewidth=1.8,linestyle='',color='#284450',label='Maximum-site match'),
             Line2D([],[],marker='D',markersize=4,linestyle='',color='#284450',label='RMS match + volume agreement')]
    fig.legend(handles=handles,loc='lower center',frameon=False,ncol=3,bbox_to_anchor=(.51,-.005),fontsize=7.8,columnspacing=1.15,handletextpad=.4)
    save(fig,"03_recovery_definitions")
if __name__=='__main__':
    motifs();budgets();recovery()

