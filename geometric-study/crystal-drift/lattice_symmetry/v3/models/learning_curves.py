"""Regenerate validation recovery by exposure and measured elapsed training time."""
import csv,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .sampler import HERE,V3,weight_path

def rows():
    result=[]
    methods=['angular-laplace-11','contacts-laplace-11','meanflow-laplace-11-commonorder']
    for size in ('small','medium'):
        for method in methods:
            tag=size+'-'+method;fold=HERE/'fits'/tag
            for k,name in [('half',.5),('full',1),('epoch4',4),('epoch8',8)]:
                base=fold if name<=1 else fold.with_name(tag+'-8pass')
                sfile=base/(k+'-dev-scores.npz')
                if not sfile.exists():continue
                with np.load(sfile) as q:
                    n=len(q['rms']);hits=int(np.isfinite(q['rms']).any(1).sum())
                c=torch.load(weight_path(base/(k+'.pt')),weights_only=True,map_location='cpu')
                result.append(dict(tier=size,method=method.split('-')[0],exposure=name,checkpoint=k,
                    anchors=c['anchors'],updates=c['step'],seconds=c['seconds'],validation_hits=hits,targets=n))
    for size in ('small','medium'):
        fold=V3/'uflow/fits'/(size+'-11')
        for base in [fold,fold.with_name(fold.name+'-long')]:
            sel=base/'selection.json'
            if not sel.exists():continue
            data=json.loads(sel.read_text()).get('results',{})
            for name,score in data.items():
                meta=base/(Path(name).stem+'.json')
                if not meta.exists() or not isinstance(score,int):continue
                c=json.loads(meta.read_text());fraction=c['seen']/26614
                if base!=fold and fraction<=1:continue
                result.append(dict(tier=size,method='uflow',exposure=round(fraction,1),checkpoint=Path(name).stem,
                     anchors=c['seen'],updates=c['step'],seconds=c['wall_seconds'],validation_hits=score,targets=96))
    return result

def main():
    table=rows();dst=HERE/'learning-curves.csv'
    with dst.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    fig,ax=plt.subplots(1,2,figsize=(7.3,3.05),sharey=True)
    labels={'angular':'Angular drift','contacts':'Contact drift','meanflow':'Orbit MeanFlow','uflow':'Retrained uFlowCSP'}
    colors={'angular':'#277d83','contacts':'#b66b26','meanflow':'#ad4772','uflow':'#625d9d'}
    markers={'angular':'o','contacts':'s','meanflow':'^','uflow':'D'}
    for pane,size in zip(ax,('small','medium')):
        pane.axhline(100*59/96,color='#777777',lw=.8,ls=':',label='Symbolic prior')
        for method,label in labels.items():
            group=sorted([r for r in table if r['tier']==size and r['method']==method],key=lambda z:z['exposure'])
            if group:pane.plot([r['exposure'] for r in group],[100*r['validation_hits']/r['targets'] for r in group],
                   color=colors[method],marker=markers[method],markerfacecolor='white',ms=5,lw=1.4,label=label)
        pane.set_xscale('log',base=2);pane.set_xticks([.5,1,4,8],['0.5','1','4','8'])
        pane.set_xlim(.43,10);pane.set_xlabel('Training traversals');pane.set_title('0.2M parameters' if size=='small' else '2M parameters')
        pane.grid(axis='y',alpha=.18);pane.spines[['top','right']].set_visible(False)
    ax[0].set_ylabel('Development MR@5 (%)')
    handles,labs=ax[1].get_legend_handles_labels();fig.legend(handles,labs,ncol=3,frameon=False,loc='lower center',fontsize=8)
    fig.subplots_adjust(left=.09,right=.985,top=.88,bottom=.31,wspace=.12)
    for ext in ('svg','pdf'):fig.savefig(HERE/('learning-curves.'+ext))
    plt.close(fig)
if __name__=='__main__':main()

