"""Vector workflow of the implemented binary CrytalDrift generator and training loss."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,FancyArrowPatch,FancyBboxPatch,Rectangle
from matplotlib.path import Path as PlotPath


OUT=Path(__file__).resolve().parent
INK='#182b39';MUTED='#4a6270';LINE='#8699a4'
TEAL='#086b66';PURPLE='#69438e';ORANGE='#ad642b';BLUE='#326d9c'


def box(ax,x,y,w,h,color='#ffffff',edge='#acbbc2',dashed=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.0,rounding_size=0.86',
          facecolor=color,edgecolor=edge,linewidth=.85,linestyle=(0,(3.1,2.4)) if dashed else '-'))


def txt(ax,x,y,s,sz=8,color=INK,bold=False,align='left',**kwargs):
    return ax.text(x,y,s,ha=align,va='center',fontsize=sz,color=color,weight='semibold' if bold else 'normal',**kwargs)


def flow(ax,points,color=LINE,dashed=False):
    path=PlotPath(points,[PlotPath.MOVETO]+[PlotPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',color=color,linewidth=1.0,
         linestyle=(0,(2.4,1.7)) if dashed else '-',mutation_scale=7.0,joinstyle='round',capstyle='round'))


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none',
                         'mathtext.fontset':'dejavusans'})
    fig=plt.figure(figsize=(8.50,5.24),facecolor='white')
    ax=fig.add_axes([.012,.012,.976,.976]);ax.set_xlim(0,100);ax.set_ylim(0,72);ax.axis('off')
    txt(ax,1.05,68.65,'CrytalDrift',17.1,INK,True)
    txt(ax,26.1,68.65,'One network evaluation returns a binary crystal unit cell',9.5,MUTED)
    ax.plot([1.05,98.9],[65.2,65.2],color='#c7d3d7',lw=.7)

    txt(ax,1.05,61.9,'Generation',11.0,TEAL,True)
    y,h=41.55,16.8
    box(ax,1.05,y,19.05,h,'#f2f7f8','#becdd1')
    txt(ax,2.5,55.9,'Inputs',9.45,INK,True)
    txt(ax,2.5,52.8,'Cell composition',8.1,INK,True)
    txt(ax,2.5,50.24,'Given counts, or sample',7.25,MUTED)
    txt(ax,2.5,48.08,'the de novo chemistry prior',7.05,MUTED)
    txt(ax,2.5,44.62,'Gaussian cell noise',7.35,INK)
    txt(ax,2.5,42.88,'+ uniform site noise',7.35,INK)
    flow(ax,[(20.5,50.0),(23.58,50.0)],TEAL)

    box(ax,24.0,y,27.7,h,'#eaf5f2','#5c9e94')
    txt(ax,25.6,55.9,'Generator  '+r'$G_\theta$',9.7,TEAL,True)
    txt(ax,25.6,52.91,'Element embeddings + relative phases',7.25,INK)
    txt(ax,25.6,50.05,'Shared site MLP',8.05,INK,True)
    # The same site MLP is applied to each token; the set is pooled before two output heads.
    for col in range(4):
        x0=46.2+1.1*col
        for row in range(3):
            if col<3:ax.plot([x0,x0+1.1],[50.15+row*.49,50.15+(2-row)*.49],color='#82aaa7',lw=.35)
            ax.add_patch(Circle((x0,50.15+row*.49),.17,facecolor=TEAL,edgecolor='none'))
    txt(ax,25.6,47.23,'Masked mean gives cell context',7.55,INK)
    txt(ax,25.6,44.22,'Site + context',7.0,MUTED)
    txt(ax,39.1,44.22,'Cell context',7.0,MUTED)
    txt(ax,25.6,42.59,'position head',7.25,TEAL,True)
    txt(ax,39.1,42.59,'lattice head',7.25,TEAL,True)
    flow(ax,[(52.15,50.0),(55.33,50.0)],TEAL)

    box(ax,55.65,y,21.15,h,'#f2f7f8','#aebfc6')
    txt(ax,57.1,55.9,'Returned crystal',9.0,INK,True)
    txt(ax,57.1,52.74,'Fractional coordinates',7.7,INK,True)
    txt(ax,57.1,50.05,r'$f_i=(u_i+\Delta_i)\ \mathrm{mod}\ 1$',8.8,TEAL)
    txt(ax,57.1,46.65,'Decoded lattice and metric',7.15,INK,True)
    txt(ax,57.1,43.95,r'$L=\exp(S),\ G=LL^\mathsf{T}$',8.6,TEAL)
    flow(ax,[(77.15,50.0),(80.5,50.0)],ORANGE,True)
    box(ax,80.9,y,18.0,h,'#fff8ed','#c39560',True)
    txt(ax,82.25,55.9,'Physical screen',9.,ORANGE,True)
    txt(ax,82.25,52.61,'After generation',7.3,MUTED)
    txt(ax,82.25,49.21,'CHGNet relaxes',7.7,INK,True)
    txt(ax,82.25,46.93,'cell and sites.',7.65,INK)
    txt(ax,82.25,43.90,'Assess forces and energy.',7.05,INK)

    ax.plot([1.05,98.9],[38.65,38.65],color='#c7d3d7',lw=.7)
    txt(ax,1.05,35.67,'Training',10.95,PURPLE,True)
    txt(ax,13.2,35.67,'The reference banks may be pooled or restricted to the exact cell composition.',7.95,MUTED)
    x,w=1.05,21.0
    for yy,label,desc,ink,fill in [(25.65,'Generated query','One draw from the generator',TEAL,'#eaf5f2'),
            (16.0,'Real references','MP-20 training crystals',BLUE,'#eef5fc'),
            (6.35,'Generated references','Independent generator draws',ORANGE,'#fcf3e9')]:
        box(ax,x,yy,w,7.45,fill,ink);txt(ax,x+1.15,yy+4.85,label,8.3,ink,True)
        txt(ax,x+1.15,yy+2.0,desc,7.0,MUTED)
        flow(ax,[(x+w+.25,yy+3.72),(25.55,yy+3.72)],ink)
    box(ax,25.95,6.35,24.5,26.75,'#f5f2f8','#aa95ba')
    txt(ax,27.45,29.99,'Shared feature map  '+r'$\Phi$',8.85,PURPLE,True)
    txt(ax,27.45,26.33,'Standardized log-cell chart',7.72,INK)
    txt(ax,27.45,22.9,'Species-resolved periodic',7.72,INK)
    txt(ax,27.45,20.72,'Fourier cross-powers',7.72,INK)
    ax.plot([27.5,48.6],[17.65,17.65],color='#d5cadc',lw=.6)
    txt(ax,27.45,14.72,'Multisite model also uses',7.25,MUTED)
    txt(ax,27.45,12.35,'within-species powers and',7.25,MUTED)
    txt(ax,27.45,10.02,'mixed bispectra.',7.25,MUTED)

    # Live query bypass to the objective. The individual references and the query value
    # enter the field without an autograd edge, exactly as in train.py and model.py.
    flow(ax,[(50.8,29.8),(79.15,29.8)],TEAL)
    txt(ax,60.6,31.4,'Query features '+r'$x$',7.35,TEAL,True)
    flow(ax,[(53.0,29.4),(53.0,22.3),(55.75,22.3)],PURPLE,True)
    flow(ax,[(50.8,18.55),(55.75,18.55)],BLUE,True)
    flow(ax,[(50.8,11.3),(55.75,11.3)],ORANGE,True)
    box(ax,56.1,6.35,19.6,18.4,'#f4eff7','#af95bf')
    txt(ax,57.4,22.32,'Detached drift field',8.0,PURPLE,True)
    txt(ax,57.4,19.0,'Attract to real',7.72,BLUE,True)
    txt(ax,57.4,16.55,'Repel from generated',7.25,ORANGE,True)
    txt(ax,57.4,12.82,'Normalize & average',7.3,INK)
    txt(ax,57.4,10.25,r'$\tau\in\{0.3,0.8\}$',8.6,PURPLE)
    flow(ax,[(76.05,16.6),(79.15,16.6)],PURPLE,True)
    box(ax,79.55,6.35,19.34,26.75,'#f4f7f8','#a5b9c2')
    txt(ax,80.9,26.12,'Training target',8.5,INK,True)
    txt(ax,80.9,22.75,r'$t=\mathrm{sg}(x+V)$',9.0,PURPLE)
    txt(ax,80.9,18.7,'Feature-space loss',8.0,INK,True)
    txt(ax,80.9,15.6,r'$\frac{1}{2}\|x-t\|^2$',9.8,INK)
    txt(ax,80.9,10.89,'Update the generator',7.23,TEAL,True)
    txt(ax,80.9,8.88,'through the query.',7.23,TEAL,True)
    ax.plot([1.15,4.1],[2.65,2.65],color=PURPLE,lw=1,ls=(0,(2.5,1.7)))
    txt(ax,4.85,2.65,'Training dashed arrows carry detached values.  sg = stop-gradient.',7.25,MUTED)
    for ext in ('pdf','svg','png'):fig.savefig(OUT/f'crytaldrift-workflow.{ext}',dpi=320,facecolor='white')
    plt.close(fig)
    inputs={p:hashlib.sha256((OUT.parent/p).read_bytes()).hexdigest() for p in ('model.py','train.py','de_novo.py','relax_candidates.py')}
    (OUT/'workflow-metadata.json').write_text(json.dumps({'model_name':'CrytalDrift','sources_sha256':inputs,
        'encoding':'Native vector typography, diagram connectors, formulae and layout. The optional potential-based screen only follows generation.',
        'generator':'19,733 parameters; cross-powers are part of the training feature map, while relative latent phases are used by the inference set network.'},indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()

