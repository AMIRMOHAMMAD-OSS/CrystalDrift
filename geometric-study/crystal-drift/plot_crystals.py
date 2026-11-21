"""Draw the first four distinct sampled formulas, then summarize all 32 CHGNet evaluations."""
import csv
import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

from compare import digest, write_json
from data import ELEMENTS, unpack_gram


HERE=Path(__file__).resolve().parent
DE_NOVO=HERE/'de-novo'
OUT=DE_NOVO/'figures'
# Fixed orthographic projection and origin choice for every structure and relaxed partner.
DEPTH=np.array([.80,-1.23,.83]); DEPTH/=np.linalg.norm(DEPTH)
RIGHT=np.cross([0,0,1],DEPTH); RIGHT/=np.linalg.norm(RIGHT)
UP=np.cross(DEPTH,RIGHT)
PROJECTION=np.stack([RIGHT,UP,DEPTH],axis=1)
CORNERS=np.asarray(list(product((0,1),repeat=3)))
# Discrete species colors, recorded explicitly in each panel legend.
COLORS={'Mg':'#247d66','Sm':'#865a9a','Li':'#5783c6','Pd':'#93605e','Cu':'#bc683d','Au':'#ae871c',
        'V':'#536499','Br':'#a34658','Ag':'#537b91','Bi':'#9c618a','S':'#40816c'}
RADIUS=.255


def formula(s):
    sub=str.maketrans('0123456789','₀₁₂₃₄₅₆₇₈₉')
    return s.translate(sub)


def read_rows(path):
    with path.open(encoding='utf8') as stream:return list(csv.DictReader(stream))


def palette(z):
    return COLORS.get(ELEMENTS[z],plt.get_cmap('tab20')(int(z)%20))


def neighbors(x,cell):
    """Find each central atom's full geometric shell out to 1.18 times its nearest distance."""
    sigma=np.linalg.svd(cell,compute_uv=False).min()
    n=len(x)
    radius=1
    while True:
        shifts=np.asarray(list(product(range(-radius,radius+1),repeat=3)))
        delta=x[None,:,None,:]+shifts[None,None,:,:]-x[:,None,None,:]
        d=np.linalg.norm(delta@cell,axis=-1)
        same=(np.arange(n)[:,None]==np.arange(n)[None,:])[:,:,None] & (shifts==0).all(1)
        d[same]=np.inf
        nearest=d.min(axis=(1,2))
        # Wrapped coordinates differ by <1; every unseen image has a Cartesian norm > sigma*radius.
        if 1.18*nearest.max()<sigma*radius:break
        radius+=1
        if radius>20:raise ValueError('Severely ill-conditioned cell in figure')
    return [(i,j,shifts[s]) for i,j,s in np.argwhere(d<=1.18*nearest[:,None,None])]


def clip_segment(a,b):
    start,end=0.,1.
    v=b-a
    for i in range(3):
        if abs(v[i])<1e-12:continue
        p,q=sorted(((0.-a[i])/v[i],(1.-a[i])/v[i]))
        start,end=max(start,p),min(end,q)
    if start>end:return None
    return np.stack([a+start*v,a+end*v])


def prepare(k,f,z):
    active=z>0
    z=z[active]
    cell=np.linalg.cholesky(unpack_gram(k))
    # All cells have their original first atom at the same fractional position in the drawing.
    x=(f[active]-f[active][0]+np.array([.21,.20,.16]))%1
    proj=cell@PROJECTION
    corners=(CORNERS-.5)@proj
    objects=[]
    for a in range(8):
        for b in range(a+1,8):
            if np.abs(CORNERS[a]-CORNERS[b]).sum()==1:
                pos=corners[[a,b]]
                objects.append((float(pos[:,2].mean()),'cell',pos,None))
    seen=set()
    for i,j,shift in neighbors(x,cell):
        a,b=x[i],x[j]+shift
        visible=clip_segment(a,b)
        if visible is None:continue
        key=tuple(sorted(tuple(np.round(v,7)) for v in visible))
        if key in seen:continue
        seen.add(key);pos=(visible-.5)@proj
        # Stop the projected connector at the displayed sphere rim so multiple
        # periodic contacts cannot obscure the colour of a central site.
        edge=pos[1]-pos[0]
        length=np.linalg.norm(edge[:2])
        trim=np.asarray([0.94*RADIUS if np.allclose(v,atom) else 0. for v,atom in zip(visible,(a,b))])
        if length<=trim.sum():continue
        pos[0]+=trim[0]*edge/length
        pos[1]-=trim[1]*edge/length
        objects.append((float(pos[:,2].mean()),'contact',pos,None))
    for s in product((-1,0,1),repeat=3):
        for v,elem in zip(x,z):
            q=v+s
            core=not any(s)
            if not core and ((q<-.16).any() or (q>1.16).any()):continue
            pos=(q-.5)@proj
            objects.append((float(pos[2]),'atom' if core else 'image',pos,int(elem)))
    points=[corners[:,:2]]
    points.extend([v[2][None,:2] for v in objects if v[1] in ('atom','image')])
    return sorted(objects,key=lambda item:item[0]),np.concatenate(points)


def draw_cell(ax,prepared,extent):
    ax.set_xlim(-extent,extent);ax.set_ylim(-extent,extent);ax.set_aspect('equal');ax.set_axis_off()
    if prepared is None:
        ax.text(.5,.5,'Relaxation failed',transform=ax.transAxes,ha='center',color='#963d37',fontsize=8)
        return
    for order,(_,kind,xy,z) in enumerate(prepared[0]):
        layer=order+1
        if kind in ('cell','contact'):
            color,lw=('#9aa4aa',.65) if kind=='cell' else ('#51616a',.95)
            ax.plot(xy[:,0],xy[:,1],color=color,lw=lw,zorder=layer,solid_capstyle='round')
        else:
            core=kind=='atom';c=palette(z)
            ax.add_patch(Circle(xy[:2],RADIUS if core else RADIUS*.84,facecolor=c if core else 'white',
                               edgecolor='#29323a' if core else c,linewidth=.55,zorder=layer,alpha=1 if core else .7))
            if core:
                ax.add_patch(Circle(xy[:2]+[-.065,.070],RADIUS*.28,color='white',alpha=.31,lw=0,zorder=layer+.1))


def structural_plate(raw,manifest,relaxed,records,unseen=False):
    indices=[];seen=set()
    for i,mid in enumerate(raw['ids']):
        row=manifest[str(mid)]
        signature=row['formula']
        if unseen:
            if row['predefined_relaxation']!='True':raise ValueError(f'{mid} is outside the selected prefix')
            if row['reduced_formula_in_training'] not in ('True','False'):raise ValueError('Missing training membership')
            if row['reduced_formula_in_training']=='True':continue
            z=raw['z'][i];elements,counts=np.unique(z[z>0],return_counts=True)
            signature=tuple(zip(elements,counts//np.gcd.reduce(counts)))
        if signature in seen:continue
        indices.append(i);seen.add(signature)
        if len(indices)==4:break
    if len(indices)!=4:raise ValueError('Four eligible distinct formulas required')
    datasets=[raw] if relaxed is None else [raw,relaxed]
    if relaxed is not None:
        np.testing.assert_array_equal(raw['ids'],relaxed['ids']);np.testing.assert_array_equal(raw['z'],relaxed['z'])
    views=[[None if j>0 and not d['successful'][i] else prepare(d['k'][i],d['f'][i],d['z'][i])
            for i in indices] for j,d in enumerate(datasets)]
    extent=max(abs(q[1]).max() for row in views for q in row if q is not None)+RADIUS+.18
    num=len(datasets)
    fig,axes=plt.subplots(num,4,figsize=(8.25,2.56 if num==1 else 4.15),squeeze=False)
    fig.subplots_adjust(left=.066,right=.988,bottom=.19 if num==1 else .14,top=.82 if num==1 else .87,wspace=.045,
                        hspace=.14)
    for j,row in enumerate(views):
        for col,(ax,q,i) in enumerate(zip(axes[j],row,indices)):
            draw_cell(ax,q,extent)
            mid=str(raw['ids'][i]);row_info=manifest[mid]
            if j==0:
                ax.set_title(formula(row_info['formula']),fontsize=13,fontweight='semibold',pad=13)
                # Both species appear in the same position for the original and relaxed cells.
                elems=np.unique(raw['z'][i][raw['z'][i]>0])
                for k,elem in enumerate(elems):
                    xx=.34+.37*k
                    ax.scatter(xx-.055,1.01,s=26,transform=ax.transAxes,clip_on=False,color=palette(elem),
                               edgecolor='#36434b',linewidth=.38)
                    ax.text(xx,1.01,ELEMENTS[elem],transform=ax.transAxes,va='center',ha='left',fontsize=7.8)
            if j==num-1:
                status=records.get(mid,{}).get('status','')
                status_label={'step_cap':'step limit','converged':'converged','error':'error'}.get(status,'')
                ax.text(.5,-.032,f'{mid}'+('   ·   '+status_label if status_label else ''),
                        transform=ax.transAxes,ha='center',va='top',fontsize=7.3,color='#425058')
        axes[j,0].text(-.092,.50,'Generated' if j==0 else 'CHGNet-relaxed',transform=axes[j,0].transAxes,
                        rotation=90,ha='center',va='center',fontsize=9.5,fontweight='semibold')
    # Physical length in angstroms converted through the exact data-to-figure transform of a cell.
    fig.canvas.draw()
    transform=fig.transFigure.inverted()
    point=axes[0,0].transData.transform([[0.,0.],[2.,0.]])
    length=(transform.transform(point)[1]-transform.transform(point)[0])[0]
    x0=.084;y=.054
    fig.add_artist(Line2D([x0,x0+length],[y,y],transform=fig.transFigure,lw=2.,color='#28343b'))
    fig.text(x0+length+.01,y,'2 Å',va='center',fontsize=8)
    key=[Line2D([],[],color='#51616a',lw=1.4,label='Geometric neighbors'),
         Line2D([],[],marker='o',mfc='white',mec='#566772',ms=5,ls='',label='Periodic image'),
         Line2D([],[],color='#9aa4aa',lw=.8,label='Unit cell')]
    fig.legend(handles=key,loc='center right',bbox_to_anchor=(.977,y),ncol=3,
               handlelength=1.55,columnspacing=1.2,frameon=False,fontsize=7.5)
    name='unseen-examples' if unseen else ('crystal-examples' if num==2 else 'generated-examples')
    for ext in ('pdf','png'):fig.savefig(OUT/f'{name}.{ext}',dpi=215,facecolor='white')
    plt.close(fig)
    return [str(raw['ids'][i]) for i in indices],extent


def all_outcomes(ids,records):
    rows=[records[str(mid)] for mid in ids]
    states=Counter(r['status'] for r in rows)
    fig,axes=plt.subplots(2,2,figsize=(8.25,5.10))
    fig.subplots_adjust(left=.145,right=.976,top=.925,bottom=.11,wspace=.46,hspace=.69)
    ax=axes[0,0]
    statuses=['converged','step_cap','error'];labels=['Converged','Step limit','Error']
    for j,(s,l) in enumerate(zip(statuses,labels)):
        count=states[s]
        ax.barh(j,count,color=['#26736e','#b18447','#a05050'][j],height=.55)
        ax.text(count+.5,j,str(count),va='center',fontsize=8.5)
    ax.set(ylabel='',yticks=range(3),yticklabels=labels,xlim=(0,len(rows)+2),xlabel='Candidates');ax.invert_yaxis()
    ax.set_title('Relaxation outcome',loc='left',fontsize=10,fontweight='semibold')
    ax=axes[0,1]
    for j,(p,color,label) in enumerate([('initial','#8a6498','Generated'),('final','#147477','After relaxation')]):
        v=np.array(sorted(float(r[p+'_max_force_eV_per_A']) for r in rows if r.get(p+'_max_force_eV_per_A')))
        if len(v):
            x=np.arange(len(v))+1
            ax.scatter(x,v,s=15,facecolor=color,linewidth=0,zorder=3)
            ax.plot(x,v,color=color,lw=.8,label=f'{label}  (n={len(v)})')
    ax.set_yscale('log');ax.set_xlabel('Rank within state');ax.set_ylabel('Maximum force (eV/Å)')
    ax.set_title('Forces across all candidates',loc='left',fontsize=10,fontweight='semibold')
    ax.legend(frameon=False,fontsize=7.3,loc='upper left')
    ax.axhline(.1,color='#7f8586',ls=':',lw=.8,zorder=-1)
    ax=axes[1,0]
    for p,color,label in [('initial','#8a6498','Generated'),('final','#147477','After relaxation')]:
        v=np.array(sorted(float(r[p+'_min_covalent_radius_ratio']) for r in rows if r.get(p+'_min_covalent_radius_ratio')))
        rank=np.arange(1,len(v)+1)
        ax.scatter(rank,v,s=12,color=color,lw=0,zorder=3)
        ax.plot(rank,v,lw=.85,color=color,label=label)
    ax.axhline(.6,color='#7f8586',ls=':',lw=.8)
    ax.set(xlabel='Rank within state',ylabel='Distance / summed radii',ylim=(0,None))
    ax.set_title('Shortest normalized contact',loc='left',fontsize=10,fontweight='semibold')
    ax=axes[1,1]
    known=[r for r in rows if r.get('delta_vs_reference_final_eV_per_atom')]
    known.sort(key=lambda r:float(r['delta_vs_reference_final_eV_per_atom']))
    for status,color,marker,label in [('converged','#26736e','o','Converged'),('step_cap','#b18447','^','Step limit')]:
        rr=[(j+1,float(r['delta_vs_reference_final_eV_per_atom'])) for j,r in enumerate(known) if r['status']==status]
        if rr:ax.scatter(*zip(*rr),s=25,color=color,marker=marker,label=label,linewidth=0,zorder=3)
    ax.axhline(0,color='#7f8586',ls=':',lw=.8)
    ax.set_yscale('symlog',linthresh=.1)
    ax.set_xlabel(f'Rank among {len(known)} reference pairs')
    ax.set_ylabel('Energy gap (eV/atom)')
    ax.set_title('Same-formula reference gap',loc='left',fontsize=10,fontweight='semibold')
    ax.legend(frameon=False,fontsize=7.1,loc='upper left')
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=8)
    for ext in ('pdf','png'):fig.savefig(OUT/f'relaxation-outcomes.{ext}',dpi=215,facecolor='white')
    plt.close(fig)
    return dict(states)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unseen',action='store_true',help='Render only the unseen-reduced-formula appendix montage.')
    args=parser.parse_args()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.6,'pdf.fonttype':42})
    OUT.mkdir(exist_ok=True)
    raw=np.load(DE_NOVO/'candidates.npz')
    manifest={r['id']:r for r in read_rows(DE_NOVO/'manifest.csv')}
    directory=DE_NOVO/'relaxation'
    has_final=(directory/'relaxed-candidates.npz').exists() and (directory/'results.csv').exists()
    relaxed=np.load(directory/'relaxed-candidates.npz') if has_final else None
    records={r['sample_id']:r for r in read_rows(directory/'results.csv')} if has_final else {}
    if args.unseen:
        if not has_final:raise FileNotFoundError('Relaxation results required')
        ids,extent=structural_plate(raw,manifest,relaxed,records,unseen=True)
        gap={r['id']:r for r in read_rows(DE_NOVO/'elemental-screen'/'candidate-gaps.csv')}
        detail=[]
        for mid in ids:
            r=records[mid]
            detail.append(dict(id=mid,formula=manifest[mid]['formula'],reduced_formula_in_training=False,
                status=r['status'],minimum_final_contact_A=float(r['final_min_contact_A']),
                maximum_final_atomic_force_eV_per_A=float(r['final_max_force_eV_per_A']),
                final_energy_eV_per_atom=float(r['final_energy_eV_per_atom']),
                delta_vs_elements_eV_per_atom=float(gap[mid]['delta_vs_elements_eV_per_atom']),
                same_formula_reference=r['reference_id'] or None,
                delta_vs_reference_eV_per_atom=float(r['delta_vs_reference_final_eV_per_atom'])
                    if r['delta_vs_reference_final_eV_per_atom'] else None))
        write_json(OUT/'unseen-examples-metadata.json',dict(sample_sha256=digest(DE_NOVO/'candidates.npz'),
            manifest_sha256=digest(DE_NOVO/'manifest.csv'),relaxed_sha256=digest(directory/'relaxed-candidates.npz'),
            source_ids=len(raw['ids']),selection_rule='First four distinct reduced formulas among the fixed 32 with manifest reduced_formula_in_training=False; no use of screening outcomes.',
            selection=ids,examples=detail,view_half_width_angstrom=extent))
        print('Rendered the unseen-formula subset',ids)
        return
    if has_final:structural_plate(raw,manifest,None,{})
    ids,extent=structural_plate(raw,manifest,relaxed,records)
    outcomes=all_outcomes(raw['ids'],records) if has_final else None
    write_json(OUT/'render-metadata.json',dict(sample_sha256=digest(DE_NOVO/'candidates.npz'),selection=ids,
        selection_rule='First four distinct formulas in source candidate order. No geometric prefilter.',
        projection='Common orthographic camera, Cholesky cell, first atom at fractional (.21,.20,.16).',
        geometry='Every panel shares the same Å per figure unit. Atom diameter 0.51 Å for display; open circles represent nearby periodic images.',
        neighbors='Edges connect a site to all sites within 1.18 times its nearest periodic distance, clipped at cell boundaries. They describe geometry only.',
        view_half_width_angstrom=extent, n_candidates=len(raw['ids']), relaxation_outcomes=outcomes,
        relaxed_sha256=digest(directory/'relaxed-candidates.npz') if has_final else None))
    print('Rendered',ids,'relaxation outcomes',outcomes)


if __name__=='__main__':
    main()

