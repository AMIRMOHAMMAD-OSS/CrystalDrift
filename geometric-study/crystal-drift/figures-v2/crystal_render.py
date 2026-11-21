"""Orthographic, depth-buffered rendering of the deposited periodic unit cells."""
import csv
import hashlib
import json
import sys
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
sys.path.insert(0,str(ROOT))
from data import ELEMENTS,unpack_gram
from plot_crystals import CORNERS,PROJECTION,formula,neighbors


DATA=ROOT/'de-novo'
COLOR={'Mg':'#11ad80','Sm':'#a74ce0','Li':'#168deb','Pd':'#c9718c','Cu':'#cf5b28','Au':'#efa900',
       'V':'#4975cd','Br':'#bb3348','Ag':'#7798b2','Bi':'#a83bad','S':'#79aa25'}
RADIUS=.50
PADDING=.23
PIXELS=700
LIGHT=np.array([-.46,.65,.62]); LIGHT/=np.linalg.norm(LIGHT)
HALF=LIGHT+np.array([0,0,1]);HALF/=np.linalg.norm(HALF)
INK='#192a36'
MUTED='#546572'


def read_rows(path):
    with path.open(encoding='utf8') as stream:return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene(k,frac,atomic_numbers):
    z=atomic_numbers[atomic_numbers>0]
    cell=np.linalg.cholesky(unpack_gram(k))
    np.testing.assert_allclose(cell@cell.T,unpack_gram(k),atol=1e-8)
    x=(frac[:len(z)]-frac[0]+[.21,.20,.16])%1.
    proj=cell@PROJECTION
    spheres={}; links=[]; rods=[]
    for shift in product((-1,0,1),repeat=3):
        for j,xyz in enumerate(x):
            point=xyz+shift
            if np.any(point < -PADDING) or np.any(point > 1+PADDING):continue
            key=j,shift
            spheres[key]=((point-.5)@proj,COLOR[ELEMENTS[z[j]]],any(shift))
    # Translate every true periodic shell found around the equivalent central atom.
    shell=neighbors(x,cell)
    keys=set()
    for (i,shift),(u,_,ghost_u) in spheres.items():
        for src,j,other in shell:
            if src!=i:continue
            other_key=j,tuple((np.asarray(shift)+other).tolist())
            if other_key not in spheres:continue
            pair=tuple(sorted(((i,shift),other_key)))
            if pair in keys:continue
            keys.add(pair)
            v,_,ghost_v=spheres[other_key]
            links.append((u,v,ghost_u and ghost_v))
    corners=(CORNERS-.5)@proj
    for i,u in enumerate(CORNERS):
        for j,v in enumerate(CORNERS[i+1:],i+1):
            if np.abs(v-u).sum()==1:rods.append((corners[i],corners[j]))
    xy=np.vstack([corners[:,:2],*[p[0][None,:2] for p in spheres.values()]])
    lower,upper=xy.min(0)-RADIUS,xy.max(0)+RADIUS
    return dict(atoms=list(spheres.values()),links=links,rods=rods,center=(lower+upper)/2,
                halfextent=(upper-lower)/2,central_sites=len(z))


class Renderer:
    """Analytic front-surface intersections of orthographic rays with spheres and cylinders."""
    def __init__(self,center,halfwidth,size=PIXELS):
        self.size=size
        self.halfwidth=halfwidth
        self.x=np.linspace(center[0]-halfwidth,center[0]+halfwidth,size)
        self.y=np.linspace(center[1]+halfwidth,center[1]-halfwidth,size)
        self.depth=np.full((size,size),-np.inf,dtype=np.float32)
        self.rgb=np.ones((size,size,3),dtype=np.float32)

    def window(self,lo,hi):
        ix=np.flatnonzero((self.x>=lo[0])&(self.x<=hi[0]))
        iy=np.flatnonzero((self.y>=lo[1])&(self.y<=hi[1]))
        if not len(ix) or not len(iy):return None
        sub=np.s_[iy[0]:iy[-1]+1,ix[0]:ix[-1]+1]
        return sub,self.x[ix][None,:],self.y[iy][:,None]

    def paint(self,sub,z,nx,ny,nz,visible,color,ghost=False,metal=False):
        valid=visible & (z>self.depth[sub])
        if not valid.any():return
        nx,ny,nz=np.broadcast_arrays(nx,ny,nz)
        n=np.stack([nx[valid],ny[valid],nz[valid]],axis=1)
        # A cool ambient term and two broad specular components give a visible surface normal.
        intensity=.42+.52*np.maximum(0,n@LIGHT)+.08*np.maximum(0,n@np.array([.8,-.1,.5]))
        spec=np.maximum(0,n@HALF)
        c=np.asarray(to_rgb(color))
        surface=c[None,:]*intensity[:,None]+(.22*spec**14+.48*spec**85)[:,None]
        if metal:surface=.07+surface*.92
        rim=.05*(1-n[:,2]).clip(0,1)**2
        surface=np.clip(surface+rim[:,None],0,1)
        if ghost:
            surface=.41*surface+.59  # High-key surface encodes periodic copies.
        self.rgb[sub][valid]=surface
        self.depth[sub][valid]=z[valid]

    def sphere(self,p,r,color,ghost=False,metal=False):
        win=self.window(p[:2]-r,p[:2]+r)
        if win is None:return
        sub,X,Y=win
        dx,dy=(X-p[0])/r,(Y-p[1])/r
        t=1-dx*dx-dy*dy
        zz=np.sqrt(np.maximum(0,t))
        self.paint(sub,p[2]+r*zz,dx,dy,zz,t>=0,color,ghost,metal)

    def cylinder(self,a,b,r,color,ghost=False):
        if np.linalg.norm(b-a)<1e-9:return
        win=self.window(np.minimum(a[:2],b[:2])-r,np.maximum(a[:2],b[:2])+r)
        if win is None:return
        sub,X,Y=win;dx,dy=X-a[0],Y-a[1]
        ell=np.linalg.norm(b-a);v=(b-a)/ell
        A=max(1-v[2]**2,1e-12)
        t0=dx*v[0]+dy*v[1]
        disc=(v[2]*t0)**2-A*(dx*dx+dy*dy-t0*t0-r*r)
        dz=(v[2]*t0+np.sqrt(np.maximum(0,disc)))/A
        t=t0+dz*v[2]
        self.paint(sub,a[2]+dz,(dx-t*v[0])/r,(dy-t*v[1])/r,(dz-t*v[2])/r,
                   (disc>=0)&(t>=0)&(t<=ell),color,ghost,True)


def render_scene(s,halfwidth):
    renderer=Renderer(s['center'],halfwidth)
    for a,b in s['rods']:
        renderer.cylinder(a,b,.030,'#829bad')
    for a,b,ghost in s['links']:
        renderer.cylinder(a,b,.069,'#869aa9',ghost)
    for p,color,ghost in s['atoms']:
        renderer.sphere(p,RADIUS,color,ghost)
    return np.round(renderer.rgb*255).astype(np.uint8)


def choose_ids(raw,manifest,unseen):
    seen=set();chosen=[]
    for i,mid in enumerate(raw['ids']):
        r=manifest[str(mid)];assert r['predefined_relaxation']=='True'
        if unseen and r['reduced_formula_in_training']!='False':continue
        z=raw['z'][i];el,n=np.unique(z[z>0],return_counts=True)
        key=tuple(zip(el,n//np.gcd.reduce(n))) if unseen else r['formula']
        if key in seen:continue
        seen.add(key);chosen.append(i)
        if len(chosen)==4:break
    return chosen


def plate(name,indices,raw,relaxed,records,manifest,scenes,halfw):
    fig=plt.figure(figsize=(8.45,4.80),facecolor='white')
    left=.073;right=.983;step=(right-left)/4;tile_w=step-.008
    tile_h=tile_w*fig.get_figwidth()/fig.get_figheight()
    anchors=(.496,.101)
    for j,i in enumerate(indices):
        mid=str(raw['ids'][i]);row=records[mid];x=left+j*step
        fig.text(x+tile_w/2,.967,formula(manifest[mid]['formula']),ha='center',va='center',fontsize=13.6,
                 color=INK,fontweight='bold')
        el=np.unique(raw['z'][i][raw['z'][i]>0])
        for n,elem in enumerate(el):
            sx=x+tile_w*(.30+n*.41)
            fig.add_artist(Line2D([sx],[.928],marker='o',ms=5.0,mec='none',mfc=COLOR[ELEMENTS[elem]],
                                  transform=fig.transFigure))
            fig.text(sx+.010,.928,ELEMENTS[elem],va='center',fontsize=8.1,color=MUTED)
        for q,bottom in enumerate(anchors):
            ax=fig.add_axes([x,bottom,tile_w,tile_h])
            world=scenes[i,q]
            if world is None:
                ax.text(.5,.5,'Relaxation failed',ha='center',color='#ad3744',transform=ax.transAxes)
            else:
                ax.imshow(render_scene(world,halfw),interpolation='antialiased',extent=(-halfw,halfw,-halfw,halfw))
                if mid=="draw-0002" and q==1:
                    # Relaxed CIF sites 3 and 4 are the two Pd atoms in the closest pair.
                    cell=np.linalg.cholesky(unpack_gram(relaxed['k'][i]))
                    frac=(relaxed['f'][i]-relaxed['f'][i,0]+[.21,.20,.16])%1.
                    offset=frac[3]-frac[2]
                    offset-=np.rint(offset)
                    assert abs(np.linalg.norm(offset@cell)-.31340144)<.0001
                    midpoint=(frac[2]+.5*offset-.5)@cell@PROJECTION
                    xy=midpoint[:2]-world['center']
                    ax.annotate('Pd–Pd',xy=xy,xytext=(-halfw*.90,halfw*.75),
                                fontsize=7.2,color='#943748',ha='left',
                                arrowprops=dict(arrowstyle='->',color='#943748',lw=.8))
            ax.axis('off')
        status='step limit' if row['status']=='step_cap' else row['status']
        hue='#a63d39' if status!='converged' else MUTED
        fig.text(x+tile_w/2,.100,mid+'  ·  '+status,ha='center',va='center',fontsize=7.5,color=hue,
                 fontweight='semibold' if status!='converged' else 'normal')
        fig.text(x+tile_w/2,.071,f"Minimum contact  {float(row['final_min_contact_A']):.2f} Å",
                 ha='center',va='center',fontsize=7.2,color=MUTED)
    for j,(y,label) in enumerate(zip(anchors,('Generated','CHGNet-relaxed'))):
        fig.text(.036,y+tile_h*.5,label,rotation=90,ha='center',va='center',fontsize=9.6,fontweight='semibold',color=INK)
    fy=.028
    length=tile_w*2./(2*halfw)
    fig.add_artist(Line2D([left,left+length],[fy,fy],lw=1.8,transform=fig.transFigure,color=INK))
    fig.text(left+length+.008,fy,'2 Å',va='center',fontsize=7.7,color=INK)
    fig.text(.972,fy,'Saturated cell sites     ·     Pale periodic copies     ·     Sticks connect geometric neighbors',
             ha='right',va='center',fontsize=7.25,color=MUTED)
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'{name}.{ext}',facecolor='white',dpi=360)
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'font.size':8.5})
    raw=np.load(DATA/'candidates.npz');relaxed=np.load(DATA/'relaxation'/'relaxed-candidates.npz')
    manifest={r['id']:r for r in read_rows(DATA/'manifest.csv')}
    records={r['sample_id']:r for r in read_rows(DATA/'relaxation'/'results.csv')}
    np.testing.assert_array_equal(raw['ids'],relaxed['ids']);np.testing.assert_array_equal(raw['z'],relaxed['z'])
    selected={'crystal-examples':choose_ids(raw,manifest,False),'unseen-examples':choose_ids(raw,manifest,True)}
    for name,indices in selected.items():
        legacy='render-metadata.json' if name=='crystal-examples' else 'unseen-examples-metadata.json'
        original=json.loads((DATA/'figures'/legacy).read_text())['selection']
        np.testing.assert_array_equal(raw['ids'][indices],original)
    scenes={}
    for i in sorted(set(sum(selected.values(),[]))):
        for j,d in enumerate((raw,relaxed)):
            scenes[i,j]=scene(d['k'][i],d['f'][i],d['z'][i]) if j==0 or d['successful'][i] else None
    halfwidth=max(np.max(s['halfextent']) for s in scenes.values() if s is not None)+.09
    for name,indices in selected.items():plate(name,indices,raw,relaxed,records,manifest,scenes,halfwidth)
    provenance=dict(selections={n:raw['ids'][ix].tolist() for n,ix in selected.items()},
        source_file_hashes={f:digest(DATA/f) for f in ('candidates.npz','relaxation/relaxed-candidates.npz','manifest.csv','relaxation/results.csv')},
        orthographic_basis=PROJECTION.tolist(),viewport_width_A=2*halfwidth,illustrative_sphere_diameter_A=2*RADIUS,
        cell='Cholesky of the stored Gram. Common fractional translation places the first atom at (0.21,0.20,0.16), including after relaxation. Only frame centering may differ.',
        copies='Real neighboring translations whose three fractional coordinates lie in [-0.23,1.23]. Pale spheres distinguish the neighboring copies.',
        connectors='Undirected edges join displayed atoms within 1.18 times the certified nearest periodic distance of either endpoint. No chemical bond typing.',
        numerical_render='Orthographic per-pixel sphere/cylinder intersections with front-surface depth buffering. Unit cell edges and contacts are cylinders. No atom coordinates are fitted or drawn by hand.',
        species_colors=COLOR,panels={str(raw['ids'][i]):[dict(cell_sites=scenes[i,j]['central_sites'],displayed_periodic_copies=len(scenes[i,j]['atoms'])-scenes[i,j]['central_sites'],
            geometric_connectors=len(scenes[i,j]['links'])) if scenes[i,j] else None for j in (0,1)] for i in sorted(set(sum(selected.values(),[])))})
    (OUT/'crystal-render-metadata.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf8')
    print('Rendered deposit selections',provenance['selections'],'common width',2*halfwidth)


if __name__=='__main__':
    main()

