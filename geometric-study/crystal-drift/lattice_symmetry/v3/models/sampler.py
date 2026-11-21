"""Train-only formula-to-symbolic seed prior and one-call native-cell inference."""
import gzip
import json
import os
import string
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
V3 = HERE.parent
V2 = V3.parent / 'v2'
ROOT = V3.parents[1]
MENDELEEV = np.array(json.loads((HERE/'mendeleev.json').read_text()))
with np.load(V2/'wyckoff/charts.npz') as db:
    M, T = db['matrices'].astype(float), db['t24'].astype(float)/24
    START, GROUP = db['positions'], db['groups']


def weight_path(path):
    """Resolve heavy local training payloads retained on WSL scratch; archives may restore normal files."""
    path=Path(path)
    if path.exists():return path
    pointer=path.with_suffix(path.suffix+'.pointer.json')
    if pointer.exists():
        record=json.loads(pointer.read_text())
        physical=record.get('windows_path') if os.name=='nt' else record['path']
        if physical and Path(physical).exists():return Path(physical)
    return path


@lru_cache(None)
def chart(g, letter):
    k = GROUP[g-1] + string.ascii_letters.index(letter)
    return M[START[k]:START[k+1]], T[START[k]:START[k+1]]


@lru_cache(None)
def group_ops(g):
    k = GROUP[g]-1
    r,t = M[START[k]:START[k+1]], T[START[k]:START[k+1]]
    tr=t[(r==np.eye(3)).all((1,2))]
    bases = [np.eye(3),np.array([[-1,1,1],[1,-1,1],[1,1,-1]])/2,
             np.array([[1,1,0],[-1,1,0],[0,0,2]])/2,np.array([[2,0,0],[0,1,1],[0,-1,1]])/2,
             np.array([[1,0,1],[0,2,0],[-1,0,1]])/2,np.array([[0,1,1],[1,0,1],[1,1,0]])/2,
             np.array([[2,1,1],[-1,1,1],[-1,-2,1]])/3]
    for a in bases:
        inv = np.linalg.inv(a)
        if np.isclose(abs(np.linalg.det(a))*len(tr),1) and np.allclose(tr@inv,np.rint(tr@inv)):
            return r,a,inv
    raise ValueError(('unrecognized centering',g,tr))


def species_key(z):
    a,c=np.unique(z[z>0],return_counts=True)
    ix=np.lexsort((a,MENDELEEV[a],c))
    return tuple(c[ix]),a[ix],tuple(zip(a.tolist(),(c//np.gcd.reduce(c)).tolist()))


@lru_cache(None)
def orbit_chart(g,l):
    m,t=chart(g,l)
    _,a,inv=group_ops(g)
    # A fixed generic representative detects only exact centering copies.
    j=m@inv
    b=t@inv
    f=(np.einsum('i,sij->sj',np.array([.13711912,.28193217,.43688923]),j)+b)%1
    kept=[]
    for i in range(len(f)):
        delta=f[i]-f[kept]
        if not np.any((np.abs(delta-np.rint(delta))<1e-8).all(axis=1)):
            kept.append(i)
    if len(kept)!=round(len(t)*abs(np.linalg.det(a))):
        raise ValueError(('chart reduction',g,l))
    return j[kept], b[kept]


@dataclass
class Seed:
    L:np.ndarray
    F:np.ndarray
    Z:np.ndarray
    orbit:np.ndarray
    J:np.ndarray
    group:int
    template:tuple
    donor:str
    cell_donor:str


def compose(g,pattern,cell,parameters,donor='',cell_donor=''):
    orbit=[];J=[];f=[];z=[]
    for i,((element,l),u) in enumerate(zip(pattern,parameters)):
        j,t=orbit_chart(g,l)
        f.extend((np.einsum('i,sij->sj',u,j)+t)%1.)
        J.extend(j);orbit.extend([i]*len(j));z.extend([element]*len(j))
    return Seed(group_ops(g)[1]@cell,np.asarray(f),np.asarray(z),np.asarray(orbit),
                np.asarray(J),g,pattern,donor,cell_donor)


class Sampler:
    def __init__(self,device='cpu'):
        self.device=torch.device(device)
        # Keep cache ownership on this fitted index. A class-level lru_cache retains
        # every prior through its self keys across the multi-fit CUDA supervisor.
        self.shortlist=lru_cache(maxsize=65536)(self._lookup_shortlist)
        kept=set(json.loads((V3/'protocol/splits.json').read_text())['train_ids'])
        rows=[]
        for p in sorted((V2/'wyckoff/labels').glob('train-*.jsonl.gz')):
            with gzip.open(p,'rt') as file:
                rows.extend(r for line in file if (r:=json.loads(line))['id'] in kept and 'error' not in r and r['native_n']==r['primitive_n'])
        with np.load(ROOT/'scaled/mp20-data/train.npz') as data:
            self.allz=data['z']
        self.rows=rows;self.by_sig=defaultdict(list);self.by_pattern=defaultdict(list)
        self.by_group=defaultdict(list);self.by_formula=defaultdict(list);self.rich=[]
        self.red=[];self.cells=np.stack([r['cell'] for r in rows]);x=[];y=[]
        self.ids=[r['id'] for r in rows];self.mid={s:i for i,s in enumerate(self.ids)}
        for i,r in enumerate(rows):
            z=self.allz[r['index']];sig,elements,reduced=species_key(z)
            p=tuple((int(e),l) for e,l in r['template']);g=r['group']
            self.by_sig[sig].append(i);self.by_group[g].append(i)
            self.by_pattern[g,tuple(sorted(l for _,l in p))].append(i)
            self.by_formula[tuple(sorted(z[z>0]))].append(i)
            self.rich.append((g,p,elements));self.red.append(reduced)
            x.append(self.design(z));y.append(np.log(abs(np.linalg.det(self.cells[i]))/r['std_n']))
        X=np.array(x);y=np.array(y);I=np.eye(X.shape[1])*.2;I[0,0]=0
        self.beta=np.linalg.solve(X.T@X+I,X.T@y)
        self.residual=y-X@self.beta
        self.all=np.arange(len(rows))
        self.ranks={}
        for sig,idx in self.by_sig.items():
            self.by_sig[sig]=np.array(idx)
            self.ranks[sig]=np.stack([self.rich[j][-1] for j in idx])

    @staticmethod
    def design(z):
        c=np.bincount(z[z>0],minlength=119)[1:]
        return np.r_[1.,np.log(c.sum()),c/c.sum()]

    def _lookup_shortlist(self, formula, exclude, k=20):
        z=np.array(formula);sig,elements,reduced=species_key(z)
        out=[];seen=set()
        if sig in self.by_sig:
            e=self.ranks[sig]
            d=(abs(MENDELEEV[e]-MENDELEEV[elements])+2*(e!=elements)).mean(1)
            for rank in np.argsort(d,kind='stable'):
                ix=self.by_sig[sig][rank]
                if exclude and self.red[ix]==reduced:continue
                g,pat,orig=self.rich[ix]; remap=dict(zip(orig,elements))
                p=tuple(sorted((int(remap[e]),l) for e,l in pat))
                if (g,p) in seen:continue
                seen.add((g,p));out.append((g,p,self.ids[ix]))
                if len(out)>=k:break
        if not out:out=[(1,tuple((int(e),'a') for e in z),'')]
        return out

    def draw(self,z,option,rng,exclude=False):
        g,p,donor=option
        sig,e,reduced=species_key(z)
        for bank in (self.by_pattern.get((g,tuple(sorted(l for _,l in p))),[]),
                     self.by_group.get(g,[]),self.all):
            if len(bank)==0:continue
            # Rejection with systematic fallback handles all-excluded small banks.
            ix=None
            for trial in range(5):
                j=bank[rng.integers(len(bank))]
                if not exclude or self.red[j]!=reduced:ix=j;break
            if ix is None:
                allowed=[j for j in bank if self.red[j]!=reduced]
                if allowed:ix=allowed[rng.integers(len(allowed))]
            if ix is not None:break
        else:raise ValueError('No eligible cell donor')
        C=self.cells[ix];rot,A,_=group_ops(g)
        G=C@C.T
        H=(rot@G@rot.transpose(0,2,1)).mean(0)
        L=np.linalg.cholesky((H+H.T)/2)
        specific=np.exp(np.clip(self.design(z)@self.beta+self.residual[ix],np.log(3),np.log(160)))
        L*=np.cbrt(specific*len(z)/(abs(np.linalg.det(A))*abs(np.linalg.det(L))))
        u=rng.random((len(p),3))
        rec=compose(g,p,L,u,donor,self.ids[ix])
        if len(z)!=len(rec.Z) or not np.array_equal(np.sort(z),np.sort(rec.Z)):
            raise ValueError('Invalid multiplicity')
        return rec

    def seeds(self,z,draws=5,seed=1234):
        z=np.asarray(z);z=np.sort(z[z>0])
        opts=self.shortlist(tuple(z),False)
        rng=np.random.default_rng(seed)
        return [self.draw(z,opts[i%len(opts)],rng) for i in range(draws)]

    def target(self,i):
        r=self.rows[i];g,p,_=self.rich[i]
        return compose(g,p,self.cells[i],r['u'])

    def load(self,path):
        from .network import make_model
        obj=torch.load(weight_path(path),map_location='cpu',weights_only=True)
        net=make_model(obj['architecture'])
        net.load_state_dict(obj['net']);return net.to(self.device).eval()

    def apply(self,model,seeds):
        if model is None:return [(s.L.copy(),s.F.copy(),s.Z.copy()) for s in seeds]
        from .network import pack
        args=pack(seeds,self.device)
        with torch.inference_mode():
            L,F=model(*args)
        L=L.cpu().numpy();F=F.cpu().numpy()
        return [(l,f[:len(s.Z)],s.Z.copy()) for l,f,s in zip(L,F,seeds)]

