"""Build deterministic exact-formula training blocks on the protected train split only."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .sampler import HERE, V3, Sampler, group_ops
from .network import pack
from ..geometry.descriptors import feature_blocks,fit_stats,combine,inter_orbit_contacts,orbit_variance,reciprocal_shells
from lattice_symmetry.v2.neighbors import edges
from lattice_symmetry.v2.physical import features

CACHE=HERE/'cache'
BLOCK=128
torch.set_num_threads(1)


@torch.no_grad()
def raw(recs):
    out={}
    for first in range(0,len(recs),24):
        L,F,Z,O,*_=pack(recs[first:first+24])
        graph=edges(L,F,Z,4.)
        a=dict(base=features(L,F,Z),contacts=inter_orbit_contacts(L,F,Z,O,graph=graph),
               variance=orbit_variance(L,F,Z,O,graph=graph),reciprocal=reciprocal_shells(L,F,Z))
        for k,v in a.items():out.setdefault(k,[]).append(v.numpy())
    return {k:np.concatenate(x) for k,x in out.items()}


def safe_save(path,**kw):
    tmp=path.with_suffix('.building.npz')
    np.savez_compressed(tmp,**kw);os.replace(tmp,path)


def calibrate(s):
    rng=np.random.default_rng(2293);ix=rng.choice(len(s.rows),256,replace=False)
    positives=[s.target(i) for i in ix];rawp=raw(positives);stats=fit_stats(rawp)
    query=[]
    for i in ix:
        z=np.sort(s.allz[s.rows[i]['index']]);z=z[z>0]
        opts=s.shortlist(tuple(z),True)
        query.append(s.draw(z,opts[rng.integers(min(5,len(opts)))],rng,True))
    q=raw(query)['base']
    mu,sd=stats['base'];dist=np.linalg.norm((q-rawp['base'])/sd,axis=1)
    sigma=float(np.median(dist));temps=(max(.01,.3*sigma),max(.01,.8*sigma))
    safe_save(CACHE/'calibration.npz',ids=np.array([s.ids[i] for i in ix]),temperatures=temps,
              distances=dist,**{f'{k}_{n}':a for k,(m,v) in stats.items() for n,a in [('mean',m),('scale',v)]})
    return stats


def align_endpoint(seed,ref,p):
    # Align atom labels by a Hungarian assignment at a fixed conventional origin.
    # Project shortest componentwise torus chords to the seed's occupied-orbit tangent.
    F=seed.F;L=seed.L;Z=seed.Z
    delta=(ref.F[None,:,:]-F[:,None,:]+.5)%1-.5
    cost=np.linalg.norm(delta@L,axis=-1)
    cost[Z[:,None]!=ref.Z[None,:]]=1e9
    ix,iy=linear_sum_assignment(cost)
    v=delta[ix,iy]@L
    du=np.zeros((max(seed.orbit)+1,3))
    np.add.at(du,seed.orbit,np.einsum('si,sij->sj',v,p[:len(Z)]))
    df=np.einsum('si,sij->sj',du[seed.orbit],seed.J)
    # Compatible target metric projected in seed fractional setting.
    operations,A,inv=group_ops(seed.group)
    # Row basis A maps native cell = A*conventional: R_native=A R_conv A^-1.
    rot=A@operations@inv
    G=ref.L@ref.L.T
    H=(rot@G@rot.transpose(0,2,1)).mean(0)
    H*=np.cbrt((abs(np.linalg.det(ref.L))**2)/np.linalg.det(H))
    linv=np.linalg.inv(L);C=linv@H@linv.T
    ev,Q=np.linalg.eigh(C)
    B=(Q*np.sqrt(np.maximum(ev,1e-10)))@Q.T
    return L@(B-np.eye(3)),df


def make_shard(s,rows,k,epoch=0,in_memory=False):
    if epoch and not in_memory and (HERE/'pause-banks').exists():
        raise SystemExit('extra bank preparation paused at a shard boundary')
    directory=CACHE if epoch==0 else CACHE/f'epoch-{epoch:02}'
    if not in_memory:directory.mkdir(exist_ok=True)
    path=directory/f'train-{k:03}.npz'
    if path.exists() and not in_memory:return
    t=time.perf_counter();seeds=[];pos=[];pidx=[];option_ix=[];arr_i=[]
    for i in rows:
        entropy=[68171,int(i)]+([epoch] if epoch else [])
        rng=np.random.default_rng(np.random.SeedSequence(entropy))
        z=s.allz[s.rows[i]['index']];z=np.sort(z[z>0]);formula=tuple(z)
        idx=[i,int(rng.choice(s.by_formula[formula]))]
        pos.extend(s.target(x) for x in idx);pidx.extend(idx)
        opts=s.shortlist(formula,True)
        for j in range(4):
            pick=rng.integers(min(5,len(opts)))
            seeds.append(s.draw(z,opts[pick],rng,True));option_ix.append(pick)
        arr_i.append(i)
    args=[t.numpy() for t in pack(seeds)]
    rawpos=raw(pos)
    targetL=[];targetF=np.zeros_like(args[1])
    for i,seed in enumerate(seeds):
        dL,dF=align_endpoint(seed,pos[2*(i//4)+i%2],args[-1][i])
        targetL.append(dL);targetF[i,:len(dF)]=dF
    result=dict(indices=np.array(arr_i),pidx=np.array(pidx),option_ix=np.array(option_ix),
              donor=[s.mid.get(q.donor,-1) for q in seeds],cell_donor=[s.mid[q.cell_donor] for q in seeds],
              groups=[q.group for q in seeds],targetL=np.array(targetL,dtype=np.float32),targetF=targetF,
              **{f'x{j}':v.astype(np.int16) if j in (2,3,4,5,6) else v for j,v in enumerate(args)},
              **{f'p_{key}':x for key,x in rawpos.items()})
    if in_memory:return result
    safe_save(path,**result)
    print('shard',k,len(rows),round(time.perf_counter()-t,2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--part',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--limit',type=int);p.add_argument('--epoch',type=int,default=0);a=p.parse_args()
    CACHE.mkdir(exist_ok=True);s=Sampler()
    ix=np.random.default_rng(55171+a.epoch).permutation(len(s.rows))
    if a.part==0 and a.epoch==0:
        if not (CACHE/'calibration.npz').exists():calibrate(s)
        (CACHE/'inventory.json').write_text(json.dumps(dict(n=len(ix),block=BLOCK,
                ids_sha256=hashlib.sha256(('\n'.join(s.ids)+'\n').encode()).hexdigest(),
                split_sha256=hashlib.sha256((V3/'protocol/splits.json').read_bytes()).hexdigest(),
                eligible=s.ids,order=ix.tolist(),singleton=sum(len(v)==1 for v in s.by_formula.values()),
                singleton_rows=sum(len(v) for v in s.by_formula.values() if len(v)==1)),separators=(',',':')))
    for k,indices in enumerate([ix[j:j+BLOCK] for j in range(0,len(ix),BLOCK)]):
        if k%a.workers==a.part and (a.limit is None or k<a.limit):make_shard(s,indices,k,a.epoch)


if __name__=='__main__':main()

