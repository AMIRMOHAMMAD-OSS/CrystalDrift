"""Independent geometric scores and training-geometry retrieval on held-out crystals."""
import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiment import HERE,load,save_predictions,select,sha


def gram(k):
    s=np.diag(k[:3].astype(float)); i,j=np.triu_indices(3,1)
    s[i,j]=s[j,i]=k[3:]/math.sqrt(2)
    w,q=np.linalg.eigh(s)
    return (q*np.exp(2*w))@q.T


def physical(k,f,z):
    """Log volume/site and min, median, 90th percentile site-neighbor distance.

    Periodic images are certified using the reciprocal-cell norm bound. The
    diagonal includes self images and excludes only the zero translation.
    """
    if np.max(np.abs(k))>12:raise ValueError('Numerically exceptional metric')
    g=gram(k); n=int(np.count_nonzero(z)); x=f[:n].astype(float)
    d=x[None,:,:]-x[:,None,:];d-=np.floor(d+.5)
    dg=d@g; dd=(d*dg).sum(-1); rd=np.diag(np.linalg.inv(g)).max()
    for radius in (1,2,3,4,6,8,12):
        nearest=np.full(n,np.inf)
        offsets=np.array(list(itertools.product(range(-radius,radius+1),repeat=3)))
        for block in np.array_split(offsets,max(1,math.ceil(len(offsets)/128))):
            d2=dd[:,:,None]+2*(dg@block.T)+((block@g)*block).sum(1)
            origin=np.where(~block.any(1))[0]
            if len(origin):d2[np.arange(n),np.arange(n),origin[0]]=np.inf
            nearest=np.minimum(nearest,d2.min(axis=(1,2)))
        if nearest.max()*rd < (radius+.5)**2*(1-1e-12):
            dist=np.sqrt(np.maximum(nearest,0))
            return np.r_[k[:3].sum()-math.log(n),np.min(dist),np.median(dist),np.quantile(dist,.9)]
    raise ValueError('Periodic-image search could not certify the distances')


def process(k,f,z):
    out=np.full((len(z),4),np.nan)
    for i in range(len(z)):
        try:out[i]=physical(k[i],f[i],z[i])
        except (ValueError,np.linalg.LinAlgError,OverflowError):pass
    return out


class Retrieval:
    def __init__(self,train):
        self.train=train; self.members=defaultdict(list); self.by_count=defaultdict(list); self.by_pattern=defaultdict(list)
        for i,z in enumerate(train['z']):
            key=tuple(z[z>0]);self.members[key].append(i)
            self.by_count[len(key)].append(i);self.by_pattern[self.pattern(key)].append(i)
        self.cache={}

    @staticmethod
    def pattern(z):return tuple(len(list(g)) for _,g in itertools.groupby(z))

    def pool(self,z):
        key=tuple(z[z>0])
        if key not in self.cache:
            if key in self.members:group=self.members[key]
            else:
                group=self.by_pattern.get(self.pattern(key),self.by_count[len(key)])
                if not len(group):raise ValueError('No training structure with this atom count')
                group=np.asarray(group);d=np.abs(self.train['z'][group]-z).sum(1)
                group=group[d==d.min()]
            self.cache[key]=np.asarray(group)
        return self.cache[key]

    def save(self,part,seed,draws,ntarget,path):
        idx=select(part['ids'],ntarget);z=part['z'][idx];rng=np.random.default_rng(100000+seed)
        donor=np.array([rng.choice(self.pool(row),size=draws) for row in z])
        save_predictions(path,ids=part['ids'][idx],z=z,k=self.train['k'][donor],f=self.train['f'][donor],
                            donor_ids=self.train['ids'][donor])


def score(path,ref,scale):
    with np.load(path) as d:
        ids,z,k,f=[d[x] for x in ('ids','z','k','f')]
    idx=select(ref['ids'],len(ids));np.testing.assert_array_equal(ids,ref['ids'][idx]);np.testing.assert_array_equal(z,ref['z'][idx])
    n,draws=k.shape[:2]
    g=process(k.reshape(-1,6),f.reshape(-1,20,3),np.repeat(z,draws,0)).reshape(n,draws,4)
    obs=process(ref['k'][idx],ref['f'][idx],z)
    assert np.isfinite(obs).all()
    sc=g/scale;target=obs/scale
    err=np.linalg.norm(sc-target[:,None,:],axis=-1).mean(1)
    spread=np.linalg.norm(sc[:,:,None]-sc[:,None,:],axis=-1).sum(axis=(1,2))/(2*draws*(draws-1))
    finite=np.isfinite(g).all(-1);contact=finite&(g[:,:,1]>=.5)
    es=err-spread
    if not np.isfinite(es).all():raise ValueError(f'{path}: one or more candidates had unmeasurable geometry')
    return dict(targets=n,draws=draws,finite=int(finite.sum()),contact_valid=int(contact.sum()),
                contact_valid_fraction=float(contact.mean()),conditional_energy=float(es.mean()),
                mean_absolute_log_volume_error=float(np.abs(g[:,:,0]-obs[:,None,0]).mean()),
                mean_absolute_min_contact_error_angstrom=float(np.abs(g[:,:,1]-obs[:,None,1]).mean()),
                per_target_energy=es.tolist(),target_ids=ids.tolist(),per_target_contact_fraction=contact.mean(1).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=HERE/'mp20-data');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fit-data',type=Path,help='Source of retrieval donors/scales for transfer; defaults to evaluation train split')
    p.add_argument('--seeds',nargs='+',type=int,default=[11,23,37]);p.add_argument('--targets',type=int,default=128)
    p.add_argument('--draws',type=int,default=5);p.add_argument('--tag-prefix',default='')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    source=args.fit_data or args.data;train=load(source/'train.npz');retr=Retrieval(train)
    indices=select(train['ids'],2048); scale_data=process(train['k'][indices],train['f'][indices],train['z'][indices]);assert np.isfinite(scale_data).all()
    scale=scale_data.std(0,ddof=1).clip(1e-5)
    report=dict(scale=scale.tolist(),scale_sample=len(scale_data),train_sha256=sha(source/'train.npz'),
                columns=['log volume per atom','minimum site-neighbor distance','median site-neighbor distance','90th percentile site-neighbor distance'])
    (args.out/(args.tag_prefix+'score-scale.json')).write_text(json.dumps(report,indent=2)+'\n')
    for part in ['val','test']:
        ref=load(args.data/(part+'.npz'));tag=args.tag_prefix+part
        assert not set(train['ids'].tolist())&set(ref['ids'].tolist()), 'training IDs overlap held-out cases'
        for seed in args.seeds:
            retr.save(ref,seed,args.draws,args.targets,args.out/f'retrieval-seed{seed}-{tag}.npz')
            for method in ['drift','flow1','flow16','retrieval']:
                path=args.out/f'{method}-seed{seed}-{tag}.npz'
                if not path.is_file():continue
                output=score(path,ref,scale)
                output.update(method=method,seed=seed,split=tag,input_sha256=sha(path))
                path.with_suffix('.score.json').write_text(json.dumps(output,indent=1)+'\n')
                print(path.stem,round(output['conditional_energy'],4),round(output['contact_valid_fraction'],4),flush=True)


if __name__=='__main__':main()
