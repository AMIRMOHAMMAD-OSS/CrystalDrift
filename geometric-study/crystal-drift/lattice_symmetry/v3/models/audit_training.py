"""Recheck train-bank exclusion, observed gradients, one-call replay and a tiny real fit."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from .sampler import HERE,V3,Sampler
from .network import OrbitNet,pack,TIERS
from .prepare import CACHE
from .train import sha
from ..geometry.descriptors import feature_blocks,combine,drift_loss
torch.set_num_threads(1)

def main():
    s=Sampler();stats={};cal=np.load(CACHE/'calibration.npz')
    for k in ('base','contacts','variance','reciprocal'):
        stats[k]=(cal[k+'_mean'],cal[k+'_scale'])
    seen=[];groups={};draws=examples=fall=0;forbidden=0
    norms={k:[] for k in ('contacts','reciprocal','variance')}
    coords=[];movable=[];shell_zero=0
    for path in sorted(CACHE.glob('train-*.npz')):
        with np.load(path) as data:
            i=data['indices'];seen.extend(i.tolist());d=data['donor'];c=data['cell_donor']
            q=np.repeat(i,4);draws+=len(q);fall+=int(np.sum(d<0))
            if not np.array_equal(data['pidx'][::2],i):
                raise AssertionError('full anchor traversal broken')
            forbidden+=sum(s.red[a]==s.red[b] for a,b in zip(q,c))
            forbidden+=sum(s.red[a]==s.red[b] for a,b in zip(q,d) if b>=0)
            for k in norms:norms[k].extend(np.linalg.norm(data['p_'+k][::2],axis=1).tolist())
            movable.extend(data['x5'].max(1)>0)
    assert len(set(seen))==len(s.rows)==len(seen) and forbidden==0
    path=HERE/'fits/small-angular-laplace-11'
    record=json.loads((path/'selection.json').read_text());net=s.load(path/record['chosen'])
    with np.load(path/(Path(record['chosen']).stem+'-dev.npz')) as raw:
        ids=raw['ids'];expectedL=raw['L'][0];expectedF=raw['F'][0];expectedZ=raw['Z'][0]
    from .sampler import ROOT
    with np.load(ROOT/'scaled/mp20-data/val.npz') as raw:
        idx=list(raw['ids']).index(ids[0]);z=raw['z'][idx]
    seeds=s.seeds(z,5,int(np.random.SeedSequence([38471,idx]).generate_state(1)[0]))
    calls=[];hook=net.register_forward_hook(lambda _m,a,_r:calls.append(len(a[0])))
    s.apply(None,seeds);assert not calls
    outs=[s.apply(net,[seed])[0] for seed in seeds]
    assert calls==[1]*5
    hook.remove()
    errL=max(abs(l-v).max() for (l,_,_),v in zip(outs,expectedL))
    errF=max(abs((f-v[:len(f)]+.5)%1-.5).max() for (_,f,_),v in zip(outs,expectedF))
    assert errL<2e-4 and errF<2e-4
    # Overfit one real anchor with appreciable free coordinate motion and a nonzero pair graph.
    with np.load(CACHE/'train-000.npz') as data:
        for idx in range(len(data['indices'])):
            if (data['x5'][4*idx:4*idx+4].max(1)>0).all() and np.abs(data['targetF'][4*idx:4*idx+4]).max()>.1:break
        args=[torch.as_tensor(data['x'+str(k)][4*idx:4*idx+4],dtype=torch.long if k in (2,3) else torch.float) for k in range(8)]
        target=torch.tensor(data['p_base'][2*idx:2*idx+2]);pos=combine({'base':target},stats)[None]
        mid=s.ids[int(data['indices'][idx])]
    torch.manual_seed(5519);model=OrbitNet(*TIERS['tiny']);opt=torch.optim.AdamW(model.parameters(),lr=.001)
    with torch.no_grad():L0,F0=model(*args)
    history=[];grad=[]
    for j in range(24):
        L,F=model(*args);raw=feature_blocks(L,F,args[2],args[3],'angular')
        y=combine(raw,stats)[None];loss=drift_loss(y[:,:2],pos,y[:,2:],temperatures=tuple(cal['temperatures']))
        opt.zero_grad();loss.backward()
        grad.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),2.)));opt.step()
        history.append(float(torch.cdist(y[:,:2],pos).mean().detach()))
    with torch.no_grad():L,F=model(*args)
    site=float((((F-F0+.5)%1-.5)@args[0]).norm(dim=-1).max())
    cell=float((L-L0).norm(dim=(-1,-2)).max())
    assert site>1e-5 and cell>1e-4
    stat=dict(eligible=len(seen),distinct_compositions=len(s.by_formula),singleton_rows=sum(len(x)==1 for x in s.by_formula.values()),
              scheduled_seed_draws=draws,p1_fallbacks=fall,excluded_reduced_formula_donors=forbidden,
              fraction_any_free=float(np.mean(movable)),raw_nonzero_fraction={k:float(np.mean(np.array(x)>1e-8)) for k,x in norms.items()},
              split_sha256=sha(V3/'protocol/splits.json'),dev_replay=dict(id=str(ids[0]),checkpoint=record['sha256'],
              serial_hook_batches=calls,zero_prior_calls=0,max_abs_L=float(errL),max_wrapped_F=float(errF)),
              overfit=dict(id=mid,seed=5519,steps=24,descr='fixed real bank, tiny OrbitNet and identical seeds on every update',
                           max_change_in_sites_A=site,max_cell_frobenius_A=cell,feature_distance=history,grad_norm=grad))
    (HERE/'training-audit.json').write_text(json.dumps(stat,indent=1))
    print(json.dumps({k:v for k,v in stat.items() if k!='overfit'},indent=1),flush=True)

if __name__=='__main__':main()

