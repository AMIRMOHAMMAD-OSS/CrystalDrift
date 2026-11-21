"""Wall time from supplied element slots to host cell and Cartesian/fractional sites."""
import argparse
import csv
import json
import math
import os
import platform
import time
from pathlib import Path

for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[v]='1'
import numpy as np
import torch

from evaluate import Retrieval
from experiment import load,select
from family import SetNetwork,draw


def bundle(run,seeds):
    folder=run/'portable';folder.mkdir(exist_ok=True)
    for seed in seeds:
        for name in ('drift','flow'):
            source=torch.load(run/f'{name}-seed{seed}.pt',map_location='cpu',weights_only=True)
            weights={k:v.detach().numpy() for k,v in source['model'].items()}
            np.savez_compressed(folder/f'{name}-seed{seed}.npz',mean=source['mean'].numpy(),scale=source['scale'].numpy(),**weights)


def model_state(path,device):
    m=SetNetwork().eval()
    with np.load(path) as d:
        m.load_state_dict({k:torch.from_numpy(d[k].copy()) for k in m.state_dict()})
        mean=torch.from_numpy(d['mean'].copy()).to(device);scale=torch.from_numpy(d['scale'].copy()).to(device)
    return m.to(device),mean,scale


def decode(k,f):
    s=np.zeros((len(k),3,3),dtype=np.float64)
    for i in range(3):s[:,i,i]=k[:,i]
    i,j=np.triu_indices(3,1);s[:,i,j]=s[:,j,i]=k[:,3:]/math.sqrt(2)
    eig,q=np.linalg.eigh(s);cell=(q*np.exp(eig)[:,None,:])@q.transpose(0,2,1)
    return cell,f,f@cell


@torch.inference_mode()
def neural(state,kind,steps,z,device):
    m,mean,scale=state;a=torch.as_tensor(z,device=device,dtype=torch.long)
    k=torch.randn((len(z),6),device=device);f=torch.rand((*z.shape,3),device=device)
    k,f=draw(m,kind,a,k,f,steps)
    return decode((k*scale+mean).cpu().numpy(),f.cpu().numpy())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--device',choices=('cpu','cuda'),required=True)
    p.add_argument('--export',action='store_true');p.add_argument('--seeds',nargs='+',type=int,default=[11,23,37]);p.add_argument('--repeats',type=int,default=15)
    args=p.parse_args();device=args.device; torch.set_num_threads(1);torch.manual_seed(1951);rng=np.random.default_rng(1951)
    if args.export:bundle(args.run,args.seeds)
    test=load(args.data/'test.npz');train=load(args.data/'train.npz');cases=test['z'][select(test['ids'],128)]
    tic=time.perf_counter();retrieval=Retrieval(train);donors={tuple(z):retrieval.pool(z) for z in cases};setup=time.perf_counter()-tic
    rows=[];methods=['drift','flow1','flow16','retrieval'];warmups=3
    for seed in args.seeds:
        models={m:model_state(args.run/'portable'/f'{m}-seed{seed}.npz',device) for m in ('drift','flow')}
        for b in (1,64,256):
            for r in range(-warmups,args.repeats):
                z=cases[rng.integers(len(cases),size=b)]
                for method in rng.permutation(methods):
                    if device=='cuda':torch.cuda.synchronize()
                    t=time.perf_counter_ns()
                    if method=='retrieval':
                        pools=[donors[tuple(a)] for a in z]
                        indices=[ix[int(u*len(ix))] for ix,u in zip(pools,rng.random(len(pools)))]
                        cell,f,cart=decode(train['k'][indices],train['f'][indices])
                    else:
                        kind='drift' if method=='drift' else 'flow';steps=16 if method=='flow16' else 1
                        cell,f,cart=neural(models[kind],kind,steps,z,device)
                    if device=='cuda':torch.cuda.synchronize()
                    ms=(time.perf_counter_ns()-t)/1e6
                    assert np.isfinite(cell).all() and np.isfinite(cart).all() and np.isfinite(f).all()
                    if r>=0:rows.append(dict(seed=seed,batch=b,method=method,repeat=r,wall_ms=ms))
    out=args.run/f'latency-{device}';summary=[]
    with out.with_suffix('.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    for b in (1,64,256):
        for m in methods:
            seed_medians=[np.median([r['wall_ms'] for r in rows if r['batch']==b and r['method']==m and r['seed']==s]) for s in args.seeds]
            summary.append(dict(batch=b,method=m,seed_median_ms=seed_medians,median_ms=float(np.median(seed_medians))))
    report=dict(quantities='End-to-end wall time from supplied Z array to decoded host cell, fractional and Cartesian sites',
                retrieval='CPU in every arm; training donor tables cached before benchmarking',device=device,
                accelerator=torch.cuda.get_device_name() if device=='cuda' else platform.processor(),cuda=torch.version.cuda,
                torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),platform=platform.platform(),
                threads=1,repeats=args.repeats,untimed_warmups=warmups,seeds=args.seeds,donor_setup_seconds=setup,
                observations=len(rows),summaries=summary)
    out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n'); print(device,summary,flush=True)


if __name__=='__main__':main()
