"""Fit exact-composition multielement crystal models and export held-out predictions."""
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch

from family import SetNetwork,draw,objective


HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save_predictions(path,**arrays):
    tmp=path.with_suffix('.partial')
    with tmp.open('wb') as stream:np.savez_compressed(stream,**arrays)
    tmp.replace(path)


def select(ids,n):
    return np.array(sorted(range(len(ids)),key=lambda j:hashlib.sha256(str(ids[j]).encode()).hexdigest())[:n])


def load(path):
    with np.load(path,allow_pickle=False) as stream:
        d={k:stream[k] for k in ('ids','z','f','k')}
    ids,z,f,k=(d[k] for k in ('ids','z','f','k'))
    assert z.shape==(len(ids),20) and f.shape==(*z.shape,3) and k.shape==(len(z),6)
    assert (np.diff(np.where(z==0,120,z),axis=1)>=0).all() and (z[:,0]>0).all() and z.max()<=118
    assert len(set(ids.tolist()))==len(ids) and np.isfinite(k).all() and np.isfinite(f).all()
    assert (f[z==0]==0).all() and f.min()>=0 and f.max()<=1
    return d


def group_indices(z):
    _,labels,counts=np.unique(z,axis=0,return_inverse=True,return_counts=True)
    table=np.argsort(labels,kind='stable'); starts=np.cumsum(counts)-counts
    return labels,table,starts,counts


def fit(d,method,seed,args,checkpoint):
    utils=args.device
    z=torch.as_tensor(d['z'],dtype=torch.long,device=utils)
    f=torch.as_tensor(d['f'],dtype=torch.float32,device=utils)
    k=torch.as_tensor(d['k'],dtype=torch.float32,device=utils)
    mean=k.mean(0);scale=k.std(0,unbiased=False).clamp_min(1e-5);k=(k-mean)/scale
    labels,table,starts,counts=group_indices(d['z'])
    rng=np.random.default_rng(seed+321); torch.manual_seed(seed)
    net=SetNetwork().to(utils); opt=torch.optim.AdamW(net.parameters(),lr=.001)
    g=args.group_size; groups=args.batch//g
    steps=args.updates if args.updates is not None else math.ceil(len(z)/groups)*args.epochs
    seen=np.zeros(len(z),dtype=bool); anchor_seen=seen.copy(); perm=np.empty(0,dtype=int)
    started=time.perf_counter(); history=[]
    for step in range(steps):
        if len(perm)<groups:perm=np.concatenate((perm,rng.permutation(len(z))))
        anchors,perm=perm[:groups],perm[groups:]; anchor_seen[anchors]=True
        gid=labels[anchors]
        within=(rng.random((groups,g))*counts[gid,None]).astype(int)
        indices=table[starts[gid,None]+within];indices[:,0]=anchors
        indices=indices.reshape(-1); seen[indices]=True
        ix=torch.as_tensor(indices,dtype=torch.long,device=utils)
        loss=objective(net,method,z[ix],k[ix],f[ix],g)
        opt.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(net.parameters(),10)
        opt.step()
        if step==0 or (step+1)%200==0 or step+1==steps:
            if not torch.isfinite(norm):raise RuntimeError(f'Nonfinite gradient at {step}')
            history.append({'step':step+1,'loss':float(loss.detach()),'grad_norm':float(norm)})
    if utils.startswith('cuda'):torch.cuda.synchronize()
    train_seconds=time.perf_counter()-started
    state={'model':{key:v.detach().cpu() for key,v in net.state_dict().items()},
           'mean':mean.cpu(),'scale':scale.cpu(),'method':method,'seed':seed}
    tmp=checkpoint.with_suffix('.tmp'); torch.save(state,tmp);tmp.replace(checkpoint)
    return dict(seed=seed,method=method,updates=steps,train_rows=len(z),unique_positive_rows=int(seen.sum()),
                unique_anchor_rows=int(anchor_seen.sum()),batch=args.batch,group_size=g,condition='exact cell composition',
                width=64,parameters=sum(p.numel() for p in net.parameters()),optimizer='AdamW, lr 0.001, defaults',
                train_seconds=train_seconds,train_sha256=sha(args.data/'train.npz'),
                device=utils,threads=torch.get_num_threads(),python=platform.python_version(),
                torch=torch.__version__,numpy=np.__version__,history=history)


def predict(state,part,seed,method,tag,steps,args):
    net=SetNetwork().to(args.device)
    net.load_state_dict(state['model']);net.eval()
    selected=select(part['ids'],args.targets);ids,z=part['ids'][selected],part['z'][selected]
    rng=np.random.default_rng(100000+seed)
    all_z=np.repeat(z,args.draws,axis=0)
    all_k=rng.standard_normal((len(all_z),6)).astype(np.float32)
    all_f=rng.random((len(all_z),20,3)).astype(np.float32)
    out_k=[];out_f=[]; tic=time.perf_counter()
    mean,scale=state['mean'].to(args.device),state['scale'].to(args.device)
    for j in range(0,len(all_z),128):
        rows=slice(j,j+128)
        a=torch.as_tensor(all_z[rows],dtype=torch.long,device=args.device)
        k=torch.as_tensor(all_k[rows],device=args.device);f=torch.as_tensor(all_f[rows],device=args.device)
        u,v=draw(net,method,a,k,f,steps)
        out_k.append((u*scale+mean).cpu().numpy());out_f.append(v.cpu().numpy())
    k=np.concatenate(out_k).reshape(len(z),args.draws,6)
    f=np.concatenate(out_f).reshape(len(z),args.draws,20,3)
    assert np.isfinite(k).all() and np.isfinite(f).all() and (f*(z[:,None,:,None]==0)==0).all()
    name=(method if method=='drift' else f'flow{steps}')+f'-seed{seed}-{tag}'
    path=args.out/(name+'.npz');save_predictions(path,ids=ids,z=z,k=k,f=f)
    print(path.name,len(ids),'targets,',round(time.perf_counter()-tic,2),'s',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=HERE/'mp20-data');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--methods',nargs='+',choices=['drift','flow'],default=['drift','flow'])
    p.add_argument('--seeds',nargs='+',type=int,default=[11,23,37]);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu');p.add_argument('--group-size',type=int,default=4)
    p.add_argument('--batch',type=int,default=64);p.add_argument('--epochs',type=int,default=1);p.add_argument('--updates',type=int)
    p.add_argument('--targets',type=int,default=128);p.add_argument('--draws',type=int,default=5)
    p.add_argument('--flow-steps',nargs='+',type=int,default=[1,16]);p.add_argument('--replay',type=Path)
    p.add_argument('--tag-prefix',default='',help='Label cross-dataset predictions distinctly')
    args=p.parse_args();args.out.mkdir(exist_ok=True,parents=True);torch.set_num_threads(args.threads)
    assert args.batch%args.group_size==0 and args.batch>=args.group_size>=2
    splits={part:load(args.data/(part+'.npz')) for part in ['train','val','test']}
    ids=[set(value['ids'].tolist()) for value in splits.values()]
    assert not any(ids[i]&ids[j] for i in range(3) for j in range(i))
    for seed in args.seeds:
        for method in args.methods:
            base=f'{method}-seed{seed}'; checkpoint=(args.replay or args.out)/(base+'.pt')
            if not args.replay:
                info=fit(splits['train'],method,seed,args,checkpoint)
                (args.out/(base+'-fit.json')).write_text(json.dumps(info,indent=2)+'\n')
                print(base,info['updates'],round(info['train_seconds'],1),'training seconds',flush=True)
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert state['method']==method and state['seed']==seed
            for part in ['val','test']:
                for nfe in ([1] if method=='drift' else args.flow_steps):
                    predict(state,splits[part],seed,method,args.tag_prefix+part,nfe,args)


if __name__=='__main__':main()
