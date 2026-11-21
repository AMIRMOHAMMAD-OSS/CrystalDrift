"""Materialize nested MP-20 candidate pools with the original 128-by-five prefixes retained."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
SCALED=ROOT/'scaled'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(SCALED))
from experiment import load, select
from evaluate import Retrieval
from family import SetNetwork, draw


def pin():
    try:
        import psutil
        proc=psutil.Process()
        choices=[c for c in proc.cpu_affinity() if c not in (0,2,4)]
        if choices:proc.cpu_affinity(choices)
    except (ImportError,AttributeError, OSError):
        pass


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def publish(path,record):
    tmp=path.with_suffix('.writing');tmp.write_text(json.dumps(record,indent=1)+'\n');tmp.replace(path)


def pool(test,train,bank,seed,method,out,draws=20):
    path=out/f'{method}-seed{seed}'
    path.mkdir(parents=True,exist_ok=True)
    source=SCALED/'runs/mp20'/f'{method}-seed{seed}-test.npz'
    ck=SCALED/'runs/mp20'/f'{"flow" if method.startswith("flow") else method}-seed{seed}.pt'
    signature=dict(archive=digest(source),checkpoint=None if method=='retrieval' else digest(ck),
                   seed=seed,draws=draws,method=method,generator_script=digest(Path(__file__)))
    if (path/'ready.json').exists():
        old=json.loads((path/'ready.json').read_text())
        if old['input']==signature and all((path/k).exists() for k in old['outputs']):return
    tic=perf_counter(); ids=test['ids']; z=test['z']; n=len(z)
    with np.load(source,allow_pickle=False) as orig:
        orig={k:orig[k] for k in orig.files}
    np.testing.assert_array_equal(ids[:128],orig['ids']);np.testing.assert_array_equal(z[:128],orig['z'])
    names={'k':((n,draws,6),np.float32),'f':((n,draws,20,3),np.float32)}
    if method=='retrieval':names['donor_index']=((n,draws),np.int32)
    mm={s:np.lib.format.open_memmap(path/(s+'.part.npy'),mode='w+',dtype=dtype,shape=shape)
        for s,(shape,dtype) in names.items()}
    rng=np.random.default_rng(np.random.SeedSequence([20261120,seed]))
    if method=='retrieval':
        train_lookup={s:i for i,s in enumerate(train['ids'])}
        reference=np.vectorize(train_lookup.__getitem__)(orig['donor_ids'])
    else:
        state=torch.load(ck,map_location='cpu',weights_only=True)
        net=SetNetwork().eval();net.load_state_dict(state['model'])
        kind=state['method'];steps=16 if method=='flow16' else 1
    for start in range(0,n,16):
        stop=min(n,start+16);chem=z[start:stop]
        if method=='retrieval':
            donor=np.array([rng.choice(bank.pool(row),size=draws) for row in chem])
            if start<128:donor[:min(stop,128)-start,:5]=reference[start:min(stop,128)]
            mm['donor_index'][start:stop]=donor
            k=train['k'][donor];f=train['f'][donor]
        else:
            expanded=np.repeat(chem,draws,axis=0);zc=torch.as_tensor(expanded,dtype=torch.long)
            noise_k=torch.from_numpy(rng.normal(size=(len(zc),6)).astype(np.float32))
            noise_f=torch.from_numpy(rng.random((len(zc),20,3),dtype=np.float32))
            with torch.inference_mode():
                y,x=draw(net,kind,zc,noise_k,noise_f,steps)
                k=(y*state['scale']+state['mean']).numpy().reshape(-1,draws,6)
                f=x.numpy().reshape(-1,draws,20,3)
            if start<128:
                hi=min(stop,128)
                k[:hi-start,:5]=orig['k'][start:hi];f[:hi-start,:5]=orig['f'][start:hi]
        if not np.isfinite(k).all() or not np.isfinite(f).all():raise ValueError('Nonfinite candidate')
        mm['k'][start:stop]=k;mm['f'][start:stop]=f
    for field in ('k','f'):np.testing.assert_array_equal(mm[field][:128,:5],orig[field])
    for array in mm.values():array.flush()
    del array,mm
    outputs={}
    for name in names:
        final=path/(name+'.npy');(path/(name+'.part.npy')).replace(final);outputs[final.name]=digest(final)
    publish(path/'ready.json',dict(input=signature,outputs=outputs,targets=n,draws=draws,
        original_prefix_exact=True,elapsed_seconds=perf_counter()-tic))
    print(method,seed,n,draws,'seconds',round(perf_counter()-tic,1),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods',nargs='+',default=['retrieval','drift','flow1','flow16'])
    p.add_argument('--seeds',nargs='+',type=int,default=[11,23,37]);args=p.parse_args()
    pin();torch.set_num_threads(1)
    out=HERE/'mp20';out.mkdir(parents=True,exist_ok=True)
    whole=load(SCALED/'mp20-data/test.npz');idx=select(whole['ids'],len(whole['ids']))
    test={key:v[idx] for key,v in whole.items()};train=load(SCALED/'mp20-data/train.npz')
    if not (out/'cohort.npz').exists():
        tmp=out/'cohort.writing'
        with tmp.open('wb') as stream:np.savez_compressed(stream,**test,release_index=idx)
        tmp.replace(out/'cohort.npz')
    with np.load(out/'cohort.npz') as ref:
        for key in test:np.testing.assert_array_equal(test[key],ref[key])
    bank=Retrieval(train)
    for method in args.methods:
        for seed in args.seeds:pool(test,train,bank,seed,method,out)


if __name__=='__main__':main()
