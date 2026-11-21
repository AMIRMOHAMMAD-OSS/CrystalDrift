"""Replay frozen MP-20 checkpoints on the locked 128-ID Alexandria validation cohort."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'

import numpy as np
import torch

from evaluate import Retrieval,process,score
from experiment import load,predict,select,sha
from prepare_alex import write_json


HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,default=HERE/'alex-validation-transfer')
    p.add_argument('--source',type=Path,default=HERE/'mp20-data');p.add_argument('--checkpoints',type=Path,default=HERE/'runs'/'mp20')
    p.add_argument('--out',type=Path,default=HERE/'runs'/'mp20-to-alex');p.add_argument('--seeds',type=int,nargs='+',default=[11,23,37])
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    audit=json.loads((args.cohort/'cohort.json').read_text())
    assert audit['selected']==128 and not any(audit[k] for k in ('selected_id_overlap','selected_embedded_alias_overlap','selected_reduced_formula_overlap'))
    part=load(args.cohort/'val.npz');assert part['ids'].tolist()==audit['ids']
    assert sha(args.cohort/'val.npz')==audit['selected_sha256'] and sha(args.source/'train.npz')==audit['source_train_sha256']
    train=load(args.source/'train.npz');retr=Retrieval(train)
    if (args.checkpoints/'score-scale.json').exists():
        info=json.loads((args.checkpoints/'score-scale.json').read_text());assert info['train_sha256']==audit['source_train_sha256']
        scale=np.asarray(info['scale'])
    else:
        ids=select(train['ids'],2048);values=process(train['k'][ids],train['f'][ids],train['z'][ids])
        assert np.isfinite(values).all();scale=values.std(0,ddof=1).clip(1e-5)
    write_json(args.out/'transfer-design.json',{'source':'full MP-20 training','target':'released Alex-MP validation, Alexandria namespace',
        'selection':audit['rule'],'splits_scored':['val'],'seeds':args.seeds,'draws':5,'target_count':128,
        'fit_source_sha256':audit['source_train_sha256'],'selected_target_sha256':audit['selected_sha256'],
        'feature_scale_from_mp20_train':scale.tolist()})
    parameters=SimpleNamespace(device='cpu',draws=5,targets=128,out=args.out)
    scores={}
    for seed in args.seeds:
        for method in ('drift','flow'):
            ck=args.checkpoints/f'{method}-seed{seed}.pt';seedlog=json.loads((args.checkpoints/f'{method}-seed{seed}-fit.json').read_text())
            assert seedlog['train_sha256']==audit['source_train_sha256']
            state=torch.load(ck,map_location='cpu',weights_only=True)
            assert state['method']==method and state['seed']==seed
            for steps in ([1] if method=='drift' else [1,16]):
                predict(state,part,seed,method,'val',steps,parameters)
        retr.save(part,seed,5,128,args.out/f'retrieval-seed{seed}-val.npz')
        for method in ('drift','flow1','flow16','retrieval'):
            file=args.out/f'{method}-seed{seed}-val.npz';v=score(file,part,scale)
            v.update(method=method,seed=seed,split='val',input_sha256=sha(file))
            write_json(file.with_suffix('.score.json'),v)
            scores[method,seed]=v
    n=len(part['ids']);boot=np.random.default_rng(741123).multinomial(n,np.full(n,1/n),size=10000)
    summary={};case={}
    for m in ('drift','flow1','flow16','retrieval'):
        case[m]=np.mean([scores[m,s]['per_target_energy'] for s in args.seeds],axis=0)
        lo,hi=np.quantile((boot@case[m])/n,[.025,.975])
        summary[m]=dict(energy=float(case[m].mean()),low=float(lo),high=float(hi),
            valid_fraction=float(np.mean([scores[m,s]['contact_valid_fraction'] for s in args.seeds])))
    contrasted={}
    for m in ('flow1','flow16','retrieval'):
        diff=case['drift']-case[m];lo,hi=np.quantile((boot@diff)/n,[.025,.975])
        contrasted['drift minus '+m]=[float(diff.mean()),float(lo),float(hi)]
    write_json(args.out/'geometry-summary.json',dict(partition='released validation',n_targets=n,seeds=args.seeds,
        methods=summary,paired_energy=contrasted))
    print('Official validation, three-checkpoint geometric means',summary,flush=True)


if __name__=='__main__':main()

