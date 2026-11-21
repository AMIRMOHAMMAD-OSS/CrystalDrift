"""Check frozen selections, ordered validation arrays, counts and checkpoint contents."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from .sampler import HERE,V3,weight_path
from .network import make_model
from .train import sha

def verify():
    locked=json.loads((V3/'protocol/splits.json').read_text());known={}
    for panel in ('dev','confirm'):
        with np.load(HERE/'fits/prior'/(panel+'-scores.npz')) as p:
            if not np.array_equal(p['ids'],locked[panel+'_ids']):raise ValueError('prior ID order')
    for path in sorted((HERE/'fits').iterdir()):
        sel=path/'selection.json'
        if not sel.exists():continue
        choice=json.loads(sel.read_text());check=path/choice['chosen']
        if sha(check)!=choice['sha256']:raise ValueError(f'{path}: chosen digest')
        is_long=path.name.endswith('-8pass')
        base=path.with_name(path.name.removesuffix('-8pass')) if is_long else path
        miles=[(base,'half'),(base,'full')]+([(path,'epoch4'),(path,'epoch8')] if is_long else [])
        out=[]
        for parent,stage in miles:
            source=parent/(stage+'-dev.npz');result=parent/(stage+'-dev-scores.npz')
            with np.load(source) as q,np.load(result) as r:
                if not np.array_equal(r['ids'],locked['dev_ids']) or not np.array_equal(r['ids'],q['ids']):raise ValueError('dev ID order')
                if str(r['source_sha'])!=sha(source):raise ValueError('predictions changed')
                h=np.isfinite(r['rms'])&(r['error']==0)
                if int(h.any(1).sum())!=choice['dev'][stage]['hits']:raise ValueError('reported hit count')
                if h.shape!=(96,5):raise ValueError('ordered prefix')
                out.append((stage,int(h.any(1).sum())))
        pick=max(out,key=lambda p:p[1])[0]
        if Path(check).stem!=pick:raise ValueError('selection disagrees with locked rule')
        c=torch.load(weight_path(check),weights_only=True,map_location='cpu')
        net=make_model(c['architecture']);net.load_state_dict(c['net'])
        assert all(torch.isfinite(x).all() for x in net.parameters())
        count=sum(x.numel() for x in net.parameters())
        fit=json.loads((path/'complete.json').read_text())
        if fit['parameters']!=count or fit['distinct_anchors']!=26419:raise ValueError('parameter/traversal count')
        if fit['split_sha256']!=sha(V3/'protocol/splits.json'):raise ValueError('split changed')
        confirm=path/'frozen-confirm-scores.npz'
        if confirm.exists():
            with np.load(confirm) as rr:
                assert np.array_equal(rr['ids'],locked['confirm_ids'])
                assert int(np.isfinite(rr['rms']).any(1).sum())==json.loads((path/'confirmation.json').read_text())['hits']
        known[path.name]=dict(pick=choice['chosen'],sha256=choice['sha256'],parameter_count=count,
                selected_anchors=c['anchors'],selected_updates=c['step'],selected_elapsed_job_s=c['seconds'],
                full_elapsed_job_s=fit['seconds'],dev=dict(out),confirm_frozen=confirm.exists())
    return known

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--complete',action='store_true');a=p.parse_args();results=verify()
    if a.complete:
        win=json.loads((HERE/'promotion.json').read_text())['arm']
        required={f'{tier}-{arm}-laplace-{s}' for tier in ('small','medium') for arm in ('angular',win) for s in (11,23,37)}
        required|={f'small-{arm}-laplace-11' for arm in ('pair','center0','variance','reciprocal','radial4')}
        required|={f'small-{arm}-gaussian-11' for arm in ('angular','contacts')}
        required|={f'{tier}-meanflow-laplace-11-commonorder' for tier in ('small','medium')}
        required|={'small-angular-laplace-11-8pass','medium-angular-laplace-11-8pass',f'medium-{win}-laplace-11-8pass',
                   'medium-meanflow-laplace-11-commonorder-8pass','tiny-angular-laplace-11-published','small-angular-laplace-11-published',
                   'small-angular-laplace-11-local','small-angular-laplace-11-systemaux'}
        if not (HERE/'large-storage-gate.json').exists():required.add('large-angular-laplace-11')
        gate=HERE/'reference-capacity.json'
        if gate.exists():
            decision=json.loads(gate.read_text())
            if decision['supported']:required.add(f"reference-{decision['arm']}-laplace-11")
        if required-set(results):raise ValueError(sorted(required-set(results)))
    (HERE/'fit-verification.json').write_text(json.dumps(results,indent=1))
    print('Verified',len(results),'frozen fits.',flush=True)

