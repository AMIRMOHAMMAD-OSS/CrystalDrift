"""Rebuild the exact first-pass minibatch order and compare saved final RNG states."""
import hashlib,json
import numpy as np
import torch
from .train import HERE,CACHE

def reconstruct(arm,separate=False,seed=11,passes=1):
    inv=json.loads((CACHE/'inventory.json').read_text());base=np.array(inv['order'])
    rng=np.random.default_rng(seed);events=np.random.default_rng(seed+74419) if separate else rng
    orders=[]
    plan=np.arange((len(base)+127)//128)
    if seed!=11:np.random.default_rng(seed+1802).shuffle(plan)
    for epoch in range(passes):
        indexes=base if epoch==0 else np.random.default_rng(55171+epoch).permutation(len(base))
        for index in plan:
            raw=indexes[index*128:index*128+128];take=rng.permutation(len(raw));orders.extend(raw[take].tolist())
            if arm=='meanflow':
                for j in range(0,len(raw),16):events.random((3,2*len(take[j:j+16])))
    return np.array(orders),rng.bit_generator.state

def main():
    columns={};meta={}
    for label,arm,separate in [('angular','angular',False),('meanflow','meanflow',False),('meanflow_strict','meanflow',True)]:
        seq,state=reconstruct(arm,separate)
        columns[label]=seq
        tag=f'small-{arm}-laplace-11'+('-commonorder' if separate else '')
        file=HERE/'fits'/tag/'resume.pt'
        equal=None
        if file.exists():
            obj=torch.load(file,map_location='cpu',weights_only=True)
            if obj['anchors']==len(seq):
                equal=(obj['rng']==state)
                if not equal:raise ValueError(tag+' RNG replay failed')
        meta[label]=dict(sha256=hashlib.sha256(seq.astype('<i4').tobytes()).hexdigest(),saved_rng_exact=equal)
    assert np.array_equal(columns['meanflow_strict'],columns['angular'])
    assert np.array_equal(np.sort(columns['meanflow']),np.sort(columns['angular']))
    meta['same_positions_original']=int((columns['meanflow']==columns['angular']).sum())
    meta['same_positions_strict']=int((columns['meanflow_strict']==columns['angular']).sum())
    from .sampler import weight_path
    comparisons={}
    for seed in (11,23,37):
        seq,state=reconstruct('angular',seed=seed);columns['matched_'+str(seed)]=seq
        if len(set(seq))!=len(seq):raise ValueError('Anchor repeated within a traversal')
        for tier in ('small','medium'):
            for arm in ('angular','contacts'):
                label=f'{tier}-{arm}-laplace-{seed}';actual=HERE/'fits'/label/'full.pt'
                if not (actual.parent/'complete.json').exists():continue
                ck=torch.load(weight_path(actual),weights_only=True,map_location='cpu')
                if ck['rng']!=state or ck['anchors']!=len(seq):raise ValueError('Nonpaired '+label)
                comparisons[label]=dict(updates=ck['step'],anchors=ck['anchors'],final_rng_exact=True)
    for target in (HERE/'fits').glob('*-8pass'):
        if not (target/'complete.json').exists():continue
        arm=target.name.split('-')[1];strict='commonorder' in target.name
        for count in (4,8):
            seq,state=reconstruct(arm,strict,passes=count);ck=torch.load(weight_path(target/f'epoch{count}.pt'),weights_only=True,map_location='cpu')
            if ck['rng']!=state or ck['anchors']!=len(seq):raise ValueError('Extended ordering '+target.name)
            comparisons[target.name+f'-epoch{count}']=dict(updates=ck['step'],anchors=ck['anchors'],final_rng_exact=True,
                order_sha256=hashlib.sha256(seq.astype('<i4').tobytes()).hexdigest())
    meta['checks']=comparisons
    np.savez_compressed(HERE/'paired-anchor-order.npz',**columns)
    (HERE/'paired-anchor-order.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()

