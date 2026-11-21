"""Reopen an atomic training checkpoint twice, advance identical real banks, and compare."""
import json
import numpy as np
import torch
from .sampler import HERE
from .network import make_model
from .train import CACHE,tensors,sha
from ..geometry.descriptors import feature_blocks,combine,drift_loss

def main():
    torch.set_num_threads(1)
    ck=HERE/'fits/small-angular-laplace-11/resume.pt'
    data=torch.load(ck,weights_only=True,map_location='cpu')
    sc=np.load(CACHE/'calibration.npz');stats={'base':(sc['base_mean'],sc['base_scale'])}
    with np.load(CACHE/'train-001.npz') as src:
        a=tensors(src,np.arange(16),'cpu');pos=combine({'base':torch.from_numpy(src['p_base'][:8])},stats).reshape(4,2,-1)
    results=[]
    for _ in range(2):
        obj=torch.load(ck,weights_only=True,map_location='cpu');model=make_model(obj['architecture'])
        model.load_state_dict(obj['net']);opt=torch.optim.AdamW(model.parameters())
        opt.load_state_dict(obj['optimizer'])
        L,F=model(*a)
        y=combine(feature_blocks(L,F,a[2],a[3],'angular'),stats).reshape(4,4,-1)
        loss=drift_loss(y[:,:2],pos,y[:,2:],temperatures=tuple(sc['temperatures']))
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        with torch.no_grad():post=model(*a)
        results.append((torch.cat([p.detach().flatten() for p in model.parameters()]),post[0],post[1]))
    errors=[float((a-b).abs().max()) for a,b in zip(*results)]
    assert max(errors)==0
    record=dict(optimizer_checkpoint=sha(ck),saved_step=data['step'],saved_shards=data['shard'],continuation_steps=1,
                anchors=4,result_parameter_max=errors[0],result_cell_max=errors[1],result_sites_max=errors[2],device='cpu')
    (HERE/'resume-replay.json').write_text(json.dumps(record,indent=2));print(record,flush=True)
if __name__=='__main__':main()

