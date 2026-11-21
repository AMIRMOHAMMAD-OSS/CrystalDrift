"""Frozen initialization control with the same small network and symbolic seeds."""
import json
import torch
from .network import OrbitNet,TIERS
from .sampler import HERE,Sampler
from .train import save_pt,dev,sha
from .select_weights import score

def main():
    out=HERE/'controls/untrained-small-11';out.mkdir(parents=True,exist_ok=True)
    s=Sampler();torch.manual_seed(11);w,d=TIERS['small'];arch=dict(width=w,depth=d,global_context=True)
    net=OrbitNet(**arch);net.eval();p=out/'initial.pt'
    if not p.exists():save_pt(p,dict(architecture=arch,net=net.state_dict(),seed=11,step=0,
               conditioning='same train-only symbolic seeds as the fitted small model'))
    stored=s.load(p)
    for k,v in net.state_dict().items():assert torch.equal(v,stored.state_dict()[k])
    rec=dict(checkpoint=sha(p),training_updates=0,parameters=sum(q.numel() for q in net.parameters()))
    for panel in ('dev','confirm'):
        path=out/(panel+'.npz')
        if not path.exists():dev(s,stored,path,panel)
        rec[panel]=score(path)
    (out/'control.json').write_text(json.dumps(rec,indent=2))
    print(json.dumps(rec),flush=True)
if __name__=='__main__':main()

