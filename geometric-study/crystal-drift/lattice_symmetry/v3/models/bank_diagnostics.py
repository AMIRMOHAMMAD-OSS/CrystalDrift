"""Conditional positive variation and live sensitivity of every proposed information block."""
import json
import numpy as np
import torch
from .train import CACHE,HERE,tensors,base_dim
from .sampler import Sampler
from ..geometry.descriptors import feature_blocks,combine,drift_loss
torch.set_num_threads(1)

def main():
    s=Sampler();keys=['base','variance','contacts','reciprocal'];different=0
    changed={k:0 for k in keys};sumdiff={k:0. for k in keys}
    for f in sorted(CACHE.glob('train-*.npz')):
        with np.load(f) as v:
            a=v['pidx'].reshape(-1,2)
            mask=a[:,0]!=a[:,1];different+=int(mask.sum())
            for k in keys:
                x=v['p_'+k].reshape(len(a),2,-1)
                d=np.linalg.norm(x[:,0]-x[:,1],axis=1)
                changed[k]+=int((d[mask]>1e-8).sum());sumdiff[k]+=float(d[mask].sum())
    sc=np.load(CACHE/'calibration.npz')
    stats={k:(sc[k+'_mean'],sc[k+'_scale']) for k in keys}
    path=HERE/'fits/small-angular-laplace-11';selection=json.loads((path/'selection.json').read_text())
    model=s.load(path/selection['chosen']);checks={}
    with np.load(CACHE/'train-000.npz') as v:
        for arm in ('angular','variance','contacts','reciprocal'):
            site=[];cell=[];negdiv=[];grad=[];neighbor=[];field=[]
            for lo in range(0,128,8):
                at=np.arange(lo*4,(lo+8)*4);x=tensors(v,at,'cpu')
                with torch.no_grad():L,F=model(*x)
                L=L.detach().requires_grad_(True);F=F.detach().requires_grad_(True)
                raw=feature_blocks(L,F,x[2],x[3],arm)
                live=combine(raw,stats).reshape(8,4,-1)
                p={k:torch.from_numpy(v['p_'+k][2*lo:2*(lo+8)]) for k in raw}
                pos=combine(p,stats).reshape(8,2,-1)
                loss,info=drift_loss(live[:,:2],pos,live[:,2:],temperatures=tuple(sc['temperatures']),return_info=True)
                tL,tF=torch.autograd.grad(loss,(L,F))
                # Loss averages across 16 differentiable queries. Multiply back to individual gradients.
                q=torch.tensor([4*j+k for j in range(8) for k in (0,1)],dtype=torch.long)
                cell.extend((16*tL[q].flatten(1).norm(dim=1)).tolist())
                site.extend((16*tF[q].flatten(1).norm(dim=1)).tolist())
                negdiv.extend((live[:,2]-live[:,3]).detach().norm(dim=1).tolist())
                neighbor.extend(a['effective_negative'] for a in info);field.extend(a['raw_field_rms'] for a in info)
            checks[arm]=dict(real_anchors=128,median_cell_gradient=float(np.median(cell)),
                median_fractional_site_gradient=float(np.median(site)),fraction_zero_site_gradient=float(np.mean(np.array(site)<1e-8)),
                median_distinct_negative_feature_distance=float(np.median(negdiv)),mean_effective_negative=float(np.mean(neighbor)),
                mean_raw_field_rms=float(np.mean(field)))
    out=dict(exact_formula_second_positive_distinct_ID=different,
        raw_block_diff_gt1e_8_when_different=changed,mean_raw_distance_when_different={k:v/different for k,v in sumdiff.items()},
        selected_angular_checkpoint=selection['sha256'],training_shard0=checks)
    (HERE/'bank-diagnostics.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()

