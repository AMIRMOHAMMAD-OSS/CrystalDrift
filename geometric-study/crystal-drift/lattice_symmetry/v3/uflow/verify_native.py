"""Check the published objective against an analytic differentiable velocity and a sealed EMA sample."""
import argparse
import json
import numpy as np
import torch
from runtime import HERE,ROOT,atomic_json,counts,model,propose,transport

class AffineProbe(torch.nn.Module):
    def __init__(self, coefficient):
        super().__init__()
        self.coefficient=torch.nn.Parameter(torch.tensor(coefficient,dtype=torch.float64))
    def forward(self,x,t,r,atom_types,mask,spacegroup):
        feature=.3*x+(t-.2*r)[:,None,None]+.001*spacegroup[:,None,None]
        return self.coefficient*feature*mask[...,None]

def objective():
    flow=transport()
    B,N,D=12,100,12
    x=torch.sin(torch.arange(B*N*D,dtype=torch.float64).reshape(B,N,D)*.13)
    mask=torch.arange(N)[None,:]<(torch.arange(B)[:,None]+2)
    x=x*mask[...,None]
    types=mask.long()*14
    groups=torch.tensor([1,4,43,47,125,162,225,64,186,34,5,70])
    cond=dict(atom_types=types,mask=mask,spacegroup=groups,generate_atom_types=False)
    net,ema=AffineProbe(.37),AffineProbe(-.42)
    seed=319
    torch.manual_seed(seed);np.random.seed(seed)
    loss,mse=flow.loss(net,x,cond,ema_model=ema)
    grad=torch.autograd.grad(loss,net.coefficient)[0].item()
    # Independent straight-line path and analytic directional derivative (v_hat,1,0).
    torch.manual_seed(seed);rng=np.random.RandomState(seed)
    rawn=rng.randn(B,2).astype("f")-.4
    times=1./(1.+np.exp(-rawn))
    t=np.max(times,axis=1);r=np.min(times,axis=1);same=rng.permutation(B)[:B//4]
    r[same]=t[same]
    t,r=(torch.tensor(a)[:,None,None] for a in (t,r))
    noise=torch.randn_like(x)
    e=noise-noise.mean(1,keepdim=True)
    v=e-x;z=x+t*v
    uncond=-.42*(.3*z+.8*t)*mask[...,None]
    cfg=torch.rand(B)<.8
    vh=torch.where(cfg[:,None,None],v,1.5*v-.5*uncond)
    actual_group=torch.where(cfg,0,groups)[:,None,None]
    feature=(.3*z+(t-.2*r)+.001*actual_group)*mask[...,None]
    u=.37*feature
    jvp=(.37*(.3*vh+1.))*mask[...,None]
    residual=(u-(vh-(t-r)*jvp))*mask[...,None]
    norm=mask.sum(1)*D
    e2=residual.square().sum((1,2))/norm
    w=(e2+.001).reciprocal()
    expected_grad=float((2*w*(residual*feature).sum((1,2))/norm).mean())
    errors=dict(loss=abs(float(loss.detach())-float((w*e2).mean())),
                mse=abs(float(mse)-float(e2.mean())),gradient=abs(grad-expected_grad))
    assert max(errors.values()) < 2e-7
    return dict(errors=errors,actual_gradient=grad,analytic_gradient=expected_grad,
                affine_coefficient=.37,jvp_direction="(v_guided,1,0)",zero_interval_rows=len(same),
                raw_velocity_rows=int(cfg.sum()),batch=B,padding=N)

def released_check(device,probe_size=4):
    official=ROOT/"revision/full/external/mp20"
    with np.load(official/"targets.npz") as f:
        ids=f["ids"][:probe_size];z=f["z"][:probe_size]
    with np.load(official/"draw05-cases00000-00128.npz") as ref:
        assert np.array_equal(ref["ids"][:probe_size],ids)
        ref_k,ref_f=ref["k"][:probe_size],ref["f"][:probe_size]
    net=model("released",device).eval()
    count=counts(net)
    assert count["registered"]==131215116
    flow=transport()
    torch.manual_seed(2026000000+5*1000000)
    noise=flow.sample_prior(24,100,12,"cpu")[:probe_size]
    calls=[]
    hook=net.register_forward_hook(lambda *_:calls.append(1))
    cell,f,zout=propose(net,flow,z,noise)
    hook.remove()
    assert len(calls)==1 and np.array_equal(zout,z)
    G=cell.astype(float)@cell.astype(float).transpose(0,2,1)
    w,q=np.linalg.eigh(G)
    S=(q*(np.log(w)/2)[:,None,:])@q.transpose(0,2,1)
    k=np.stack([S[:,0,0],S[:,1,1],S[:,2,2],np.sqrt(2)*S[:,0,1],np.sqrt(2)*S[:,0,2],np.sqrt(2)*S[:,1,2]],axis=1)
    fk=float(np.abs(k-ref_k).max())
    err=f-ref_f;ff=float(np.abs(err-np.round(err)).max())
    assert fk<.003 and ff<.003
    return dict(tier="released",parameters=count,device=device,material_ids=ids.tolist(),
                draw_zero_based=5,null_sg=True,actual_calls=len(calls),candidates=len(ids),
                max_canonical_chart_error=fk,max_fractional_torus_error=ff,torch=torch.__version__)

def learned_path(device):
    with np.load(HERE/"train-cache/00000.npz") as data:
        rows=np.arange(8)
        z=np.array([np.pad(np.sort(z[z>0]),(0,sum(z==0))) for z in data["z"][rows]])
        ids=data["ids"][rows].tolist()
    torch.manual_seed(2487)
    flow=transport()
    noise=flow.sample_prior(len(z),100,12,"cpu")
    L0,F0,_=propose(None,flow,z,noise,device=device)
    out={}
    for tier in ("small","medium"):
        net=model(tier,device).eval()
        uL,uF,_=propose(net,flow,z,noise)
        assert np.array_equal(uL,L0) and np.array_equal(uF,F0)
        base=HERE/f"fits/{tier}-11"
        selected=json.loads((base/"selection.json").read_text())
        net.load_state_dict(torch.load(base/selected["checkpoint"],weights_only=True,map_location="cpu"))
        L,F,Z=propose(net,flow,z,noise)
        delta=F-F0
        shift=np.linalg.norm((delta-np.rint(delta)),axis=2)
        G=L@np.swapaxes(L,1,2);G0=L0@np.swapaxes(L0,1,2)
        assert np.all(np.linalg.norm(G-G0,axis=(1,2))>1e-4) and np.any(shift[Z>0]>.01)
        out[tier]=dict(identical_untrained_native_prior=True,all_geometry_rows_move=len(z),
                 median_fractional_torus_movement=float(np.median(shift[Z>0])),
                 median_log_volume_change=float(np.median(np.log(np.abs(np.linalg.det(L))/np.abs(np.linalg.det(L0))))),
                 active_shifted_sites=int((shift[Z>0]>1e-4).sum()),sites=int((Z>0).sum()),sha=selected["checkpoint_sha"])
    return dict(training_ids=ids,seed=2487,device=device,results=out)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--release",action="store_true");p.add_argument("--path",action="store_true");p.add_argument("--device",default="cpu")
    a=p.parse_args()
    torch.set_num_threads(1);torch.set_float32_matmul_precision("highest")
    if a.release:
        atomic_json(HERE/"released-replay.json",released_check(a.device))
    elif a.path:
        atomic_json(HERE/"live-path.json",learned_path(a.device))
    else:
        atomic_json(HERE/"objective-check.json",objective())
