"""Exact-composition physical prior and one-call service. The prior is fitted only on train.npz."""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from scaled.family import SetNetwork, mean_sites
from data import pack_gram, unpack_gram

ARMS = ("radial","angular","power0","fixed-angular","flow16")
GROUPS = 8
BANK = 4
SEEDS = (11,23,37)
GENERATOR = dict(cutoff=4.0,width=24,n_radial=5)
DESCRIPTOR = dict(cutoff=4.0,n_radial=2,species_slots=7)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as fp:
        for block in iter(lambda:fp.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def write_json(path,data):
    path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_suffix(".building")
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf8")
    tmp.replace(path)


def save_npz(path,**kw):
    path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_suffix(".building")
    with tmp.open("wb") as fp:np.savez_compressed(fp,**kw)
    tmp.replace(path)


def count_design(z):
    n=(z>0).sum(1)
    if (n<1).any() or (n>20).any():raise ValueError("1 to 20 occupied sites required")
    x=np.zeros((len(z),139))
    b,i=np.nonzero(z)
    np.add.at(x,(b,z[b,i]),1/n[b])
    x[np.arange(len(z)),118+n]=1.
    return x,n


class Prior:
    def __init__(self,coefficients,residual,means,roots,limits):
        self.coefficients=coefficients
        self.residual=residual
        self.means=means
        self.roots=roots
        self.limits=limits

    @classmethod
    def fit(cls,data):
        z=data["z"];k=np.asarray(data["k"],dtype=np.float64)
        x,n=count_design(z)
        vol=k[:,:3].sum(1)-np.log(n)
        coef=np.linalg.solve(x.T@x+np.eye(x.shape[1]),x.T@vol)
        err=vol-x@coef
        shape=k.copy();shape[:,:3]-=k[:,:3].mean(1)[:,None]
        means=np.zeros((21,6));roots=np.zeros((21,6,6));resid=np.zeros(21)
        for count in range(1,21):
            sl=(n==count)
            a=shape[sl] if sl.sum()>7 else shape
            means[count]=a.mean(0)
            ev,q=np.linalg.eigh(np.cov(a,rowvar=False))
            roots[count]=(q*np.maximum(ev,0)**.5)@q.T
            resid[count]=max(.01,np.std(err[sl]) if sl.sum()>7 else err.std())
        return cls(coef,resid,means,roots,np.array([vol.min(),vol.max()]))

    def save(self,path):save_npz(path,**self.__dict__)

    @classmethod
    def load(cls,path):
        with np.load(path) as fp:return cls(**{k:fp[k] for k in fp.files})

    def draw(self,z,rng):
        x,n=count_design(z);b=len(z)
        vol=(x@self.coefficients + rng.standard_normal(b)*self.residual[n]).clip(*self.limits)
        sk=self.means[n]+np.einsum("bi,bij->bj",rng.standard_normal((b,6)),self.roots[n])
        s=np.zeros((b,3,3))
        ix=np.arange(3);s[:,ix,ix]=sk[:,:3]
        i,j=np.triu_indices(3,1);s[:,i,j]=s[:,j,i]=sk[:,3:]/math.sqrt(2)
        ev,q=np.linalg.eigh(s);ev=np.clip(ev,-1.2,1.2);ev-=ev.mean(1)[:,None]
        s=(q*ev[:,None,:])@q.transpose(0,2,1)
        k=np.column_stack((s[:,ix,ix],math.sqrt(2)*s[:,i,j]))
        k[:,:3]+=(vol+np.log(n))[:,None]/3
        f=rng.random((*z.shape,3),dtype=np.float32)*(z>0)[:,:,None]
        return k.astype(np.float32),f


def make_cell(k):
    # Differentiable even for repeated stretch eigenvalues.
    a,b,c,x,y,z=k.unbind(-1);s2=math.sqrt(2)
    rows=(a,x/s2,y/s2,x/s2,b,z/s2,y/s2,z/s2,c)
    return torch.matrix_exp(torch.stack(rows,dim=-1).reshape(-1,3,3))


def new_model(arm):
    if arm in ("angular","power0","radial"):
        from lattice_symmetry.v2.physical import Generator
        return Generator(**GENERATOR)
    if arm in ("fixed-angular","flow16"):return SetNetwork(width=24)
    raise ValueError(arm)


def fixed(model,z,k,f):
    # Residual in the native fixed log-stretch chart, with the physical prior as source.
    dk,df=model(z,k,f,k.new_zeros(len(k),1))
    return k+.4*dk.tanh(),(f+.5*df.tanh()).remainder(1)*(z>0).unsqueeze(-1)


def geometry(model,arm,z,k,f):
    if arm in ("angular","power0","radial"):return model(make_cell(k),f,z)
    if arm=="fixed-angular":
        out,f=fixed(model,z,k,f)
        return make_cell(out),f
    for i in range(16):
        t=k.new_full((len(k),1),i/16)
        dk,df=model(z,k,f,t)
        k=k+dk/16
        f=(f+df/(16*math.sqrt(12))).remainder(1)*(z>0).unsqueeze(-1)
    return make_cell(k),f


def flow_loss(model,z,k0,f0,kt,ft,g):
    mask=(z>0).unsqueeze(-1)
    perm=(z.masked_fill(z==0,120)+torch.rand(z.shape,generator=g,device=z.device)*.5).argsort(1)
    ft=ft.gather(1,perm[:,:,None].expand_as(ft))
    df=(ft-f0+.5).remainder(1)-.5
    df=(df-mean_sites(df,mask)[:,None])*mask
    t=torch.rand((len(z),1),generator=g,device=z.device)
    dk,dx=model(z,k0+t*(kt-k0),(f0+t[:,:,None]*df).remainder(1),t)
    return (dk-(kt-k0)).square().mean()+((dx-math.sqrt(12)*df).square()*mask).sum()/(3*mask.sum())


def checkpoint_path(arm,seed):
    if arm=="prior":return HERE/"prior.npz"
    choice=json.loads((HERE/"selection.json").read_text())[arm]
    return HERE/"fits"/f"{arm}-{seed}"/f"update-{choice['updates']:05}.pt"


def service_from(model,arm,prior,device):
    @torch.inference_mode()
    def propose(z,rng):
        z=np.asarray(z,dtype=np.int64)
        k,f=prior.draw(z,rng)
        tensors=[torch.as_tensor(a,device=device) for a in (z,k,f)]
        L,frac=geometry(model,arm,*tensors)
        cell=L.detach().cpu().double().numpy()
        return pack_gram(cell@cell.transpose(0,2,1)),frac.detach().cpu().numpy()
    return propose


def load_service(arm,seed=11,device="cpu"):
    path=checkpoint_path(arm,seed)
    if arm=="prior":return Prior.load(path).draw
    data=torch.load(path,map_location=device,weights_only=True)
    if data["prior_sha256"]!=digest(HERE/"prior.npz"):raise ValueError("Mismatched physical prior")
    model=new_model(arm).to(device).eval()
    model.load_state_dict(data["model"],strict=True)
    return service_from(model,arm,Prior.load(HERE/"prior.npz"),device)


if __name__=="__main__":
    with np.load(ROOT/"scaled/mp20-data/train.npz") as train:
        prior=Prior.fit(train)
        prior.save(HERE/"prior.npz")
        k,f=prior.draw(train["z"],np.random.default_rng(3104))
        actual=train["k"];z=train["z"];n=(z>0).sum(1)
        def summaries(kk):
            ev=np.linalg.eigvalsh(unpack_gram(kk));v=np.exp(kk[:,:3].sum(1))/n
            return dict(aspect=np.quantile(np.sqrt(ev[:,-1]/ev[:,0]),[0,.005,.5,.995,1]).tolist(),
                        volume_per_site=np.quantile(v,[0,.005,.5,.995,1]).tolist())
        write_json(HERE/"prior.json",dict(train_rows=len(n),unique_exact_compositions=len(np.unique(z,axis=0)),
            train_sha256=digest(ROOT/"scaled/mp20-data/train.npz"),prior_sha256=digest(HERE/"prior.npz"),
            quantiles=[0,.005,.5,.995,1],train=summaries(actual),sample=summaries(k)))
