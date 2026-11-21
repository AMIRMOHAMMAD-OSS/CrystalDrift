"""One neural evaluation on occupied-orbit tokens with analytic site expansion."""
import numpy as np
import torch
from torch import nn
from lattice_symmetry.v2.neighbors import edges
from lattice_symmetry.v2.physical import Generator as ReleasedGraph

# Each width change preserves the geometric computation and graph inputs.
TIERS={'tiny':(17,1),'small':(114,3),'medium':(319,4),'large':(704,6),'reference':(1736,9)}


class ExpandedSiteGraph(ReleasedGraph):
    def forward(self,cell,frac,z,*unused):
        return super().forward(cell,frac,z)


def make_model(architecture):
    if 'published_width' in architecture:
        return ExpandedSiteGraph(width=architecture['published_width'])
    return OrbitNet(**architecture)


def pack(seeds,device='cpu',dtype=torch.float32):
    n=max(len(q.Z) for q in seeds);b=len(seeds)
    frac=np.zeros((b,n,3));z=np.zeros((b,n),int);o=z.copy();mult=z.copy();free=z.copy()
    j=np.zeros((b,n,3,3));p=j.copy()
    for ix,s in enumerate(seeds):
        k=len(s.Z);z[ix,:k]=s.Z;frac[ix,:k]=s.F;o[ix,:k]=s.orbit;j[ix,:k]=s.J
        D=s.J@s.L
        for a in np.unique(s.orbit):
            mask=s.orbit==a;T=D[mask]
            H=np.einsum('sij,skj->ik',T,T)
            pinv=np.linalg.pinv(H,hermitian=True,rcond=1e-10)
            p[ix,:k][mask]=T.transpose(0,2,1)@pinv
            mult[ix,:k][mask]=mask.sum();free[ix,:k][mask]=np.linalg.matrix_rank(H,tol=1e-8)
    return tuple(torch.as_tensor(a,dtype=kind,device=device) for a,kind in (
        (np.stack([q.L for q in seeds]),dtype),(frac,dtype),(z,torch.long),(o,torch.long),
        (mult,dtype),(free,dtype),(j,dtype),(p,dtype)))


def aggregate(x,idx,total):
    return x.new_zeros((total,*x.shape[1:])).index_add(0,idx,x)


class OrbitNet(nn.Module):
    def __init__(self,width=120,depth=3,global_context=True):
        super().__init__()
        w=width;d=12
        self.global_context=global_context
        self.element=nn.Embedding(119,w,padding_idx=0)
        self.chemistry=nn.Embedding(119,d,padding_idx=0)
        self.encoder=nn.Linear(2*d+11,w)
        self.layers=nn.ModuleList([nn.Sequential(nn.LayerNorm(w),nn.Linear(w,w*2),nn.SiLU(),
                                                nn.Linear(w*2,w)) for _ in range(depth)])
        self.mix=nn.ModuleList([nn.Linear(w,w,bias=False) for _ in range(depth)]) if global_context else nn.ModuleList()
        self.project=nn.Linear(w,d)
        self.edge=nn.Sequential(nn.Linear(2*d+5,32),nn.SiLU(),nn.Linear(32,2))
        self.volume=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,1))
        for h in (self.edge[-1],self.volume[-1]):
            nn.init.normal_(h.weight,std=.01);nn.init.zeros_(h.bias)
        self.register_buffer('centers',torch.linspace(0,1,5))

    def forward(self,L,F,Z,O,mult,free,J,P,r=None,t=None,residual=False,return_pool=False):
        b,n=Z.shape;bn=b*n
        mask=(Z>0);live=mask.to(F.dtype);count=live.sum(1)
        if t is None:t=F.new_ones(b)
        if r is None:r=F.new_zeros(b)
        src,dst,bi,vec=edges(L,F,Z,4.)
        d2=vec.square().sum(-1)/16
        phi=torch.exp(-5*(d2[:,None]-self.centers)**2)
        envelope=(1-d2).clamp_min(0).pow(3)
        w=envelope/(1+aggregate(envelope,src,bn)[src])
        z=Z.flatten(); e=self.chemistry(z)
        msg=(e[dst]*w[:,None])
        nbr=aggregate(msg,src,bn)
        geom=aggregate(w[:,None]*phi,src,bn)
        logv=torch.linalg.slogdet(L).logabsdet-count.log()
        scalars=torch.stack([mult/20,free/3,
            logv[:,None].expand(b,n)/4,torch.log1p(count)[:,None].expand(b,n)/3,
            r[:,None].expand(b,n),t[:,None].expand(b,n)],-1).reshape(bn,6)
        hidden=self.element(z)+self.encoder(torch.cat([e,nbr,geom,scalars],-1))
        off=torch.arange(b,device=Z.device)[:,None]*n
        ids=(O+off).flatten();active=live.flatten()[:,None]
        # Average equivalent sites into one token per occupied orbit; no orbit ordering feature.
        hidden=aggregate(hidden*active,ids,bn)/aggregate(active,ids,bn).clamp_min(1)
        for i,layer in enumerate(self.layers):
            if self.global_context:
                pooled=(hidden[ids].reshape(b,n,-1)*live[:,:,None]).sum(1)/count[:,None]
                x=hidden+self.mix[i](pooled)[:,None,:].expand(b,n,-1).reshape(bn,-1)
            else:x=hidden
            hidden=hidden+layer(x)
        site=hidden[ids]
        code=self.project(site)
        edge=self.edge(torch.cat([code[src],code[dst],phi],-1)).tanh()*w[:,None]
        v=aggregate(edge[:,0:1]*vec,src,bn)
        # Least-squares free-parameter displacement from all symmetry images of an orbit.
        du=aggregate(torch.bmm(v[:,None,:],P.reshape(bn,3,3)).squeeze(1)*active,ids,bn)
        df=torch.bmm(du[ids,None,:],J.reshape(bn,3,3)).reshape(b,n,3)*live[:,:,None]
        B=aggregate(edge[:,1,None,None]*(vec[:,:,None]*vec[:,None,:]/16),bi,b)/count[:,None,None]
        I=torch.eye(3,device=F.device,dtype=F.dtype).expand(b,3,3)
        A=I+.3*B
        shape=A*torch.exp(-torch.linalg.slogdet(A).logabsdet[:,None,None]/3)
        pool=(site.reshape(b,n,-1)*live[:,:,None]).sum(1)/count[:,None]
        dlogv=.6*self.volume(pool).tanh()
        newL=L@(shape*torch.exp(dlogv/3)[:,:,None])
        output=(newL-L,df) if residual else (newL,(F+df).remainder(1)*live[:,:,None])
        return (*output,pool) if return_pool else output

