"""Finite-difference validation of the signed MeanFlow identity and the actual network JVP."""
import json
import numpy as np
import torch
from .network import OrbitNet
from .train import CACHE,HERE,tensors

def main():
    values={}
    with np.load(CACHE/'train-000.npz') as q:
        at=np.array([0,1,4,5])
        args=tensors(q,at,'cpu')
        dL=torch.tensor(q['targetL'][at]);dF=torch.tensor(q['targetF'][at]);row=q['indices'][[0,1]].tolist()
    for dtype,h,tol in ((torch.float64,1e-5,1e-6),(torch.float32,.003,.001)):
        torch.manual_seed(5518)
        model=OrbitNet(32,1).to(dtype)
        L,F,Z,O,m,free,J,P=[q.to(dtype) if q.is_floating_point() else q for q in args]
        gl=dL.to(dtype);gf=dF.to(dtype)
        t=torch.tensor([.31,.45,.56,.78],dtype=dtype);r=t*.3
        stateL=L+(1-t[:,None,None])*gl;stateF=F+(1-t[:,None,None])*gf
        def u(a,f,t):
            return model(a,f,Z,O,m,free,J,P,r,t,residual=True)
        _,jac=torch.autograd.functional.jvp(u,(stateL,stateF,t),(-gl,-gf,torch.ones_like(t)))
        plus=u(stateL-h*gl,stateF-h*gf,t+h);minus=u(stateL+h*gl,stateF+h*gf,t-h)
        errors=[float((j-(p-n)/(2*h)).abs().max().detach()) for j,p,n in zip(jac,plus,minus)]
        assert max(errors)<tol,(dtype,errors)
        # Analytic nonconstant velocity v(t)=2t, signed displacement = -u=-(r+t).
        z=lambda h: h*h
        d=-(z(t)-z(r))/(t-r);der=-torch.ones_like(t)
        target=-2*t-(t-r)*der
        assert torch.allclose(target,d)
        values[str(dtype)]=dict(step=h,tolerance=tol,max_abs_cell_jvp=errors[0],max_abs_site_jvp=errors[1],
            analytic_sign=float(abs(target-d).max()))
    (HERE/'meanflow-jvp.json').write_text(json.dumps(dict(training_shard=0,indices=row,checks=values),indent=2))
    print(values,flush=True)

if __name__=='__main__':main()

