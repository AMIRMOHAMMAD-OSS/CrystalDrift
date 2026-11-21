"""Compare the fast CSPNet replay with an uncached dense-graph forward in FP32."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from crystalflow import ROOT,HERE,EncoderCache,load_model,sample,pack_gram

@torch.inference_mode()
def check(a):
    torch.set_num_threads(1)
    model,embed,decode=load_model(a.source,a.weights,"cpu")
    with np.load(ROOT/"revision/full/mp20/cohort.npz") as x:z=x["z"][:4]
    num=torch.tensor((z>0).sum(1));graph=torch.arange(len(z)).repeat_interleave(num)
    types=torch.tensor(z[z>0],dtype=torch.long)
    torch.manual_seed(910562)
    l=torch.randn(len(z),6)*.1;l[:,-1]+=1
    coords=torch.rand(len(types),3)
    for j in range(1,6):
        times=embed(torch.full((len(z),),j/5))
        dl,dx=model(times,types,coords,l,num,graph,lattices_mat=decode(l))
        l+=dl/5;coords=(coords+dx/5)%1.
    cart=decode(l).double().numpy()
    ref=pack_gram(cart@cart.swapaxes(-1,-2))
    k,f=sample(model,embed,decode,EncoderCache(model),z,5,"cpu",910562,True)
    error=float(np.max(np.abs(k-ref)))
    delta=(f[z>0]-coords.numpy()+.5)%1-.5
    assert abs(delta).max()<1e-6 and error<2e-4
    summary=dict(cases=4,ordered_ids="first four full MP-20 hash-sorted targets",
        inputs="Same seed, fixed IEEE FP32 CPU, five Euler steps", max_torus_error=float(np.abs(delta).max()),
        max_analytic_lattice_chart_error=error,source_commit="9c25ff0245d787efd87c9d1d797a8968443608cc")
    (HERE/"decoder-parity.json").write_text(json.dumps(summary,indent=2)+"\n");print(summary)
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",type=Path,required=True)
    p.add_argument("--weights",type=Path,required=True);check(p.parse_args())
