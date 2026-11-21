"""Supervised formula-only ranking on the same twenty train-vocabulary candidates."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
V3=HERE.parent
ROOT=V3.parents[1]
sys.path.insert(0,str(ROOT))
from lattice_symmetry.v3.models.sampler import Sampler,species_key,MENDELEEV,chart

def key(g,p,e):
    rank={int(z):j for j,z in enumerate(e)}
    return int(g),tuple(sorted((rank[int(z)],l) for z,l in p))

class Selector:
    def __init__(self,sampler):
        self.s=sampler
        self.freq=Counter()
        self.form=Counter()
        for i,(g,p,e) in enumerate(sampler.rich):
            sig=species_key(sampler.allz[sampler.rows[i]["index"]])[0]
            k=(sig,key(g,p,e))
            self.freq[k]+=1
            self.form[k,sampler.red[i]]+=1
        self.model=nn.Sequential(nn.Linear(29,48),nn.SiLU(),nn.Linear(48,1))
    def describe(self,z,training=False):
        z=np.sort(z[z>0]);sig,e,red=species_key(z)
        options=self.s.shortlist(tuple(z),training)
        features=np.zeros((20,29),np.float32)
        for j,(g,p,d) in enumerate(options):
            donor=self.s.mid[d] if d else None
            k=(sig,key(g,p,e))
            count=self.freq[k]-(self.form[k,red] if training else 0)
            cz=len(z); cols=[np.log1p(count),1/(1+j),j/20,len(e)/8,cz/20,len(p)/20]
            if donor is None: cols.extend([1.,1.,1.])
            else:
                ce=self.s.rich[donor][2]
                cols.extend([np.mean(abs(MENDELEEV[ce]-MENDELEEV[e]))/100,(ce!=e).mean(),np.mean(abs(ce-e))/100])
            cols += [np.mean(e)/100,np.std(e)/100,np.min(e)/100,np.max(e)/100]
            system=np.searchsorted([2,15,74,142,167,194,230],g)
            cols += [float(i==system) for i in range(7)]
            free=np.array([np.linalg.matrix_rank(chart(g,l)[0][0]) for _,l in p])
            cols += [(free==i).mean() for i in range(4)]
            cols += [sum(l==letter for _,l in p)/len(p) for letter in "abcd"]
            cols += [max(sig)/cz]
            features[j]=cols
        return options,features
    def rank(self,z,method):
        opt,x=self.describe(z)
        if method=="lookup":return opt
        if method=="frequency":values=x[:len(opt),0]
        else:
            with torch.inference_mode(): values=self.model(torch.from_numpy(x[:len(opt)])).flatten().numpy()
        return [opt[i] for i in np.argsort(-values,kind="stable")]

def main():
    torch.set_num_threads(1);torch.manual_seed(11);s=Sampler()
    clf=Selector(s);base=HERE/"selector";base.mkdir(exist_ok=True)
    cache=base/"features.npz"
    if not cache.exists():
        x=[];y=[];n=[];t=perf_counter()
        for i,r in enumerate(s.rows):
            z=s.allz[r["index"]];opts,fe=clf.describe(z,True)
            ans=(r["group"],tuple(sorted((int(e),l) for e,l in r["template"])))
            ix=next((j for j,(g,p,d) in enumerate(opts) if (g,p)==ans),-1)
            x.append(fe);n.append(len(opts));y.append(ix)
        np.savez_compressed(cache,x=x,n=n,y=y,ids=s.ids,seconds=perf_counter()-t)
    with np.load(cache) as a:
        x=torch.from_numpy(a["x"])
        n=torch.from_numpy(a["n"]).long()
        y=torch.from_numpy(a["y"]).long()
    assert len(y)==len(s.rows)
    # The label is undefined outside the available shortlist. Select no test cases here.
    active=(y>=0).nonzero().flatten()
    opt=torch.optim.AdamW(clf.model.parameters(),lr=0.002)
    curve=[];t=perf_counter()
    for epoch in range(4):
        running=0.
        for ids in active[torch.randperm(len(active))].split(128):
            logits=clf.model(x[ids]).squeeze(-1).masked_fill(torch.arange(20)[None]>=n[ids,None],-1e9)
            loss=nn.functional.cross_entropy(logits,y[ids]);opt.zero_grad();loss.backward();opt.step()
            running+=float(loss.detach())*len(ids)
        curve.append(running/len(active))
    torch.save(clf.model.state_dict(),base/"weights.pt")
    out=dict(seed=11,epochs=4,eligible=len(y),reachable=len(active),train_loss=curve,seconds=perf_counter()-t,
        parameters=sum(p.numel() for p in clf.model.parameters()),description="29 scalar inputs per shortlist entry, one shared MLP call; separate refiner adds a second invocation")
    (base/"training.json").write_text(json.dumps(out,indent=2))
    print(out,flush=True)

if __name__=="__main__":main()
