"""Generate formula-only uFlowCSP outputs with frozen weights, native prior, and stable ordered draws."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from runtime import HERE, ROOT, atomic_json, counts, model, propose, transport

SPLITS = {"dev":"val","confirm":"val","family":"train","shared":"train","test":"test"}
SEEDS = {"val":38471,"train":38473,"test":38479}

def panel_data(panel):
    splits = json.loads((HERE.parent/"protocol/splits.json").read_text())
    data_split = SPLITS[panel]
    ids=np.array(splits[panel+"_ids"])
    with np.load(ROOT/f"scaled/mp20-data/{data_split}.npz") as data:
        lookup={str(mid):i for i,mid in enumerate(data["ids"])}
        idx=np.array([lookup[mid] for mid in ids])
        z=data["z"][idx]
    return ids,idx,z

def predict_panel(tier, net, flow, panel, k, out, sha, batch, device):
    ids,idx,z=panel_data(panel)
    out.mkdir(parents=True,exist_ok=True)
    clock=time.perf_counter()
    for lo in range(0,len(ids),256):
        hi=min(lo+256,len(ids))
        path=out/f"{lo:05}-{hi:05}.npz"
        if path.exists():
            with np.load(path) as rec:
                assert rec["checkpoint_sha"]==sha and np.array_equal(rec["ids"],ids[lo:hi])
            continue
        zz=np.repeat(z[lo:hi],k,axis=0)
        noise=[]
        for ix in idx[lo:hi]:
            # Sample one row at a time, independently keyed by official split row.
            # Ordered draws are reproducible regardless of batch, shard, tier or checkpoint.
            seed=int(np.random.SeedSequence([SEEDS[SPLITS[panel]],int(ix)]).generate_state(1)[0])
            torch.manual_seed(seed)
            noise.extend(flow.sample_prior(1,100,12,"cpu")[0] for _ in range(k))
        noise=torch.stack(noise)
        noise_sha=hashlib.sha256(memoryview(noise.numpy()).cast('B')).hexdigest()
        L,F,Z=[],[],[]
        for b in range(0,len(zz),batch):
            cell,f,types=propose(net,flow,zz[b:b+batch],noise[b:b+batch],device=device)
            L.append(cell)
            F.append(f)
            Z.append(types)
        L=np.concatenate(L).reshape(hi-lo,k,3,3)
        F=np.concatenate(F).reshape(hi-lo,k,20,3)
        Z=np.concatenate(Z).reshape(hi-lo,k,20)
        np.testing.assert_array_equal(Z,np.broadcast_to(z[lo:hi,None,:],Z.shape))
        tmp=path.with_suffix(".part.npz")
        np.savez_compressed(tmp,ids=ids[lo:hi],L=L,F=F,Z=z[lo:hi],native_indices=idx[lo:hi],
                            checkpoint_sha=sha,noise_sha256=noise_sha,nfe=0 if net is None else 1)
        tmp.replace(path)
        if lo==0 or hi==len(ids) or lo%2048==0:
            print(tier,panel,hi,len(ids),k,round(time.perf_counter()-clock,1),"s",flush=True)
    atomic_json(out/"ready.json",dict(tier=tier,panel=panel,count=len(ids),draws=k,sha256=sha,
        parameters=counts(net) if net is not None else 0,nfe=0 if net is None else 1,tokens=100,
        seconds_this_invocation=time.perf_counter()-clock,device=str(device),batch=batch,split=SPLITS[panel]))

def main(args):
    torch.set_num_threads(1)
    flow=transport()
    net=model(args.tier,args.device).eval() if args.tier!="prior" else None
    tag=(f"{args.tier}-{args.seed}"+("-long" if getattr(args,"extended",False) else "")
         +("-common" if getattr(args,"common",False) else ""))
    fit=HERE/"fits"/tag
    if args.mode=="dev":
        panels=["dev"]
        steps=sorted(fit.glob("ema-*.pt")) if args.tier not in ("released","prior") else [None]
        if args.tier not in ("released","prior"):
            assert (fit/"complete.json").is_file() and len(steps)==2
    else:
        panels=args.panels
        if args.tier not in ("released","prior"):
            selected=json.loads((fit/"selection.json").read_text())
            steps=[(fit/selected["checkpoint"]).resolve()]
            assert hashlib.sha256(steps[0].read_bytes()).hexdigest()==selected["checkpoint_sha"]
        else:
            steps=[None]
    for checkpoint in steps:
        if checkpoint:
            net.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True))
            sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        else:
            sha="identity" if args.tier=="prior" else "e1854363d9e78044d1202879e40a918ab50831cb1e08eb686cb53aeacaa85673"
        for panel in panels:
            suffix=checkpoint.stem if (args.mode=="dev" and checkpoint) else "frozen"
            target=HERE/"predictions"/tag/suffix/panel
            predict_panel(args.tier,net,flow,panel,5 if panel in ("dev","confirm") else 20,
                          target,sha,args.batch,args.device)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("tier",choices=("small","medium","released","prior"))
    p.add_argument("mode",choices=("dev","frozen"))
    p.add_argument("--panels",nargs="+",choices=SPLITS,default=["confirm","family","shared","test"])
    p.add_argument("--seed",type=int,default=11)
    p.add_argument("--device",default="cuda")
    p.add_argument("--batch",type=int,default=64)
    p.add_argument("--extended",action="store_true")
    p.add_argument("--common",action="store_true")
    main(p.parse_args())
