"""Fit the exact-formula binary generator with a single fingerprint change."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
STUDY=HERE.parents[1]
sys.path.insert(0,str(STUDY))
import model
import train


def fingerprint(k,f,z,arm):
    base=model.crystal_features(k,f,z,"multi")
    if arm=="pair":
        return base[:,:58]
    if arm=="original":
        return base
    if arm=="reweight":
        # Training-only mean added squared distance matched in checks.py.
        return torch.cat((base,1.9871711730957031*base[:,6:7]),1)
    if arm!="harmonic":
        raise ValueError(arm)
    _,a,b=model.species_masks(z)
    angle=2*math.pi*f[:,:,0]
    A=model.masked_mean(torch.stack((angle.cos(),angle.sin()),-1),a)
    B=model.masked_mean(torch.stack(((2*angle).cos(),(2*angle).sin()),-1),b)
    real=(A[:,0].square()-A[:,1].square())*B[:,0]+2*A[:,0]*A[:,1]*B[:,1]
    return torch.cat((base,real[:,None]/math.sqrt(6)),1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(arm,seed):
    out=HERE/"runs"/arm
    out.mkdir(exist_ok=True,parents=True)
    def extract(k,f,z,rep):
        assert rep=="multi"
        return fingerprint(k,f,z,arm)
    # Inject only the descriptor into the shared training implementation.
    # Its RNG, data indexing, model initialization and optimizer are untouched.
    with patch.object(train,"crystal_features",extract):
        net,mean,scale,elapsed,history=train.train(STUDY/"data","multi-exact",seed,1200,64)
    checkpoint=out/f"seed{seed}.pt"
    torch.save(dict(model=net.state_dict(),mean=mean,scale=scale,method=arm,seed=seed),checkpoint)
    result=dict(arm=arm,seed=seed,updates=1200,batch=64,formula_group_size=4,train_seconds=elapsed,
                torch=torch.__version__,numpy=np.__version__,threads=torch.get_num_threads(),
                checkpoint_sha256=sha(checkpoint),shared_train_sha256=sha(STUDY/"train.py"),
                shared_model_sha256=sha(STUDY/"model.py"),data_sha256=sha(STUDY/"data"/"multi-train.npz"),
                history=history)
    if arm=="original":
        original=torch.load(STUDY/"drift-results"/"multi-exact"/f"seed{seed}.pt",weights_only=True)
        result["max_abs_parameter_difference_from_released_checkpoint"]=max(
            (v-original["model"][n]).abs().max().item() for n,v in net.state_dict().items())
    # Preserve the first five draws of the original 20-draw evaluation RNG stream.
    for split in ("val","test"):
        with np.load(STUDY/"data"/f"multi-{split}.npz") as d:
            z,ids=d["z"],d["ids"]
        x,site=train.generate(net,train.evaluation_source(z,20,seed),mean,scale)
        k=x.numpy().reshape(len(z),20,6)[:,:5]
        f=site.numpy().reshape(len(z),20,8,3)[:,:5]
        outpath=out/f"seed{seed}-{split}.npz"
        np.savez_compressed(outpath,ids=ids,z=z,k=k,f=f)
        result[split+"_sha256"]=sha(outpath)
    (out/f"seed{seed}.json").write_text(json.dumps(result,indent=1)+"\n",encoding="utf8")
    print(arm,seed,"complete in",round(elapsed,1),"seconds",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms",nargs="+",default=["original","pair","harmonic","reweight"],choices=["pair","original","harmonic","reweight"])
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,37])
    a=p.parse_args();torch.set_num_threads(1)
    for seed in a.seeds:
        for arm in a.arms:
            fit(arm,seed)
