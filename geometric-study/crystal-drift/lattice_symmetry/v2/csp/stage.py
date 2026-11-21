"""A fixed-noise overfit gate, then matched early validation on independent noise."""
import os
for v in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[v]="1"
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from model import ARMS,DESCRIPTOR,HERE,ROOT,SEEDS,Prior,digest,geometry,make_cell,new_model,save_npz,service_from,write_json
import sys
sys.path.insert(0,str(ROOT/"scaled"))
from evaluate import process
from experiment import select
from lattice_symmetry.v2.physical import features


def overfit():
    torch.manual_seed(11);prior=Prior.load(HERE/"prior.npz")
    with np.load(ROOT/"scaled/mp20-data/train.npz") as fp:
        iz=next(i for i,r in enumerate(fp["z"]) if 5<sum(r>0)<12 and len(set(r[r>0]))>=3)
        z=fp["z"][iz:iz+1];kt=fp["k"][iz:iz+1];ft=fp["f"][iz:iz+1];mid=str(fp["ids"][iz])
    k,f=prior.draw(z,np.random.default_rng(181))
    z=torch.from_numpy(z).long();k,f=map(torch.from_numpy,(k,f))
    sc=torch.tensor(np.load(HERE/"scaling.npz")["scale"])
    target=features(make_cell(torch.from_numpy(kt).float()),torch.from_numpy(ft).float(),z,**DESCRIPTOR)/sc
    model=new_model("angular");opt=torch.optim.Adam(model.parameters(),lr=.005)
    err=[];gnorm=[]
    for step in range(25):
        L,fg=geometry(model,"angular",z,k,f);loss=((features(L,fg,z,**DESCRIPTOR)/sc-target)**2).sum()
        opt.zero_grad();loss.backward()
        gnorm.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.0,error_if_nonfinite=True)));err.append(float(loss))
        if step<24:opt.step()
    assert err[-1]<err[0] and max(gnorm)>0
    write_json(HERE/"overfit.json",dict(id=mid,train_index=iz,squared_descriptor_loss=err,grad_norm=gnorm,
               optimizer="Adam, 24 updates, fixed seed and known training target; this checks differentiability only"))
    print("Fixed sample",mid,err[0],err[-1],flush=True)


def validate(arms,seeds,updates):
    with np.load(ROOT/"scaled/mp20-data/val.npz") as fp:
        idx=select(fp["ids"],128)
        ids,z,k,f=[fp[v][idx] for v in ("ids","z","k","f")]
    prior=Prior.load(HERE/"prior.npz")
    scales=np.array(json.loads((ROOT/"scaled/runs/mp20/score-scale.json").read_text())["scale"])
    obs=process(k,f,z)/scales
    for arm in arms:
        for seed in seeds:
            for u in updates:
                folder=HERE/"fits"/f"{arm}-{seed}";path=folder/f"update-{u:05}.pt"
                val=folder/f"val-{u:05}.npz"
                if arm=="prior":
                    if u!=0:continue
                    path=HERE/"prior.npz";folder.mkdir(exist_ok=True,parents=True)
                    propose=prior.draw
                else:
                    if not path.with_suffix(".json").is_file():continue
                    state=torch.load(path,map_location="cpu",weights_only=True)
                    model=new_model(arm).eval();model.load_state_dict(state["model"])
                    propose=service_from(model,arm,prior,"cpu")
                if val.is_file():continue
                rng=np.random.default_rng(np.random.SeedSequence([15931,seed]))
                kg=[];fg=[]
                za=np.repeat(z,5,axis=0)
                for lo in range(0,len(za),32):
                    kb,fb=propose(za[lo:lo+32],rng);kg.extend(kb);fg.extend(fb)
                kg,fg=np.array(kg),np.array(fg)
                geom=process(kg,fg,za).reshape(len(z),5,4)/scales
                es=np.linalg.norm(geom-obs[:,None],axis=-1).mean(1)-np.linalg.norm(geom[:,:,None]-geom[:,None,:],axis=-1).sum((1,2))/40
                assert np.isfinite(es).all()
                save_npz(val,ids=ids,z=z,k=kg,f=fg,energy=es,checkpoint_sha256=digest(path))
                print("Validation",arm,seed,u,round(float(es.mean()),3),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--overfit",action="store_true")
    p.add_argument("--arms",nargs="+",default=ARMS);p.add_argument("--seeds",nargs="+",type=int,default=SEEDS)
    p.add_argument("--updates",nargs="+",type=int,default=(16,64,256,1024,3392))
    a=p.parse_args();torch.set_num_threads(1)
    if a.overfit:overfit()
    else:validate(a.arms,a.seeds,a.updates)
