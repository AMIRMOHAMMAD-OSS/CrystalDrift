"""Use a single CUDA process for a bounded ordered queue. Every completed workload is resumable."""
import argparse
import gc
import json
import time
from argparse import Namespace
import torch
from runtime import HERE
from train import main as fit
from generate import main as forecast

def work(name):
    kind,tier,seedstr,mode=name.split("/")
    seed=int(seedstr)
    common=mode=="common"
    long=mode=="long"
    tag=f"{tier}-{seed}"+("-common" if common else "-long" if long else "")
    path=HERE/"fits"/tag
    if kind=="fit":
        if not (path/"complete.json").exists():
            fit(Namespace(tier=tier,device="cuda",batch=32,seed=seed,lr=.0003,smoke=0,
                          passes=8 if long else 1,common=common))
        forecast(Namespace(tier=tier,seed=seed,mode="dev",extended=long,common=common,
                           device="cuda",batch=64,panels=["dev"]))
    else:
        panels=["confirm"] if kind=="confirm" else ["confirm","family","shared","test"]
        forecast(Namespace(tier=tier,seed=seed,mode="frozen",extended=long,common=common,
                           device="cuda",batch=64,panels=panels))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("jobs",nargs="+")
    args=p.parse_args()
    for job in args.jobs:
        print("window begins",job,flush=True)
        work(job)
        gc.collect();torch.cuda.empty_cache()
        print("window complete",job,flush=True)
