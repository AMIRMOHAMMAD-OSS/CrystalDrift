"""Run and sample each registered sustained-exposure comparison in turn."""
import gc
from argparse import Namespace
import torch
from runtime import HERE
from train import main as fit
from generate import main as forecast

if __name__=="__main__":
    for tier in ("small","medium"):
        if not (HERE/f"fits/{tier}-11-long/complete.json").exists():
            fit(Namespace(tier=tier,device="cuda",batch=32,seed=11,lr=.0003,smoke=0,passes=8))
        gc.collect()
        torch.cuda.empty_cache()
        forecast(Namespace(tier=tier,device="cuda",batch=64,seed=11,mode="dev",extended=True,panels=["dev"]))
        gc.collect()
        torch.cuda.empty_cache()
