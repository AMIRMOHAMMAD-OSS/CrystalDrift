"""Run the two predeclared extra training seeds with the same exposure and development looks."""
import gc
from argparse import Namespace
import torch
from runtime import HERE
from train import main as fit
from generate import main as forecast

if __name__=="__main__":
    for seed in (23,37):
        for tier in ("small","medium"):
            if not (HERE/f"fits/{tier}-{seed}/complete.json").exists():
                fit(Namespace(tier=tier,device="cuda",batch=32,seed=seed,lr=.0003,smoke=0))
            gc.collect()
            torch.cuda.empty_cache()
            forecast(Namespace(tier=tier,device="cuda",batch=64,seed=seed,mode="dev",panels=["dev"]))
            gc.collect()
            torch.cuda.empty_cache()
