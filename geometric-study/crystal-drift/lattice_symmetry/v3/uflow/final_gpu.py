"""Finish the frozen predictions and released-EMA replay in a single GPU session."""
import gc
import runpy
import time
from argparse import Namespace
import torch
from runtime import HERE,atomic_json
from verify_native import released_check
from generate import main as forecast

def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    atomic_json(HERE/"released-replay.json",released_check("cuda"))
    gc.collect();torch.cuda.empty_cache()
    target=HERE.parent/"denovo/pools/released-uflow.npz"
    if not target.is_file():
        runpy.run_path(str(HERE.parent/"denovo/pool_uflow.py"),run_name="__main__")
        gc.collect();torch.cuda.empty_cache()
    for mode,seed in (("long",11),("common",11),("primary",23),("primary",37)):
        for tier in ("small","medium"):
            tag=f"{tier}-{seed}"+("-"+mode if mode!="primary" else "")
            wait_until=time.monotonic()+180
            while not (HERE/"fits"/tag/"selection.json").is_file():
                if time.monotonic()>wait_until:
                    raise RuntimeError(f"Missing locked development selection: {tag}")
                time.sleep(2)
            forecast(Namespace(tier=tier,device="cuda",batch=64,seed=seed,extended=mode=="long",
                               common=mode=="common",mode="frozen",
                               panels=["confirm","family","shared","test"] if mode=="long" else ["confirm"]))
            gc.collect();torch.cuda.empty_cache()
if __name__=="__main__":main()
