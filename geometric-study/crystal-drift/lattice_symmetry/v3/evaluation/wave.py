"""Produce registered ordered proposals as development selections arrive on the shared GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from finalists import specs
from register import HERE,V3,freeze,model
from storage import pool
from reuse import equivalent

def run(matrix,panels,device):
    items=json.loads(matrix.read_text())["methods"]
    for panel in panels:
        equivalent(panel,items)
        if all((pool(panel,v["tag"])/"settings.json").is_file() for v in items):continue
        subprocess.run([sys.executable,str(HERE/"paired.py"),str(matrix),"--panel",panel,
            "--draws","20","--inner","32","--device",device],check=True,cwd=V3.parents[1])

def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cpu");a=p.parse_args()
    raw=Path(os.environ.get("CRYTALDRIFT_RAW",HERE/"raw"))
    while not (pool("test","empirical-rank")/"settings.json").is_file():time.sleep(10)
    run(HERE/"same-exposure-frozen.json",["family","shared","test"],a.device)
    groups=("strict","replicated","long","kernels")
    staged=set()
    while True:
        for group in groups:
            for tag,folder in specs(group):
                if tag in staged:continue
                if not (V3/"models/fits"/folder/"selection.json").is_file():continue
                matrix=freeze("weight-"+tag,[model(tag,folder)])
                panels=["family","shared","test"] if group!="kernels" else ["confirm","family","shared"]
                run(matrix,panels,a.device);staged.add(tag)
        if len(staged)==sum(len(specs(g)) for g in groups):break
        time.sleep(10)
    for group in groups:
        freeze(group+"-frozen",[model(tag,folder) for tag,folder in specs(group)])
    print("All registered follow-up proposals sealed.",flush=True)

if __name__=="__main__":main()
