"""Score completed follow-up arms, sharing target contexts and preserving failures."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from finalists import specs
from storage import HERE,pool

def main():
    p=argparse.ArgumentParser();p.add_argument("--workers",default=3,type=int);a=p.parse_args()
    tasks=[("same-exposure-frozen",["family","shared","test"])]
    for group in ("strict","replicated","long","kernels"):
        for tag,_ in specs(group):
            tasks.append(("weight-"+tag,["confirm","family","shared"] if group=="kernels" else ["family","shared","test"]))
    tasks+= [("baseline-frozen",["family","shared"])]
    while tasks:
        for index,(key,panels) in enumerate(tasks.copy()):
            file=HERE/(key+".json")
            if not file.exists():continue
            import json
            tags=[v["tag"] for v in json.loads(file.read_text())["methods"]]
            for panel in list(panels):
                if not all((pool(panel,t)/"settings.json").exists() for t in tags):continue
                subprocess.run([sys.executable,str(HERE/"farm.py"),panel,str(file),"--workers",str(a.workers),"--no-wait"],check=True)
                assert all((HERE/"scores"/panel/t/"outcomes.npz").is_file() for t in tags)
                panels.remove(panel)
            if not panels:tasks.remove((key,panels))
        if tasks:time.sleep(10)

if __name__=="__main__":main()
