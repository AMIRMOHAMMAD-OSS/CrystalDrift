"""Run architecture controls and trained larger weights on common held-out targets."""
import argparse
import json
import sys
import os
from pathlib import Path
import subprocess
import time
from register import freeze,model,HERE,V3
from wave import run

def main():
    p=argparse.ArgumentParser();p.add_argument("--matcher",required=True);p.add_argument("--workers",type=int,default=2)
    p.add_argument("--aux-only",action="store_true",help="Evaluate the seven-system training-only head after its checkpoint freezes")
    a=p.parse_args()
    items=[("sitegraph-tiny","tiny-angular-laplace-11-published"),
           ("sitegraph-small","small-angular-laplace-11-published"),("angular-small-local","small-angular-laplace-11-local"),
           ("radial4-small","small-radial4-laplace-11"),("angular-large-11","large-angular-laplace-11")]
    aux=("angular-small-systemaux","small-angular-laplace-11-systemaux")
    items=[aux] if a.aux_only else items+[aux]
    done=[]
    while items or (not a.aux_only and not (V3/"models/window-complete.json").exists()):
        for tag,folder in items.copy():
            if not (V3/"models/fits"/folder/"selection.json").exists():continue
            row=model(tag,folder);reg=freeze("weight-"+tag,[row]);panels=["family","shared","test"]
            run(reg,panels,"cpu")
            for panel in panels:
                subprocess.run([a.matcher,str(HERE/"farm.py"),panel,str(reg),"--workers",str(a.workers),"--no-wait"],check=True)
            done.append(row);items.remove((tag,folder))
        if items or (not a.aux_only and not (V3/"models/window-complete.json").exists()):time.sleep(15)
    if a.aux_only:return
    for f in (V3/"models/fits").glob("reference-*-11/selection.json"):
        row=model("reference-scale-11",f.parent.name);reg=freeze("weight-reference-scale-11",[row]);panels=["family","shared","test"]
        run(reg,panels,"cuda")
        for panel in panels:
            subprocess.run([a.matcher,str(HERE/"farm.py"),panel,str(reg),"--workers",str(a.workers),"--no-wait"],check=True)
        done.append(row)
    freeze("scale-frozen",done)
if __name__=="__main__":main()
