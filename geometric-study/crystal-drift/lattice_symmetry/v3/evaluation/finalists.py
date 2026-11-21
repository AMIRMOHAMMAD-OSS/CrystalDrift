"""Freeze and evaluate the prespecified replication and exposure panels as weights become available."""
import argparse
import json
from pathlib import Path
import sys
import subprocess
import time
from register import HERE,V3,digest,model,freeze

def specs(group):
    if group=="replicated":
        return [(f"{arm}-{tier}-{s}",f"{tier}-{arm}-laplace-{s}") for s in (23,37) for tier in ("small","medium") for arm in ("angular","contacts")]
    if group=="long":
        return [(f"{arm}-{tier}-long",f"{tier}-{arm}-laplace-11"+("-commonorder" if arm=="meanflow" else "")+"-8pass")
            for tier,arm in (("small","angular"),("medium","angular"),("medium","contacts"),("medium","meanflow"))]
    if group=="strict":
        return [(f"meanflow-{tier}-strict",f"{tier}-meanflow-laplace-11-commonorder") for tier in ("small","medium")]
    if group=="kernels":
        return [(f"{arm}-small-11",f"small-{arm}-laplace-11") for arm in ("pair","center0","variance")]+[
             (f"{arm}-small-gaussian",f"small-{arm}-gaussian-11") for arm in ("angular","contacts")]
    raise ValueError(group)

def external(tier):
    path=V3/"uflow/fits"/(tier+"-11-long");s=json.loads((path/"selection.json").read_text())
    weights=path/s["checkpoint"];assert digest(weights)==s["checkpoint_sha"]
    return dict(tag=f"uflow-{tier}-11-long",u_tier=tier,checkpoint=weights.relative_to(V3).as_posix(),sha=s["checkpoint_sha"])

def clock_matrix():
    return freeze("clock-frozen",[dict(tag="prior"),dict(tag="geometry-donor",family="donor"),model("angular-small-long","small-angular-laplace-11-8pass"),
        model("angular-medium-long","medium-angular-laplace-11-8pass"),model("contacts-medium-long","medium-contacts-laplace-11-8pass"),
        model("meanflow-medium-long","medium-meanflow-laplace-11-commonorder-8pass"),
        external("small"),external("medium"),dict(tag="released-uflow",u_tier="released")])

def main():
    p=argparse.ArgumentParser();p.add_argument("group",choices=("replicated","long","strict","kernels","clock"))
    p.add_argument("--wait",action="store_true");p.add_argument("--panels",nargs="+");p.add_argument("--device",default="cpu")
    a=p.parse_args()
    if a.group=="clock":clock_matrix();return
    pending=specs(a.group);done=[]
    while pending:
        for tag,folder in list(pending):
            ready=V3/"models/fits"/folder/"selection.json"
            if not ready.exists():continue
            spec=model(tag,folder);path=freeze("weight-"+tag,[spec]);done.append(spec);pending.remove((tag,folder))
            if a.panels:
                subprocess.run([sys.executable,str(HERE/"series.py"),str(path),"--panels",*a.panels,"--mode","predict","--device",a.device],check=True,cwd=V3.parents[1])
        if pending:
            if not a.wait:raise RuntimeError("Missing frozen choices "+str(pending))
            time.sleep(12)
    freeze(a.group+"-frozen",done if a.panels is None else [model(tag,folder) for tag,folder in specs(a.group)])

if __name__=="__main__":main()
