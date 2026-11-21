"""Freeze an independent Alexandria cohort and generate the prospective hybrid controls."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
for key in ("MKL_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="1"
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"scaled"))
from experiment import load,predict
from evaluate import Retrieval
from prepare_alex import alias_group,reduced


def cohort():
    val=load(ROOT/"scaled/alex-data/val.npz")
    protected_ids=set();protected_formulas=set()
    for split in ("train","val","test"):
        data=load(ROOT/f"scaled/mp20-data/{split}.npz")
        protected_ids.update(data["ids"].tolist())
        protected_formulas.update(reduced(row) for row in np.unique(data["z"],axis=0))
    old=load(ROOT/"scaled/alex-validation-transfer/val.npz")
    protected_ids.update(old["ids"].tolist())
    protected_formulas.update(reduced(row) for row in old["z"])
    salt="hybrid-confirm-20261120"
    ordered=sorted(range(len(val["ids"])),key=lambda i:hashlib.sha256((salt+val["ids"][i]).encode()).digest())
    picked=[];used=set();cache={}
    for i in ordered:
        mid=val["ids"][i]
        if alias_group(mid)!="alex" or mid in protected_ids:continue
        if any(a.lower() in protected_ids for a in re.findall(r"(?:mp|mvc)-\d+",mid,flags=re.I)):continue
        z=val["z"][i];key=z.tobytes()
        if key not in cache:cache[key]=reduced(z)
        comp=cache[key]
        if comp in protected_formulas or comp in used:continue
        used.add(comp);picked.append(i)
        if len(picked)==512:break
    assert len(picked)==len(used)==512
    part={k:v[picked] for k,v in val.items()}
    np.savez_compressed(HERE/"cohort.npz",**part,validation_index=picked)
    (HERE/"audit.json").write_text(json.dumps(dict(n=512,salt=salt,excluded_old_alex=128,
        mp20_all_known_formulas=len(protected_formulas),selected_formula_overlap=0,selected_id_overlap=0,
        prior_inspection="MP-20 official test and a separate 128-record Alexandria validation cohort",
        selected_id_sha256=hashlib.sha256("\n".join(part["ids"]).encode()).hexdigest(),
        design_sha256=hashlib.sha256((HERE/"DESIGN.md").read_bytes()).hexdigest()),indent=2)+"\n")
    return part


if __name__=="__main__":
    torch.set_num_threads(1)
    part=cohort()
    train=load(ROOT/"scaled/mp20-data/train.npz"); bank=Retrieval(train)
    out=HERE/"pools";out.mkdir(exist_ok=True)
    args=SimpleNamespace(device="cpu",targets=512,draws=20,out=out)
    for seed in (11,23,37):
        for method in ("drift","flow"):
            state=torch.load(ROOT/f"scaled/runs/mp20/{method}-seed{seed}.pt",weights_only=True,map_location="cpu")
            for steps in ((1,) if method=="drift" else (1,16)):
                predict(state,part,seed,method,"confirm",steps,args)
        bank.save(part,seed,20,512,out/f"retrieval-seed{seed}-confirm.npz")
