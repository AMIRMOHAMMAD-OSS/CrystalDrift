"""Compare a learned ranking, frequency and chemical distance on one train-only candidate set."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

for v in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[v]="1"
import numpy as np
import torch

from .reachability import OUT,keys,records
from ..evaluation.predict import setup_split
from ..evaluation.selector import Selector
from ..evaluation.storage import pool
from ..models.sampler import Sampler,V3

METHODS=("lookup","frequency","learned")
FIELDS=("canonical","equivalent","family","group","fixed")
TAGS={"lookup":"prior","frequency":"empirical-rank","learned":"learned-rank"}


def score(panel,svc,weights_hash):
    target,split=setup_split(panel)
    labels={r["id"]:r for r in records(split)}
    ids=np.array([r[0] for r in target]);where={s:i for i,s in enumerate(ids)}
    flags=np.zeros((len(METHODS),len(FIELDS),len(target),20),bool)
    truth=np.zeros(len(ids),bool)
    raw={}
    for selector,tag in TAGS.items():
        for shard in sorted(pool(panel,tag).glob("?????-?????.npz")):
            with np.load(shard) as d:
                for mid,groups,patterns in zip(d["ids"],d["group"],d["patterns"]):
                    raw[selector,str(mid)]=list(zip(groups,[tuple(sorted(tuple(x) for x in p)) for p in json.loads(patterns)]))
    checked=Counter();onecall=0;options_n=[]
    for ix,(mid,_j,z) in enumerate(target):
        label=labels[mid];known="error" not in label;truth[ix]=known
        if known:
            g0=label["group"];p0=tuple(sorted(tuple(x) for x in label["template"]))
            f0,e0,_=keys(g0,p0)
        opts,x=svc.describe(z)
        m=len(opts);options_n.append(m);assert 1<=m<=20
        with torch.inference_mode():
            logits=svc.model(torch.from_numpy(x[:m])).flatten().numpy()
        onecall+=1
        orders=(np.arange(m),np.argsort(-x[:m,0],kind="stable"),np.argsort(-logits,kind="stable"))
        base=np.zeros((len(FIELDS),m),bool)
        for j,(g,p,donor) in enumerate(opts):
            fam,eq,free=keys(g,p)
            base[:,j]=[known and (g,p)==(g0,p0),known and eq==e0,known and fam==f0,known and g==g0,free==0]
        for k,(method,order) in enumerate(zip(METHODS,orders)):
            ordered=order[np.arange(20)%m]
            flags[k,:,ix,:]=base[:,ordered]
            existing=raw.get((method,mid))
            if existing is not None:
                for j,(g,p) in enumerate(existing):
                    assert (g,p)==opts[ordered[j]][:2],(mid,j,method)
                checked[method]+=len(existing)
    assert np.array_equal(flags[0].any(-1),flags[1].any(-1)) and np.array_equal(flags[0].any(-1),flags[2].any(-1))
    with np.load(OUT/f"reachable-{panel}.npz") as d:
        assert np.array_equal(d["ids"],ids)
        for j,var in enumerate(FIELDS):assert np.array_equal(d[var],flags[0,j]),var
    by_method={}
    for i,m in enumerate(METHODS):
        by_method[m]={}
        for k in (1,5,20):
            a=flags[i,:,:,:k]
            by_method[m][k]={f:int(a[j].any(-1).sum()) for j,f in enumerate(FIELDS)}
            by_method[m][k]["fixed_proposals"]=int(a[-1].sum())
    result=dict(n=len(ids),truth_available=int(truth.sum()),panel=panel,weight_sha256=weights_hash,
        ranker_calls=onecall,shared_shortlist=True,cyclic_draws=True,checked_against_saved_proposals=checked,
        mean_distinct_options=float(np.mean(options_n)),by_method=by_method)
    file=OUT/f"ranking-coverage-{panel}"
    file.with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    np.savez_compressed(file.with_suffix(".npz"),ids=ids,methods=METHODS,fields=FIELDS,
                        flags=flags,shortlist_size=options_n,truth_available=truth)
    print(panel,len(ids),[(m,[by_method[m][k]["equivalent"] for k in (1,5,20)]) for m in METHODS],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("panels",nargs="+");arg=p.parse_args()
    torch.set_num_threads(1)
    ranking=Selector(Sampler());weights=V3/"evaluation/selector/weights.pt"
    weight_hash=hashlib.sha256(weights.read_bytes()).hexdigest()
    ranking.model.load_state_dict(torch.load(weights,weights_only=True));ranking.model.eval()
    for panel in arg.panels:score(panel,ranking,weight_hash)


if __name__=="__main__":main()
