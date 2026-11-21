"""Paired recovery over identical seeds with anonymous-geometry strata and family resampling."""
import argparse
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;OUT=HERE/"audits";V3=HERE.parent

def outcomes(panel,tag):
    with np.load(V3/f"evaluation/scores/{panel}/{tag}/outcomes.npz") as x:
        return x["ids"].astype(str), np.isfinite(x["rms"]), np.abs(x["log_volume_ratio"])<=np.log(1.1)


def interval(data,families):
    """Target-weighted effect, stratification fixed, sample whole family clusters."""
    if not len(data):return None
    labels,codes=np.unique(families,return_inverse=True);n=len(labels)
    counts=np.bincount(codes)
    sums=np.bincount(codes,weights=data)
    boot=np.random.default_rng(7331).integers(n,size=(5000,n))
    sim=sums[boot].sum(1)/counts[boot].sum(1)
    return dict(N=len(data),families=n,estimate=float(data.mean()),lower=float(np.quantile(sim,.025)),
                upper=float(np.quantile(sim,.975)))


def summarize(panel,label,tags,prior_tag="prior"):
    ids,p,vol=outcomes(panel,prior_tag)
    total=[];native=[]
    for tag in tags:
        id2,r,v=outcomes(panel,tag);assert np.array_equal(ids,id2)
        total.append(r);native.append(r&v)
    total=np.stack(total);native=np.stack(native)
    if panel.startswith("arrangement"):
        families=json.loads((OUT/"arrangement-frozen.json").read_text())["key_by_id"]
    else:families=json.loads((V3/"protocol/splits.json").read_text())["family_by_challenge_id"]
    family=np.array([families[x] for x in ids]);mask={"all":np.ones(len(ids),bool)}
    if panel=="shared":
        src=json.loads((OUT/"shared-geometry.json").read_text())
        found={x["id"]:any(y["rms"] is not None for y in x["attempts"]) for x in src["records"]}
        anon=np.array([found[x] for x in ids])
        mask.update(same_family_anon_match=anon,no_same_family_anon_match=~anon)
    out={}
    for name,m in mask.items():
        rows={}
        for k in (1,5,20):
            for criterion,T,P in (("scale",total,p),("native",native,p&vol)):
                r=T[:,:,:k].any(2).mean(0);p0=P[:,:k].any(1)
                rows[f"{criterion}-{k}"]=dict(learned_hits_per_seed=T[:,m,:k].any(2).sum(1).tolist(),
                    prior_hits=int(p0[m].sum()),paired_gain=interval((r-p0)[m],family[m]))
        out[name]=rows
    dest=OUT/f"family-effects-{panel}.json"
    existing=json.loads(dest.read_text()) if dest.exists() else {};existing[label]=dict(tags=tags,prior=prior_tag,rows=out)
    dest.write_text(json.dumps(existing,indent=2)+"\n",encoding="utf-8")
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument("panel",choices=["family","shared","arrangement","arrangement-control"])
    p.add_argument("label");p.add_argument("tags",nargs="+");p.add_argument("--prior",default="prior");a=p.parse_args()
    summarize(a.panel,a.label,a.tags,a.prior)
    print(a.panel,a.label)

if __name__=="__main__":main()
