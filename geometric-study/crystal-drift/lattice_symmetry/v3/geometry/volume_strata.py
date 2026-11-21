"""Paired raw/seed-volume and compatible-occupation recovery on complete ordered MP-20 arrays."""
import argparse
import csv
import json
import os
from pathlib import Path
for name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[name]="1"
import numpy as np
from ..evaluation.analysis import ci
from ..evaluation.storage import pool

HERE=Path(__file__).resolve().parent
V3=HERE.parent
OUT=HERE/"audits"
E=V3/"evaluation"
TAGS=("prior","angular-small-11","contacts-small-11","reciprocal-small-11","meanflow-small-11",
      "angular-medium-11","contacts-medium-11","meanflow-medium-11","untrained-small",
      "angular-small-long","angular-medium-long","contacts-medium-long","meanflow-medium-long",
      "angular-small-full-11","contacts-small-full-11","reciprocal-small-full-11","meanflow-small-full-11",
      "meanflow-small-strict","meanflow-medium-strict")+tuple(
      f"{arm}-{tier}-{seed}" for seed in (23,37) for tier in ("small","medium") for arm in ("angular","contacts"))
LOG_GATE=np.log(1.1)


def interval(a,b,mask):
    d=a[mask].astype(float)-b[mask].astype(float)
    if not len(d):return (None,None,None)
    return (round(100*d.mean(),5),*[round(100*x,5) for x in ci(d)])


def endpoint(name):
    path=E/f"scores/test/{name}/outcomes.npz"
    with np.load(path) as d:
        return d["ids"].astype(str), np.isfinite(d["rms"]),d["log_volume_ratio"]


def study(tags):
    with np.load(OUT/"reachable-test.npz") as d:
        ids=d["ids"].astype(str);eq=d["equivalent"];known=d["truth_available"]
    with (E/"duplicates.csv").open() as fp:
        dup={r["id"]:bool(r["earlier_training_match"]) for r in csv.DictReader(fp) if r["split"]=="test"}
    assert set(dup)==set(ids)
    same=np.array([dup[v] for v in ids])
    ids0,prior_h,prior_v=endpoint("prior")
    assert np.array_equal(ids,ids0)
    assert eq.shape==prior_h.shape==prior_v.shape==(9046,20)
    # Array-level crosswalk compares the actual prior stream with each first/controlled raw arm.
    receipt=json.loads((OUT/"raw-templates-test.json").read_text())
    assert receipt["complete"] and receipt["ordered_draws"]==eq.size
    records=[];effects=[]
    for name in tags:
        assert name=="prior" or name in receipt["matching_metadata_methods"],name
        sig=receipt.get("signatures",{}).get(name)
        if sig is None:
            settings=json.loads((pool("test",name)/"settings.json").read_text())
            sig=json.loads(settings["signature"])
        assert sig["family"]=="symbolic" and sig["selector"]=="lookup"
        ids1,learned_h,learned_v=endpoint(name)
        assert np.array_equal(ids1,ids) and learned_h.shape==prior_h.shape
        for k in (1,5,20):
            h=learned_h[:,:k];p=prior_h[:,:k]
            eqk=eq[:,:k];reaches=eqk.any(1)
            v0=np.abs(prior_v[:,:k])<=LOG_GATE;v1=np.abs(learned_v[:,:k])<=LOG_GATE
            methods=dict(scale=h.any(1),same_pattern_scale=(h&eqk).any(1),
                learned_volume=(h&v1).any(1),seed_volume=(h&v0).any(1),
                same_pattern_learned_volume=(h&eqk&v1).any(1),same_pattern_seed_volume=(h&eqk&v0).any(1))
            priors=dict(scale=p.any(1),same_pattern_scale=(p&eqk).any(1),
                learned_volume=(p&v0).any(1),seed_volume=(p&v0).any(1),
                same_pattern_learned_volume=(p&eqk&v0).any(1),same_pattern_seed_volume=(p&eqk&v0).any(1))
            for duplicates,ds in (("all",np.ones(len(ids),bool)),("exclude_tight_train_match",~same),("tight_train_match",same)):
                for subgroup,rs in (("all",np.ones(len(ids),bool)),("setting_equivalent_reachable",reaches),
                            ("known_unreachable",known&~reaches),("unknown",~known)):
                    mask=ds&rs;n=int(mask.sum())
                    if not n:continue
                    row=dict(method=name,k=k,training_match=duplicates,availability=subgroup,n=n)
                    record={**row,**{label:int(value[mask].sum()) for label,value in methods.items()}}
                    for effect,labels in (("seed_volume_minus_learned",("seed_volume","learned_volume")),
                                          ("same_pattern_seed_minus_learned",("same_pattern_seed_volume","same_pattern_learned_volume"))):
                        a,b=(methods[key] for key in labels)
                        delta,lo,hi=interval(a,b,mask)
                        record.update({effect+"_pp":delta,effect+"_low":lo,effect+"_high":hi})
                    records.append(record)
                    for label,success in methods.items():
                        reference=priors[label];delta,lo,hi=interval(success,reference,mask)
                        effects.append({**row,"criterion":label,"method_hits":int(success[mask].sum()),
                            "prior_hits":int(reference[mask].sum()),"method_only":int((success&~reference&mask).sum()),
                            "prior_only":int((~success&reference&mask).sum()),"pp":delta,"low":lo,"high":hi})
    for label,rows in (("seed-volume-pattern",records),("seed-volume-pattern-paired",effects)):
        dest=OUT/(label+".csv")
        tmp=dest.with_suffix(".part")
        with tmp.open("w",newline="",encoding="utf8") as fp:
            writer=csv.DictWriter(fp,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        tmp.replace(dest)
    print("Official volume and pattern rows",len(records),"for",tags,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--tags",nargs="+",default=TAGS)
    p.add_argument("--available",action="store_true");a=p.parse_args()
    use=[name for name in a.tags if not a.available or (E/f"scores/test/{name}/outcomes.npz").is_file()]
    study(use)
