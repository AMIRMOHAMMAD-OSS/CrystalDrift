"""Audit the published three-seed CSV and its two LaTeX tables directly from ordered score matrices."""
from collections import defaultdict
import csv
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="1"
import numpy as np

HERE=Path(__file__).resolve().parent
V3=HERE.parent
ROOT=V3.parents[1]
METHODS={"Angular 0.2M":"angular-small","Contacts 0.2M":"contacts-small",
         "Angular 2M":"angular-medium","Contacts 2M":"contacts-medium",
         "Native uFlow 0.2M":"uflow-small","Native uFlow 2M":"uflow-medium"}


def score_path(panel,arm):
    if arm.startswith("uflow-"):
        return V3/"uflow/scores"/arm.removeprefix("uflow-")/"frozen"/panel/"outcomes.npz"
    return V3/"evaluation/scores"/panel/arm/"outcomes.npz"


@lru_cache(maxsize=16)
def load(path):
    with np.load(path) as data:
        return tuple(data[key] for key in ("ids","rms","strict_fit","log_volume_ratio"))


def metrics(path,k,ids):
    rows,rms,strict,ratio=load(path)
    assert rows.tolist()==ids
    r=rms[:,:k]
    hit=np.isfinite(r)
    best=np.where(hit,r,np.inf).min(axis=1)
    return hit.any(axis=1),(hit&(np.abs(ratio[:,:k])<=np.log(1.1))).any(axis=1),strict[:,:k].any(axis=1),best


def interval(x,clusters):
    rng=np.random.default_rng(7331)
    if clusters is None:
        values,freq=np.unique(x,return_counts=True)
        draws=rng.multinomial(len(x),freq/len(x),5000)@values/len(x)
    else:
        labels,index=np.unique(clusters,return_inverse=True)
        sums=np.bincount(index,weights=x);counts=np.bincount(index)
        choice=rng.integers(len(labels),size=(5000,len(labels)))
        draws=sums[choice].sum(axis=1)/counts[choice].sum(axis=1)
    return 100*np.quantile(draws,(.025,.975))


def main():
    source=V3/"release/SEED-RESULTS.csv";tex=V3/"paper/replication-results.tex"
    split=json.loads((V3/"protocol/splits.json").read_text())
    records=defaultdict(dict)
    with source.open(encoding="utf8",newline="") as fp:
        for record in csv.DictReader(fp):
            key=(record["panel"],record["method"],int(record["k"]))
            seed=int(record["seed"]);assert seed not in records[key];records[key][seed]=record
    assert set(records)=={(p,m,k) for p in ("test","family","shared") for m in METHODS for k in (1,5,20)}
    paper=tex.read_text(encoding="utf8");verified=0
    for (panel,method,k),seeds in sorted(records.items()):
        ids=split[f"{panel}_ids"];arm=METHODS[method]
        prior_arm="uflow-prior-11" if arm.startswith("uflow-") else "prior"
        before,physical_before,_,_=metrics(score_path(panel,prior_arm),k,ids)
        gains=[];native_gains=[];hit_counts=[]
        assert set(seeds)=={11,23,37}
        for seed,r in sorted(seeds.items()):
            path=score_path(panel,f"{arm}-{seed}")
            assert ROOT/r["score_file"].replace("\\","/")==path
            hit,native,strict,best=metrics(path,k,ids)
            expected=[len(ids),hit.sum(),before.sum(),native.sum(),physical_before.sum()]
            assert expected==[int(r[key]) for key in ("targets","hits","zero_call_hits","native","zero_call_native")]
            gain=hit.astype(float)-before;gains.append(gain);native_gains.append(native.astype(float)-physical_before)
            hit_counts.append(int(hit.sum()))
            np.testing.assert_allclose([100*gain.mean(),best[hit].mean(),strict.mean()],
                                       [float(r[key]) for key in ("scale_pp","rms","strict_mr")],atol=1e-12)
        target=np.mean(gains,axis=0);d=100*target.mean();dn=100*np.mean(native_gains)
        cluster=None if panel=="test" else [split["family_by_challenge_id"][i] for i in ids]
        lo,hi=interval(target,cluster)
        for r in seeds.values():
            np.testing.assert_allclose([d,lo,hi],[float(r[key]) for key in
                ("target_bootstrap_across_fixed_seeds_pp","bootstrap_low_pp","bootstrap_high_pp")],atol=1e-12)
        if (panel,k) in (("test",5),("family",20)):
            line=" & ".join(map(str,[method,int(before.sum()),*hit_counts,f"{d:+.2f}",f"[{lo:+.2f}, {hi:+.2f}]",f"{dn:+.2f}"]))+r"\\"
            assert paper.count(line)==1,line
            verified+=1
    receipt=dict(csv_rows=sum(map(len,records.values())),paired_intervals=len(records),published_table_rows=verified,
                 csv_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),tex_sha256=hashlib.sha256(tex.read_bytes()).hexdigest())
    output=HERE/"audits/replication-table-verification.json"
    if not output.is_file() or json.loads(output.read_text())!=receipt:
        output.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf8")
    print(receipt,flush=True)


if __name__=="__main__":main()
