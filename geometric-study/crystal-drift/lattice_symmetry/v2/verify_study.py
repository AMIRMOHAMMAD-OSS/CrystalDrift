"""Independent study-level checks of conditioning, ordered evidence, latencies and attribution."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
import numpy as np

V2=Path(__file__).resolve().parent
ROOT=V2.parents[1]
C=V2/"csp";W=V2/"wyckoff"


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(4<<20),b""):h.update(b)
    return h.hexdigest()


def js(path):return json.loads(path.read_text(encoding="utf8"))


def csvdata(path):
    with path.open(newline="",encoding="utf8") as fp:return list(csv.DictReader(fp))


def fixed(new,specs,ids):
    def observed(a,k):
        best=np.where(np.isfinite(a[:,:k]),a[:,:k],np.inf).min(axis=1)
        matched=np.isfinite(best)
        return (int(matched.sum()),float(best[matched].mean()))
    seeds=csvdata(C/"seeds.csv")
    rows=csvdata(C/"quality.csv")
    out={}
    for m,paths in new.items():
        byk={1:[],5:[],20:[]}
        for s,p in zip(specs[m],paths):
            with np.load(p) as d:
                np.testing.assert_equal(d["ids"],ids)
                a=d["rms"]
                if m=="retrieval":
                    a=a[:,d["arms"].tolist().index("retrieval-seed"+str(s)),:]
                assert a.shape==(9046,20)
                for k in (1,5,20):
                    n,r=observed(a,k)
                    record=next(row for row in seeds if row["method"]==m and row["seed"]==str(s) and int(row["k"])==k)
                    assert int(record["hits"])==n
                    np.testing.assert_allclose(r,float(record["mean_normalized_rmse"]))
                    byk[k].append((n,r))
        out[m]={}
        for k,values in byk.items():
            a=np.array(values)
            mr=a[:,0].mean()/9046
            r=a[:,1].mean()
            row=next(t for t in rows if t["method"]==m and int(t["k"])==k)
            np.testing.assert_allclose([mr,r],[float(row["mr"]),float(row["mean_normalized_rmse"])])
            out[m][str(k)]=dict(mr=mr,mean_best_rms=r,matched_by_seed=a[:,0].astype(int).tolist())
    return out


def timed(ids):
    rows=csvdata(C/"timing-curves.csv")
    mlist=sorted({r["method"] for r in rows});ret={}
    for m in mlist:
        with np.load(C/f"gpu/{m}-episodes.npz") as times,np.load(C/f"gpu/{m}-rms.npz") as scores,np.load(C/f"gpu/{m}-10000.npz") as workload:
            t=times["elapsed_ms"];q=scores["rms"]
            np.testing.assert_array_equal(times["ids"],ids[:128]);np.testing.assert_array_equal(scores["ids"],ids[:128])
            assert t.shape==q.shape==(128,20) and (np.diff(t,axis=1)>0).all() and workload["counts"].sum()==10000
            hit={}
            for r in (r for r in rows if r["method"]==m):
                k=int(r["budget"])
                eligible=t<=k if r["axis"]=="ms" else np.broadcast_to(np.arange(20)<k,t.shape)
                n=int((np.isfinite(q)&eligible).any(1).sum())
                assert int(r["hits"])==n
                np.testing.assert_allclose(float(r["mean_completed"]),eligible.sum(1).mean())
                if r["axis"]=="ms" and k==100:hit=dict(hits=n,mean_completed=eligible.sum(1).mean())
            ret[m]={**hit,"seconds_for_10000":float(workload["elapsed_s"])}
    return ret


def balanced(ids):
    folder=C/"interleaved"
    data={}
    for row in csvdata(folder/"recovery.csv"):
        if int(row["ms"])!=100:continue
        arm=row["method"]
        with np.load(folder/f"{arm}-episodes.npz") as ep,np.load(folder/f"{arm}-rms.npz") as sc:
            np.testing.assert_equal(ep["ids"],ids[:128])
            np.testing.assert_equal(sc["ids"],ids[:128])
            elapsed=ep["elapsed_ms"]
            hit=(np.isfinite(sc["rms"]) & (elapsed<=100)).any(1)
            assert int(hit.sum())==int(row["hits"])
            np.testing.assert_allclose(float(row["completed"]),(elapsed<=100).sum(1).mean())
            data[arm]=dict(hits=int(hit.sum()),targets=len(hit),completed_per_target=float(row["completed"]))
    assert set(data)=={"angular","radial","flow16","old-flow16","prior","old-drift"}
    return data


def template():
    s=js(W/"summary.json")
    pool_records={}
    for arm,r in s["arms"].items():
        with gzip.open(W/f"scores/{arm}.jsonl.gz","rt") as fp:score=[json.loads(x) for x in fp]
        with np.load(W/f"pools/{arm}.npz") as d:
            assert digest(W/f"pools/{arm}.npz")==r["metadata"]["pool_sha256"]
            ids=d["id"];k=d["L"].shape[1];n=len(ids)
            assert len(score)==n*k and len(set(ids))==n
            per={i:[] for i in ids}
            for line in score:per[line["id"]].append(line)
            for z in per.values():assert [p["draw"] for p in z]==list(range(1,k+1))
            counts={}
            for budget,values in r["recovery"].items():
                best=[]
                for mid in ids:
                    valid=[p["rms"] for p in per[mid][:int(budget)] if p.get("rms") is not None and np.isfinite(p["rms"])]
                    if valid:best.append(min(valid))
                assert len(best)==values["hits"]
                np.testing.assert_allclose(np.mean(best),values["matched_rms"])
                counts[budget]=len(best)
            pool_records[arm]=dict(k=k,targets=n,hits=counts)
    with np.load(W/"full/one-draw.npz") as b,np.load(ROOT/"scaled/mp20-data/test.npz") as ref:
        np.testing.assert_equal(b["ids"],ref["ids"])
        full=js(W/"full/summary.json")
        assert np.isfinite(b["rms"]).sum(0).tolist()==full["hits"]
    return dict(cohort=pool_records,official_one_draw=full)


def main():
    from protect_baseline import baseline
    assert baseline()==js(V2/"baseline.json")
    with np.load(ROOT/"revision/full/mp20/cohort.npz") as c:ids=c["ids"]
    with np.load(ROOT/"scaled/mp20-data/test.npz") as official:assert set(ids)==set(official["ids"]) and len(ids)==9046
    names=("prior","radial","power0","angular","fixed-angular","flow16")
    specs={m:[11,23,37] for m in (*names,"old-drift","meanflow","old-flow16","retrieval")}
    specs["uflow"]=["released"]
    paths={m:[C/f"test/{m}-seed{s}/outcomes.npz" for s in specs[m]] for m in names}
    for m,key in (("old-drift","drift"),("meanflow","meanflow"),("old-flow16","flow16")):
        paths[m]=[ROOT/f"expansion/training/test/{key}-seed{s}/outcomes.npz" for s in specs[m]]
    paths["uflow"]=[ROOT/"revision/full/external/mp20/complete-crosswalk-k20.npz"]
    paths["retrieval"]=[ROOT/"revision/full/mp20/complete-crosswalk-k20.npz"]*3
    checked=js(C/"verified.json");assert checked["total_scored"]==6*3*9046*20
    fixed_rows=fixed(paths,specs,ids)
    proof=js(V2/"results/physics-acceptance.json");gpu=js(V2/"results/physics-cuda.json")
    assert max(proof["independent_batch_errors"].values())<1e-10
    assert proof["physical_site_error_angstrom"]<1e-10 and proof["addition_theorem_oracle"]<1e-10
    denovo=js(V2/"denovo/verified.json")
    assert len(denovo["raw"])==3 and all(r["attempts"]==10000 for r in denovo["raw"])
    assert all(r["draws"]==16 for r in denovo["physical"])
    inputs=[C/"verified.json",C/"selection.json",C/"quality.csv",C/"timing-curves.csv",C/"interleaved/recovery.csv",
            C/"geometry-sensitivity.csv",C/"geometry-sensitivity-settings.json",C/"cpu/timing.csv",V2/"sources/published.json",
            V2/"denovo/per-draw-verification.json",V2/"denovo/gpu-costs.csv",
            V2/"denovo/formula-strata.json",V2/"results/checkpoint-symmetry-selected.json",W/"verified.json",
            W/"summary.json",W/"full/summary.json",V2/"denovo/verified.json",V2/"results/physics-acceptance.json",
            V2/"results/physics-cuda.json",V2/"sources/primary-records.json"]
    audit=dict(formula=fixed_rows,timing100=timed(ids),balanced100=balanced(ids),wyckoff=template(),de_novo=denovo,
               core_distinct_predictions=checked["total_scored"],double_batch=proof["independent_batch_errors"],
               gpu=gpu,sha256={p.relative_to(V2).as_posix():digest(p) for p in inputs})
    (V2/"audit.json").write_text(json.dumps(audit,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print("Reconciled all conditional prefixes, retrospective deadlines, template predictions and de novo records.")

if __name__=="__main__":main()
