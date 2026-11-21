"""Independently verify nested rates, selected passes, paired inputs and GPU workload records."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024**2),b""):h.update(block)
    return h.hexdigest()


def main():
    with np.load(ROOT/"revision/full/mp20/cohort.npz") as d:ids=d["ids"];chem=d["z"]
    order=lambda x:sorted(x,key=lambda s:hashlib.sha256(str(s).encode()).digest())
    assert list(ids)==order(ids)
    select=json.loads((HERE/"selection.json").read_text())
    with (HERE/"test-quality-seeds.csv").open(newline="") as f:
        report={(r["method"],int(r["seed"]),int(r["k"])):r for r in csv.DictReader(f) if r["method"]!="crystalflow5"}
    ages={}; counters={}; hashes={}
    for m in ("drift","iid","flow","meanflow"):
        spec=select[m];nfe=16 if m=="flow" else 1
        means={}
        for p in (1,3,10):
            observations=[]
            for seed in (11,23,37):
                base=HERE/f"fits/{m}-seed{seed}"
                val=np.load(base/f"pass{p}-val-{nfe}.npz")
                observations.append(val["energy"].mean())
                log=json.loads((base/f"pass{p}.json").read_text())
                ck=base/f"pass{p}.pt"
                assert log["updates"]==p*1696 and log["positive_examples"]==p*108544
                assert log["checkpoint_sha256"]==digest(ck)
                hashes[str(ck.relative_to(HERE))]=log["checkpoint_sha256"]
            means[p]=np.mean(observations)
        assert min(means,key=means.get)==spec["pass"]
        ages[m]=spec["pass"]
    with np.load(ROOT/"revision/full/mp20/complete-crosswalk-k20.npz") as archive:
        np.testing.assert_array_equal(archive["ids"],ids)
        original_arms=archive["arms"].tolist()
        original_rms=archive["rms"]
    replication={}
    total=0
    for m in ("drift","iid","flow1","flow16","meanflow"):
        for seed in (11,23,37):
            base=HERE/f"test/{m}-seed{seed}";cross=base/"outcomes.npz"
            meta=json.loads((base/"ready.json").read_text())
            assert meta["pass"]==ages["flow" if m.startswith("flow") else m]
            with np.load(cross) as a:
                np.testing.assert_array_equal(a["ids"],ids)
                r=a["rms"]; assert r.shape==(9046,20)
                if m=="drift":
                    archived=original_rms[:,original_arms.index(base.name)]
                    accepted=np.isfinite(r)&np.isfinite(archived)
                    errors=np.abs(r[accepted]-archived[accepted])
                    flags=np.argwhere(np.isfinite(r)!=np.isfinite(archived))
                    replication[base.name]=dict(hit_flag_disagreements=len(flags),
                        changed_candidates=[dict(id=str(ids[i]),draw=int(j+1),archived=bool(np.isfinite(archived[i,j])),
                                                 refit=bool(np.isfinite(r[i,j]))) for i,j in flags],
                        changed_target_indicators={k:int((np.isfinite(r[:,:k]).any(1)!=np.isfinite(archived[:,:k]).any(1)).sum())
                                                   for k in (1,5,20)},
                        common_accepted_pairs=int(accepted.sum()),mean_absolute_rms_change=float(errors.mean()),
                        maximum_absolute_rms_change=float(errors.max()))
                for k in (1,5,20):
                    rec=report[m,seed,k]
                    hit=np.isfinite(r[:,:k]).any(1)
                    vals=np.where(np.isfinite(r[:,:k]),r[:,:k],np.inf).min(1)[hit]
                    assert hit.sum()==int(rec["hits"])
                    np.testing.assert_allclose([hit.mean(),vals.mean()],[float(rec["mr"]),float(rec["rmse"])])
                counters[base.name]=dict(candidates=r.size,matcher_errors=int(a["error"].sum()),
                                         public_checks=int(a["public_checks"]),hits_at20=int(np.isfinite(r).any(1).sum()))
                total+=r.size
            for filename,expected in (("k.npy",(9046,20,6)),("f.npy",(9046,20,20,3))):
                path=base/filename;data=np.load(path,mmap_mode="r")
                assert data.shape==expected and np.isfinite(data).all()
                if filename=="f.npy":
                    for i in range(0,len(chem),128):
                        assert not np.any(data[i:i+128]*(chem[i:i+128,None,:,None]==0))
                hashes[str(path.relative_to(HERE))]=digest(path)
            hashes[str(cross.relative_to(HERE))]=digest(cross)
    assert total==2713800
    gpu=HERE/"gpu"
    with (gpu/"runtime.csv").open(newline="") as f:
        times={r["method"]:r for r in csv.DictReader(f)}
    for method in ("drift","iid","meanflow","flow1","flow16","retrieval","uflow"):
        with np.load(gpu/f"{method}-episodes.npz") as a,np.load(gpu/f"{method}-rms.npz") as scores,np.load(gpu/f"{method}-10000.npz") as b:
            np.testing.assert_array_equal(a["ids"],ids[:128])
            np.testing.assert_array_equal(scores["ids"],ids[:128])
            np.testing.assert_array_equal(b["ids"],ids[np.arange(10000)%128])
            assert scores["rms"].shape==(128,20) and b["counts"].sum()==10000
            dt=np.diff(a["elapsed_ms"],axis=1,prepend=0)
            assert (dt>0).all()
            np.testing.assert_allclose(np.median(dt),float(times[method]["b1_median_ms"]))
            np.testing.assert_allclose(float(b["elapsed_s"]),float(times[method]["seconds_for_10000"]))
        for path in gpu.glob(method+"-*.npz"):hashes[str(path.relative_to(HERE))]=digest(path)
    external=HERE.parent/"external"
    crystalflow={}
    if (external/"timing-s5/runtime.json").exists():
        timing=json.loads((external/"timing-s5/runtime.json").read_text())
        with np.load(external/"timing-s5/serial.npz") as serial,np.load(external/"timing-s5/batch.npz") as batch:
            np.testing.assert_array_equal(serial["ids"],ids[:128])
            np.testing.assert_array_equal(batch["ids"],ids[np.arange(10000)%128])
            np.testing.assert_allclose(timing["median_b1_ms"],np.median(np.diff(serial["elapsed_ms"],axis=1,prepend=0)))
            np.testing.assert_allclose(timing["batch_total_seconds"],float(batch["wall_seconds"]))
            np.testing.assert_allclose(timing["per_second"],10000/float(batch["wall_seconds"]))
        crystalflow.update(timed_requests=10000,serial_requests=2560)
    if (external/"csp-s5/score-crosswalk.npz").exists():
        summary=json.loads((external/"csp-s5/summary.json").read_text())
        with np.load(external/"csp-s5/score-crosswalk.npz") as cross:
            np.testing.assert_array_equal(cross["ids"],ids)
            confirmed=[]
            for k in (1,5,20):
                key=f"9046@{k}"
                if key not in summary:continue
                assert cross["scored"][:,:k].all()
                rms=cross["rms"][:,:k]
                valid=np.isfinite(rms)
                matched=valid.any(1)
                best=np.where(valid,rms,np.inf).min(1)[matched]
                np.testing.assert_allclose([matched.mean(),best.mean()],[summary[key]["mr"],summary[key]["rmse"]])
                confirmed.append(k)
            crystalflow["complete_case_budgets_verified"]=confirmed
    output=dict(selected_passes=ages,total_generated_and_scored=total,official_test_ids=len(ids),fits=12,checkpoints=36,
        per_test_arm=counters,archived_one_pass_drift_replication=replication,gpu_workloads=len(times),gpu_candidates_per_method=2560,throughput_requests_per_method=10000,
        additional_crystalflow=crystalflow,material_id_order_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),sha256=hashes)
    (HERE/"verification.json").write_text(json.dumps(output,indent=1)+"\n")
    print(total,"conditional test candidates verified,",len(times),"GPU timing workloads,",len(hashes),"source artifacts")


if __name__=="__main__":main()
