"""Audit ordered structure pools and matched records against actual fitted models and public IDs."""
import os
for var in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[var]="1"
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from model import ARMS,HERE,ROOT,SEEDS,checkpoint_path,digest,load_service,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--partial",action="store_true",help="Check currently assembled arms; withhold the final seal.")
    args=parser.parse_args()
    torch.set_num_threads(1)
    splits={}
    for split in ("train","val","test"):
        with np.load(ROOT/f"scaled/mp20-data/{split}.npz") as archive:splits[split]=archive["ids"].tolist()
    assert len(set(splits["train"])|set(splits["val"])|set(splits["test"]))==sum(map(len,splits.values()))
    with np.load(ROOT/"revision/full/mp20/cohort.npz") as cohort:ids=cohort["ids"];z=cohort["z"]
    test_index={m:i for i,m in enumerate(splits["test"])}
    with np.load(ROOT/"scaled/mp20-data/test.npz") as test:
        ix=[test_index[m] for m in ids]
        for field in ("z","k","f"):
            with np.load(ROOT/"revision/full/mp20/cohort.npz") as d:np.testing.assert_equal(d[field],test[field][ix])
    with np.load(ROOT/"scaled/mp20-data/train.npz") as train:train_z=train["z"]
    total=exceptions=checks=0
    manifest={};selections=json.loads((HERE/"selection.json").read_text())
    for seed in SEEDS:
        previous=None
        for arm in ("prior",*ARMS):
            folder=HERE/f"test/{arm}-seed{seed}";ready=json.loads((folder/"ready.json").read_text())
            if args.partial and not (folder/"outcomes.npz").exists():continue
            ck=checkpoint_path(arm,seed)
            assert ready["checkpoint_sha256"]==digest(ck)
            assert ready["cell_sha256"]==digest(folder/"k.npy") and ready["sites_sha256"]==digest(folder/"f.npy")
            with np.load(folder/"crosswalk.npz") as x:np.testing.assert_equal(x["ids"],ids);np.testing.assert_equal(x["z"],z)
            with np.load(folder/"outcomes.npz") as d:
                np.testing.assert_equal(d["ids"],ids);assert d["rms"].shape==d["strict_fit"].shape==d["error"].shape==(9046,20)
                total+=d["rms"].size;exceptions+=int(d["error"].sum());checks+=int(d["public_checks"])
            if arm!="prior":
                with np.load(HERE/f"fits/{arm}-{seed}/schedule.npz") as sch:
                    anchors=sch["anchors"];pos=sch["positive"]
                    assert sorted(anchors.ravel().tolist())==list(range(len(train_z)))
                    for j in range(4):np.testing.assert_equal(train_z[pos[:,:,j]],train_z[anchors])
                    if previous is not None:np.testing.assert_equal(previous,pos)
                    previous=pos
                choice=selections[arm]
                lockfile=HERE/"locks"/(ready["selection_sha256"]+".json")
                assert digest(lockfile)==ready["selection_sha256"] and json.loads(lockfile.read_text())[arm]==choice
                assert choice["updates"]==min((256,1024,3392),key=lambda k:np.mean(choice["validation_energy_by_seed"][str(k)]))
            ks=np.load(folder/"k.npy",mmap_mode="r");fs=np.load(folder/"f.npy",mmap_mode="r")
            model=load_service(arm,seed)
            for i in (0,1424,9044):
                j=min(i+4,len(z));rng=np.random.default_rng(np.random.SeedSequence([410932,seed,i]))
                k,f=model(np.repeat(z[i:j],20,axis=0),rng)
                np.testing.assert_allclose(ks[i:j],k.reshape(j-i,20,6),atol=3e-6,rtol=2e-6)
                np.testing.assert_allclose(fs[i:j],f.reshape(j-i,20,20,3),atol=3e-6,rtol=2e-6)
            manifest[folder.name]=dict(checkpoint=digest(ck),scores=digest(folder/"outcomes.npz"),
                order=digest(folder/"crosswalk.npz"),cell=digest(folder/"k.npy"),sites=digest(folder/"f.npy"))
    if not args.partial:assert total==len(SEEDS)*(len(ARMS)+1)*9046*20
    for path in sorted((HERE/"gpu").glob("*-settings.json")):
        cfg=json.loads(path.read_text());m=cfg["method"]
        for stream in ("10000","episodes"):assert cfg[stream+"_sha256"]==digest(HERE/f"gpu/{m}-{stream}.npz")
        with np.load(HERE/f"gpu/{m}-episodes.npz") as timing,np.load(HERE/f"gpu/{m}-rms.npz") as scored:
            np.testing.assert_equal(ids[:128],timing["ids"]);np.testing.assert_equal(ids[:128],scored["ids"])
            assert (np.diff(timing["elapsed_ms"],axis=1)>0).all()
        with np.load(HERE/f"gpu/{m}-10000.npz") as w:assert w["counts"].sum()==10000
    inter=HERE/"interleaved";cfg=json.loads((inter/"settings.json").read_text())
    order=np.asarray(cfg["order"]);methods=cfg["methods"]
    assert order.shape==(128,len(methods)) and np.array_equal(np.sort(order,axis=1),np.tile(np.arange(len(methods)),(128,1)))
    differences=np.stack([(order==j).sum(0) for j in range(len(methods))])
    assert (differences.max(1)-differences.min(1)).max()<=1
    for i in range(128):
        with np.load(inter/"shards"/f"{i:03}.npz") as a:
            assert str(a["id"])==ids[i] and np.array_equal(order[i],a["order"])
            assert (a["end_unix"]>a["start_unix"]).all()
            assert a["elapsed_ms"].shape==(len(methods),20) and (np.diff(a["elapsed_ms"],axis=1)>0).all()
    for j,m in enumerate(methods):
        with np.load(inter/f"{m}-episodes.npz") as ep,np.load(inter/f"{m}-rms.npz") as s:
            np.testing.assert_array_equal(ep["ids"],ids[:128]);np.testing.assert_array_equal(s["ids"],ids[:128])
            for row in (0,63,127):
                with np.load(inter/"shards"/f"{row:03}.npz") as shard:
                    for field in ("k","f","elapsed_ms"):np.testing.assert_equal(ep[field][row],shard[field][j])
    cpu_clock=HERE/"cpu"
    cfg=json.loads((cpu_clock/"settings.json").read_text())
    with np.load(cpu_clock/"batch-one.npz") as single:
        np.testing.assert_array_equal(single["ids"],ids[:128]);assert single["ms"].shape==(128,6,5)
        assert np.isfinite(single["k"]).all() and np.isfinite(single["f"]).all() and (single["ms"]>0).all()
        np.testing.assert_equal(single["methods"],cfg["methods"])
    for m in cfg["methods"]:
        with np.load(cpu_clock/f"{m}-10000.npz") as w:
            np.testing.assert_equal(w["ids"],ids[np.arange(10000)%128]);assert w["counts"].sum()==10000
            assert float(w["wall_seconds"])*1000>=sum(w["batch_ms"]) and (w["batch_ms"]>0).all()
    if args.partial:
        print("Partially audited",len(manifest),"available arm-seeds,",total,"scored proposals. Final seal withheld.")
        return
    write_json(HERE/"verified.json",dict(splits={k:len(v) for k,v in splits.items()},
        total_scored=total,exception_draws=exceptions,public_api_checks=checks,seed_arms=manifest,
        audit="Forward checkpoint replays at target offsets 0,1424,9044; each 4 or final 2 target block × 20 draws"))
    print("Verified",total,"candidate scores and",checks,"direct public matcher checks.")

if __name__=="__main__":main()
