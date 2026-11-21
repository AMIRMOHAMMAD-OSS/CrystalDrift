"""Check the final outputs, export the ID crosswalk, and hash all benchmark inputs."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
from standard_match import order,paths


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inputs={ROOT/"data/multi-test.npz",ROOT/"data/multi-train.npz",
            ROOT/"uflowcsp-results/multi-seed2026-euler1.npz",ROOT/"uflowcsp-results/multi-crosswalk.csv",
            ROOT/"uflowcsp-results/train-ids.json",ROOT/"uflowcsp-results/test-ids.json"}
    with np.load(ROOT/"data/multi-test.npz") as test:
        ix=order("multi",test["ids"])
        ids,z=test["ids"][ix],test["z"][ix]
    with np.load(ROOT/"data/multi-train.npz") as train:
        counts=np.sum((train["z"][:,None,:]==z[None,:,:]).all(-1),axis=0)
        assert set(ids).isdisjoint(train["ids"])
    release_train=set(json.loads((ROOT/"uflowcsp-results/train-ids.json").read_text()))
    release_test=set(json.loads((ROOT/"uflowcsp-results/test-ids.json").read_text()))
    assert set(ids).isdisjoint(release_train) and set(ids)<=release_test
    with (OUT/"target-crosswalk.csv").open("w",newline="",encoding="utf8") as fp:
        writer=csv.writer(fp)
        writer.writerow(["case","test_index","material_id","atomic_numbers","active_sites","count_exact_train_composition","in_uflow_train","in_uflow_test"])
        for i,(mid,row) in enumerate(zip(ids,z)):
            writer.writerow([i,ix[i],mid," ".join(map(str,row[row>0])),sum(row>0),counts[i],mid in release_train,mid in release_test])
    with np.load(OUT/"uflow-20.npz") as new,np.load(ROOT/"uflowcsp-results/multi-seed2026-euler1.npz") as old:
        for key in ("k","f"):
            assert new[key][:,:5].tobytes()==old[key].tobytes()
    report={}
    methods=("exact","flow1","flow16","retrieval","uflow")
    workload_ids=ids[np.arange(10000)%128]
    for method in methods:
        seeds=(2026,) if method=="uflow" else (11,23,37)
        for seed in seeds:
            if method!="uflow":
                path=paths("multi",method,seed)
                if path is not None: inputs.add(path)
                if method!="retrieval":
                    name="drift" if method=="exact" else "flow"
                    inputs.add(ROOT/"speed-results/portable"/f"multi-{name}-{seed}.npz")
            historical=ROOT/"structure-matching"/f"multi-{method}-{seed}.json"
            inputs.add(historical)
            previous=json.loads(historical.read_text())
            for kind in ("headline","episode"):
                with np.load(OUT/f"rms-{kind}-{method}-{seed}.npz") as src:
                    np.testing.assert_equal(src["ids"],ids)
                    assert src["rms"].shape==(128,20)
                    report[f"{kind}-{method}-{seed}"]=dict(n_candidates=src["rms"].size,
                        matcher_exceptions=int(src["exception_count"]))
                    if kind=="headline":
                        np.testing.assert_allclose(src["rms"][:,:5],np.asarray(previous["rms_by_candidate"],float),
                                                  equal_nan=True,rtol=1e-7,atol=1e-7)
            with np.load(OUT/f"episode-{method}-{seed}.npz") as timed:
                np.testing.assert_equal(timed["ids"],ids)
                np.testing.assert_equal(timed["z"],z)
                assert np.isfinite(timed["k"]).all() and np.isfinite(timed["f"]).all()
                assert (timed["elapsed_ms"]>0).all() and (np.diff(timed["elapsed_ms"],axis=1)>0).all()
        with np.load(OUT/f"ten-thousand-{method}-{seeds[0]}.npz") as batch:
            np.testing.assert_equal(batch["ids"],workload_ids)
            np.testing.assert_equal(batch["counts"],[12]*833+[4])
            assert float(batch["elapsed_s"])*1000>=sum(batch["batch_ms"])
    with (OUT/"runtime.csv").open() as fp:
        assert len(list(csv.DictReader(fp)))==5
    outputs=list(OUT.glob("*.npz"))+list(OUT.glob("*.csv"))+list(OUT.glob("*.py"))
    outputs += [OUT/"settings.json",OUT/"uflow-extension.json",OUT/"revision-binary-accuracy-time.svg",
                ROOT/"paper/revision-benchmark.tex",ROOT/"paper/revision-binary-accuracy-time.pdf"]
    manifest=dict(targets=len(ids),fitted_local_seeds=[11,23,37],released_seed=2026,
                  results=report,inputs={str(p.relative_to(ROOT)):sha(p) for p in sorted(inputs)},
                  artifacts={str(p.relative_to(ROOT)):sha(p) for p in sorted(outputs)})
    (OUT/"verification.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf8")
    print("Validated",len(report),"case-by-draw matrices, 5 identical 10,000-request sequences and all source prefixes")


if __name__=="__main__":
    main()
