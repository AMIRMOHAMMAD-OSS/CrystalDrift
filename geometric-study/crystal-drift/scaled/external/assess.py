"""Use the shared estimators, matched ID ordering and fixed-checkpoint case bootstrap."""
import argparse
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parent
SCALED = ROOT.parent
sys.path.insert(0,str(SCALED))


def metrics(cohort):
    import evaluate
    import match
    from experiment import load, select, sha
    folder = ROOT / cohort
    product = folder / "uflow-seed2026-test.npz"
    test_path = SCALED / (cohort+"-data") / "test.npz"
    train_path = test_path.with_name("train.npz")
    match.score((product,test_path))
    local_scale = SCALED / "runs" / cohort / "score-scale.json"
    if local_scale.exists():
        scale_report = json.loads(local_scale.read_text())
        assert scale_report["train_sha256"] == sha(train_path)
    else:
        train = load(train_path)
        idx = select(train["ids"],2048)
        fit = evaluate.process(train["k"][idx],train["f"][idx],train["z"][idx])
        assert np.isfinite(fit).all()
        scale_report = dict(scale=fit.std(0,ddof=1).clip(1e-5).tolist(),
                            scale_sample=len(idx),train_sha256=sha(train_path))
    match.write(folder/"score-scale.json",scale_report)
    ref = load(test_path)
    result = evaluate.score(product,ref,np.asarray(scale_report["scale"]))
    result.update(method="uflow",seed=2026,split="test",input_sha256=sha(product),
                  scale_train_sha256=scale_report["train_sha256"])
    match.write(product.with_suffix(".score.json"),result)
    if cohort=="wbm":
        transfer_scale = json.loads((SCALED/"runs"/"mp20"/"score-scale.json").read_text())
        transfer = evaluate.score(product,ref,np.asarray(transfer_scale["scale"]))
        transfer.update(method="uflow",seed=2026,split="test",input_sha256=sha(product),
                        scale_train_sha256=transfer_scale["train_sha256"])
        match.write(folder/"uflow-seed2026-test.mp20-scale.score.json",transfer)
    return result


def series(path,key):
    d = json.loads(path.read_text())
    if key=="energy":
        return d["target_ids"], np.asarray(d["per_target_energy"])
    hit = np.isfinite(np.asarray(d["rms_by_candidate"],float))
    return d["material_ids"], hit[:,:int(key[1:])].any(1).astype(float)


def compare(cohort,local_run=None):
    import match
    folder = ROOT/cohort
    local_run = local_run or cohort
    local = SCALED/"runs"/local_run
    output = {}
    for key in ["@1","@5","energy"]:
        suffix = ".score.json" if key=="energy" else "-matching.json"
        external_suffix = ".mp20-scale.score.json" if key=="energy" and local_run=="mp20-to-wbm" else suffix
        ids,external = series(folder/("uflow-seed2026-test"+external_suffix),key)
        n = len(ids)
        boot = np.random.default_rng(match.BOOT_SEED).multinomial(n,np.full(n,1/n),size=10000)
        output[key] = {"uflow":match.interval(external,boot),"local_minus_uflow":{}}
        for name in ["drift","flow1","flow16","retrieval"]:
            rows,seeds=[],[]
            for seed in [11,23,37]:
                path = local/(f"{name}-seed{seed}-test"+suffix)
                if path.exists():
                    local_ids,values = series(path,key)
                    assert ids==local_ids
                    rows.append(values);seeds.append(seed)
            if rows:
                output[key]["local_minus_uflow"][name]=dict(seeds=seeds,**match.interval(np.asarray(rows)-external,boot))
    destination = "paired.json" if local_run==cohort else "paired-"+local_run+".json"
    match.write(folder/destination,dict(results=output,bootstrap_case_count=n,case_ids=ids,local_run=local_run,
                design="10000 paired resamples of target IDs after averaging available local fitted seeds. One fixed uFlowCSP checkpoint and inference seed."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort",choices=["mp20","wbm"],nargs="+",default=["mp20","wbm"])
    parser.add_argument("--site",type=Path,nargs="*",default=[])
    parser.add_argument("--paired-only",action="store_true")
    args=parser.parse_args()
    sys.path.extend(map(str,args.site))
    for item in args.cohort:
        if not args.paired_only:
            r=metrics(item)
            print(item,"physical energy score",r["conditional_energy"],"valid",r["contact_valid"],flush=True)
        compare(item)
        if item=="wbm":
            compare(item,"mp20-to-wbm")
