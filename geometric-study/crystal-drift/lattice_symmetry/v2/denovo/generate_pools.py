"""Sample chemistry and raw geometry at fixed fit seed using the same one-thread host boundary."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT),str(ROOT/"expansion/denovo")]
from generate import cells, setup
from prior import CompositionPrior
from scaled.family import draw
from lattice_symmetry.v2.csp import model as fitted


def service(name):
    if name in ("angular","radial"):
        return fitted.load_service(name,11,"cpu"),fitted.checkpoint_path(name,11)
    net,state,_,path=setup("drift","cpu")
    @torch.inference_mode()
    def propose(z,rng):
        k=rng.standard_normal((len(z),6)).astype(np.float32)
        f=rng.random((len(z),20,3),dtype=np.float32)
        k,f=draw(net,"drift",torch.tensor(z,dtype=torch.long),torch.from_numpy(k),torch.from_numpy(f),1)
        return (k*state["scale"]+state["mean"]).numpy(),f.numpy()
    return propose,path


def run(name):
    torch.set_num_threads(1)
    with np.load(ROOT/"scaled/mp20-data/train.npz") as archive:
        prior=CompositionPrior(archive["z"],shrinkage=4.)
    propose,path=service(name)
    # Warm caches with unrelated noise. The measured stream starts from fixed seeds below.
    fresh=np.random.default_rng(112926)
    for _ in range(3):
        z=prior.sample(fresh,16,"learned")
        k,f=propose(z,fresh)
        _=f@cells(k)
    rng=np.random.default_rng(22112026); geom=np.random.default_rng(71831)
    batch=128
    records=[]
    seconds=[]
    for start in range(0,10000,batch):
        begin=perf_counter()
        z=prior.sample(rng,min(batch,10000-start),"learned")
        k,f=propose(z,geom)
        l=cells(k)
        x=f@l
        valid=np.isfinite(k).all(1)&np.isfinite(f).all((1,2))&np.isfinite(x).all((1,2))
        seconds.append(perf_counter()-begin)
        records.append((z,k,f,l,valid))
    z,k,f,l,good=map(np.concatenate,zip(*records))
    fixed=ROOT/"expansion/denovo/pools/learned-drift.npz"
    with np.load(fixed) as original:
        np.testing.assert_array_equal(z,original["z"])
        ids=original["ids"]
        difference=None
        if name=="fixed-basis":
            dk=np.max(np.abs(k-original["k"]))
            df=f-original["f"]
            df=np.max(np.abs(df-np.rint(df)))
            difference=dict(max_chart_abs=float(dk),max_fractional_abs=float(df))
            # FP32 GEMM reduction order can differ between CPU implementations.
            if max(dk,df)>2e-4:raise ValueError(difference)
    out=HERE/"pools"
    out.mkdir(exist_ok=True)
    fitted.save_npz(out/(name+".npz"),ids=ids,z=z,k=k,f=f,lattice=l,valid_output=good,
                    elapsed_by_batch=np.asarray(seconds))
    clock_rng=np.random.default_rng(271119)
    b1=[]
    for i in range(131):
        start=perf_counter()
        z1=prior.sample(clock_rng,1,"learned")
        k1,f1=propose(z1,clock_rng)
        _=f1@cells(k1)
        b1.append(1000*(perf_counter()-start))
    costs=dict(arm=name,checkpoint=path.relative_to(ROOT).as_posix(),checkpoint_sha256=fitted.digest(path),
               batch=batch,elapsed_s=sum(seconds),b1_ms=b1[3:],batch_one_warmups=3,
               train_hash=fitted.digest(ROOT/"scaled/mp20-data/train.npz"),
               compose_reference_sha256=fitted.digest(fixed),draws=len(ids),finite=int(good.sum()),
               torch=str(torch.__version__),numpy=np.__version__,python=platform.python_version(),
               device="CPU",threads=torch.get_num_threads(),geometry_seed=71831,composition_seed=22112026,
               prior_alpha=4,fit_seed=11,matched_fixed_basis=difference,
               boundary="composition RNG, physical prior and site noise, model, host lattice, fractional and Cartesian sites",
               data_sha256=fitted.digest(out/(name+".npz")))
    if name!="fixed-basis":
        costs["physical_seed_prior_sha256"]=fitted.digest(fitted.HERE/"prior.npz")
        selected=json.loads((fitted.HERE/"selection.json").read_text())[name]
        if selected["selected_checkpoint_sha256"]["11"]!=costs["checkpoint_sha256"]:
            raise ValueError("The checkpoint differs from the frozen validation row.")
        fitted.write_json(out/(name+"-selection.json"),selected)
        costs["validation_selection_sha256"]=fitted.digest(out/(name+"-selection.json"))
        costs["validation_selection_scope"]="archived immutable arm-specific record; other arm selections may be pending"
    # The baseline report code interprets pools/{name}.json as a GPU timer.
    # Store the explicitly CPU clock separately.
    fitted.write_json(out/(name+"-cpu.json"),costs)
    print(name,len(ids),round(sum(seconds),3),"CPU s",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("arms",nargs="+",choices=("fixed-basis","radial","angular"))
    for arm in p.parse_args().arms:run(arm)
