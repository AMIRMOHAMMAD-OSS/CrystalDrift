"""Unified single-process CPU proposal and wall-clock harness for the 128 binary cases."""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "3"

import numpy as np
import psutil
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
import data
import speed
from standard_match import order


def json_write(path, content):
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf8")


class Service:
    def __init__(self, args):
        self.args = args
        with np.load(ROOT/"data/multi-test.npz") as test:
            idx = order("multi", test["ids"])
            self.ids, self.z = test["ids"][idx], test["z"][idx]
        with np.load(ROOT/"data/multi-train.npz") as train:
            self.train = {name: train[name] for name in ("k", "f", "z")}
        self.inputs = self.z.astype(np.int64)
        self.donors = speed.donors(self.train["z"], self.z)
        self.models = {}
        self.reset(2026)

    def reset(self, seed):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

    def load(self, method, seed):
        if method == "retrieval":
            return
        if method == "uflow":
            if method in self.models:
                return
            with Path(self.args.weights).open("rb") as fp:
                digest=hashlib.sha256()
                for block in iter(lambda:fp.read(1024*1024),b""):
                    digest.update(block)
            assert digest.hexdigest()=="e1854363d9e78044d1202879e40a918ab50831cb1e08eb686cb53aeacaa85673"
            sys.path.extend(str(Path(p).resolve()) for p in self.args.site)
            import timm
            from safetensors import safe_open
            import uflowcsp_replay
            assert timm.__version__ == "1.0.21"
            self.source_dir = tempfile.TemporaryDirectory()
            uflowcsp_replay.stage_sources(Path(self.source_dir.name))
            from src.models.denoisers.atomic_features import build_atomic_feature_table
            from src.models.denoisers.meanflow_sit_raw_formula_atomwise_system import (
                MeanFlowSiTRawFormulaAtomwiseSystem, build_sg_to_crystal_system)
            from src.models.meanflow_transport import MeanFlowTransport
            with torch.device("meta"):
                model = MeanFlowSiTRawFormulaAtomwiseSystem(
                    input_size=100, atom_type_vocab_size=120, hidden_size=768, depth=12,
                    num_heads=12, predict_atom_types=False, use_spacegroup_conditioning=False,
                    use_formula_embedding=True, use_atomwise_features=True, use_crystal_system_embedding=True)
            with safe_open(self.args.weights, framework="pt", device="cpu") as weights:
                state = {k: weights.get_tensor(k) for k in weights.keys()}
            model.load_state_dict(state, strict=True, assign=True)
            model.atomic_feature_table = build_atomic_feature_table(120)
            model.sg_to_crystal_system = build_sg_to_crystal_system(230)
            assert model.atomic_feature_table.abs().sum() > 0
            self.models[method] = model.eval()
            self.transport = MeanFlowTransport(torus_coords=False, lattice_per_structure=False,
                proj_lattice=False, proj_coords_anchor=False, proj_coords_rotavg=False)
            self.ordered = {}
            with (ROOT/"uflowcsp-results/multi-crosswalk.csv").open() as fp:
                rows=list(csv.DictReader(fp))
                assert len(rows)==len(self.ids)==128
                for mid,z,row in zip(self.ids,self.z,rows):
                    assert mid == row["id"] and row["in_train_ids"] == "False" and row["in_test_ids"] == "True"
                    atoms = list(map(int,row["official_active_z"].split()))
                    assert sorted(atoms) == z[z>0].tolist()
                    self.ordered[tuple(z)] = atoms+[0]*(len(z)-len(atoms))
        else:
            key = ("drift" if method=="exact" else "flow", seed)
            if key not in self.models:
                self.models[key] = speed.load_model("multi", *key, "cpu")

    @torch.inference_mode()
    def propose(self, method, seed, atoms):
        batch = len(atoms)
        if method == "uflow":
            selected = np.asarray([self.ordered[tuple(z)] for z in atoms], dtype=np.int64)
            types = torch.zeros((batch,100),dtype=torch.long)
            types[:,:8] = torch.from_numpy(selected)
            valid = types.ne(0)
            cond = dict(atom_types=types,mask=valid,spacegroup=torch.zeros(batch,dtype=torch.long),
                        generate_atom_types=False)
            noise = self.transport.sample_prior(batch,100,12,"cpu")
            y = self.transport.sample(self.models[method], noise, cond, num_steps=1)
            active = valid[:,:8,None]
            C = (y[:,:8,3:].masked_fill(~active,0).sum(1)/active.sum(1)).reshape(-1,3,3)
            frac = (y[:,:8,:3]@torch.linalg.pinv(C)).remainder(1).masked_fill(~active,0).numpy()
            restore = np.argsort(np.where(selected==0,120,selected),kind="stable",axis=1)
            f = np.take_along_axis(frac,restore[:,:,None],axis=1)
            C = C.double().numpy()
            k = data.pack_gram(C@C.swapaxes(-1,-2))
        elif method == "retrieval":
            pools = [self.donors[tuple(z)] for z in atoms]
            which = [p[int(x*len(p))] for p,x in zip(pools,self.rng.random(batch))]
            k, f = self.train["k"][which], self.train["f"][which]
        else:
            key = ("drift" if method=="exact" else "flow", seed)
            model, mean, scale = self.models[key]
            z = torch.from_numpy(atoms)
            k = torch.randn(batch,6)
            f = torch.rand(batch,8,3)
            if method == "exact":
                k,f = model(z,k,f)
            else:
                steps = int(method[4:])
                c = model.prepare(z)
                f = f*c[0]
                for j in range(steps):
                    t = torch.full((batch,1),j/steps)
                    dk,df = model(c,k,f,t)
                    k = k+dk/steps
                    f = (f+df/(math.sqrt(12)*steps)).remainder(1)*c[0]
            k,f = (k*scale+mean).numpy(),f.numpy()
        # Same end point for all arms: an owned host representation with species,
        # canonical lattice, wrapped fractional positions, and Cartesian sites.
        f = np.where(atoms[:,:,None]>0,np.remainder(f,1),0).astype(np.float64,copy=False)
        cell = speed.physical_cell(k)
        return atoms.copy(),cell,f,f@cell,k


def extend(service):
    with np.load(ROOT/"uflowcsp-results/multi-seed2026-euler1.npz") as src:
        prefix = {name:src[name] for name in ("ids","z","k","f")}
    np.testing.assert_equal(prefix["ids"],service.ids)
    np.testing.assert_equal(prefix["z"],service.z)
    service.load("uflow",2026)
    service.reset(20261121)
    repeated = np.repeat(service.inputs,15,axis=0)
    cells,sites = [],[]
    tic = time.perf_counter()
    for j in range(0,len(repeated),12):
        _,_,f,_,k = service.propose("uflow",2026,repeated[j:j+12])
        cells.append(k)
        sites.append(f)
        if j%240 == 0: print("suffix",j,time.perf_counter()-tic,flush=True)
    k = np.concatenate((prefix["k"],np.concatenate(cells).reshape(128,15,6)),axis=1)
    f = np.concatenate((prefix["f"],np.concatenate(sites).reshape(128,15,8,3).astype(prefix["f"].dtype)),axis=1)
    for key,arr in [("k",k),("f",f)]:
        assert arr[:,:5].tobytes() == prefix[key].tobytes() and np.isfinite(arr).all()
    np.savez_compressed(OUT/"uflow-20.npz",ids=service.ids,z=service.z,k=k,f=f)
    json_write(OUT/"uflow-extension.json",dict(prefix_seed=2026,independent_suffix_seed=20261121,
        prefix_count=5,suffix_count=15,batch=12,seconds=time.perf_counter()-tic,
        prefix_sha256=hashlib.sha256((ROOT/"uflowcsp-results/multi-seed2026-euler1.npz").read_bytes()).hexdigest(),
        weight_sha256=hashlib.sha256(Path(service.args.weights).read_bytes()).hexdigest(),
        pinned_commit="e87af8483e6d45b0217b4ede430862e992d2b1d0"))


def measure(service,method,seed,mode,batch):
    tag = f"{method}-{seed}"
    path = OUT/(("episode-" if mode=="episode" else "ten-thousand-")+tag+".npz")
    if path.exists(): return
    service.load(method,seed)
    # Identical warm-up count, sample order, batch sizes and precision across arms.
    b = 1 if mode=="episode" else batch
    warm = service.inputs[np.arange(b)%128]
    service.reset(41000+seed)
    for _ in range(5): service.propose(method,seed,warm)
    service.reset((410000 if mode=="episode" else 510000)+seed)
    tic = time.perf_counter()
    if mode=="episode":
        k = np.empty((128,20,6)); f = np.empty((128,20,8,3)); times = np.empty((128,20))
        for i in range(128):
            atoms = service.inputs[i:i+1]
            buf = []
            start = time.perf_counter_ns()
            for j in range(20):
                result = service.propose(method,seed,atoms)
                done = time.perf_counter_ns()
                buf.append((result,done))
            for j,(result,done) in enumerate(buf):
                k[i,j],f[i,j],times[i,j] = result[4][0],result[2][0],(done-start)*1e-6
            if i%32==0: print(tag,mode,i,round(time.perf_counter()-tic,1),flush=True)
        assert np.isfinite(k).all() and np.isfinite(f).all() and (np.diff(times,axis=1)>0).all()
        np.savez_compressed(path,k=k,f=f,ids=service.ids,z=service.z,elapsed_ms=times)
    else:
        ix = np.arange(10000)%128
        durations,counts = [],[]
        start = time.perf_counter_ns()
        for j in range(0,10000,batch):
            before = time.perf_counter_ns()
            result = service.propose(method,seed,service.inputs[ix[j:j+batch]])
            durations.append((time.perf_counter_ns()-before)*1e-6)
            counts.append(len(result[0]))
            if j%1200==0: print(tag,mode,j,round(time.perf_counter()-tic,1),flush=True)
        elapsed = (time.perf_counter_ns()-start)*1e-9
        assert sum(counts)==10000 and np.isfinite(result[3]).all()
        np.savez_compressed(path,elapsed_s=elapsed,batch_ms=durations,counts=counts,ids=service.ids[ix])
    print(tag,mode,"finished",round(time.perf_counter()-tic,1),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes",nargs="+",choices=["extend","episode","ten-thousand"],required=True)
    parser.add_argument("--methods",nargs="+",default=["exact","flow1","flow16","retrieval","uflow"])
    parser.add_argument("--weights",type=Path)
    parser.add_argument("--site",nargs="*",default=[])
    parser.add_argument("--cores",nargs="+",type=int,default=[0,2,4])
    parser.add_argument("--batch",type=int,default=12)
    args=parser.parse_args()
    affinity=getattr(psutil.Process(),"cpu_affinity",None)
    if affinity:
        affinity(args.cores)
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    from threadpoolctl import threadpool_limits,threadpool_info
    with threadpool_limits(limits=3):
        service=Service(args)
        settings=dict(processor=platform.processor(),os=platform.platform(),torch=torch.__version__,
            numpy=np.__version__,threads=torch.get_num_threads(),interop=torch.get_num_interop_threads(),
            affinity=affinity() if affinity else "OS does not expose CPU affinity",batch=args.batch,machine=platform.machine(),
            precision="FP32 neural forward; float64 canonical physical output",warmups=5,
            rate_workload=10000,episode_draws=20,threadpools=threadpool_info())
        json_write(OUT/"settings.json",settings)
        for mode in args.modes:
            if mode=="extend":
                extend(service)
                continue
            for method in args.methods:
                seeds = [2026] if method=="uflow" else ([11,23,37] if mode=="episode" else [11])
                for seed in seeds: measure(service,method,seed,mode,args.batch)
