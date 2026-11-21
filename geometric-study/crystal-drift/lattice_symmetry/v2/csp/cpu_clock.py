"""One-thread CPU timing sensitivity for the six new prior/model arms."""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[key]="1"

import csv
import json
import platform
import time
import numpy as np
import torch
from model import ARMS,HERE,ROOT,checkpoint_path,digest,load_service,save_npz,write_json
from timing import protocol


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    with np.load(ROOT/"revision/full/mp20/cohort.npz") as c:
        ids,z=c["ids"][:128],c["z"][:128]
    destination=HERE/"cpu"
    destination.mkdir(exist_ok=True)
    methods=("prior",*ARMS)
    def wrap(propose):
        def complete(compositions,rng):
            k,f=propose(compositions,rng)
            f=np.where(compositions[:,:,None]>0,np.remainder(f,1),0.).astype(np.float64)
            cell,f,cart=protocol.decode(k,f)
            return compositions.copy(),cell,f,cart,k
        return complete
    services=[wrap(load_service(m,11,"cpu")) for m in methods]
    for batch in (1,24):
        for fun in services:
            rng=np.random.default_rng(13651)
            for i in range(5):fun(z[:batch],rng)
    order_rng=np.random.default_rng(867418)
    orders=[]
    while len(orders)<len(ids):
        base=order_rng.permutation(len(methods))
        rotations=order_rng.permutation(len(methods))
        orders.extend([np.roll(base,offset) for offset in rotations])
    orders=np.asarray(orders[:len(ids)])
    lat=np.empty((len(ids),len(methods),5))
    k_pool=np.empty((*lat.shape,6));f_pool=np.empty((*lat.shape,20,3))
    for target,order in enumerate(orders):
        for arm in order:
            rng=np.random.default_rng(np.random.SeedSequence([554011,target,803]))
            start=time.perf_counter_ns()
            for draw in range(5):
                z_out,cell,f,cart,k=services[arm](z[target:target+1],rng)
                finish=time.perf_counter_ns()
                lat[target,arm,draw]=(finish-start)*1e-6
                start=finish
                k_pool[target,arm,draw]=k[0];f_pool[target,arm,draw]=f[0]
    save_npz(destination/"batch-one.npz",ids=ids,z=z,methods=methods,order=orders,
             ms=lat,k=k_pool,f=f_pool)
    index=np.arange(10000)%len(ids);rows=[];batch_order=order_rng.permutation(len(methods))
    for arm in batch_order:
        m=methods[arm];fn=services[arm];rng=np.random.default_rng(661011)
        durations=[];counts=[]
        start_unix=time.time();tick=time.perf_counter_ns()
        for i in range(0,10000,24):
            a=time.perf_counter_ns()
            result=fn(z[index[i:i+24]],rng)
            durations.append((time.perf_counter_ns()-a)*1.e-6)
            counts.append(len(result[0]))
        seconds=(time.perf_counter_ns()-tick)*1.e-9
        assert sum(counts)==10000 and np.isfinite(result[3]).all()
        save_npz(destination/f"{m}-10000.npz",ids=ids[index],counts=counts,batch_ms=durations,
                 wall_seconds=seconds,start_unix=start_unix,end_unix=time.time())
        rows.append(dict(method=m,b1_median_ms=np.median(lat[:,arm]),seconds_per_10000=seconds,
                         proposals_per_second=10000/seconds,checkpoint_sha256=digest(checkpoint_path(m,11))))
        print(m,rows[-1]["b1_median_ms"],seconds,flush=True)
    with (destination/"timing.csv").open("w",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(destination/"settings.json",dict(machine="Intel Core i7-12700H",platform=platform.platform(),torch=torch.__version__,
       threads=1,train_seed=11,precision="float32 model, float64 decoded arrays",warmups=5,methods=methods,draws_per_target=5,
       b1_episodes=128,b1_order="balanced order in six-target blocks",batch_size=24,throughput_requests=10000,
       throughput_order=[methods[i] for i in batch_order],host_contention="Concurrent public StructureMatcher CPU workers",
       output="host species, 3x3 canonical cell, wrapped fractional and Cartesian sites; no scoring or relaxation"))

if __name__=="__main__":main()
