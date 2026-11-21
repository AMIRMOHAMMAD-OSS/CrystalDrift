"""Balanced-order serial replay. Each method's episode consists of 20 uninterrupted draws."""
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[key]="1"
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
import torch
from timing import BASE,ORIGINAL,timed_service
from model import HERE,ROOT,checkpoint_path,digest,save_npz,write_json

METHODS=("prior","radial","angular","old-drift","flow16","old-flow16")
DST=HERE/"interleaved"


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument("--site",nargs="*",default=[]);a=p.parse_args()
    sys.path.extend(a.site);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest");torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    DST.mkdir(exist_ok=True);(DST/"shards").mkdir(exist_ok=True)
    with np.load(ROOT/"revision/full/mp20/cohort.npz") as d:ids,z=d["ids"][:128],d["z"][:128]
    order=np.zeros((len(ids),len(METHODS)),dtype=int);rng=np.random.default_rng(871155)
    for start in range(0,len(ids),len(METHODS)):
        base=rng.permutation(len(METHODS));rows=rng.permutation(len(METHODS))
        for j in range(min(len(ids)-start,len(METHODS))):order[start+j]=np.roll(base,rows[j])
    start_unix=time.time()
    with tempfile.TemporaryDirectory() as tmp:
        services={m:timed_service(m,z,None,Path(tmp)) for m in METHODS}
        for m in METHODS:
            r=np.random.default_rng(31428)
            for _ in range(5):services[m](z[:1],r)
        torch.cuda.synchronize()
        for row in range(len(ids)):
            sh=DST/"shards"/f"{row:03}.npz"
            if sh.exists():continue
            k=np.empty((len(METHODS),20,6));frac=np.empty((len(METHODS),20,20,3));ms=np.empty((len(METHODS),20))
            start_times=[];load=[];end_times=[]
            for j in order[row]:
                # Prior, radial, angular and matched flow receive identical native physical seeds.
                r=np.random.default_rng(np.random.SeedSequence([893101,row]))
                torch.manual_seed(663901+row)
                torch.cuda.synchronize();start_times.append(time.time());load.append(os.getloadavg()[0]);arr=[]
                tic=time.perf_counter_ns()
                for b in range(20):
                    out=services[METHODS[j]](z[row:row+1],r);stamp=time.perf_counter_ns()
                    arr.append((out,stamp))
                end_times.append(time.time())
                for b,(out,stamp) in enumerate(arr):
                    k[j,b]=out[4][0];frac[j,b]=out[2][0];ms[j,b]=(stamp-tic)*1e-6
            save_npz(sh,id=ids[row],methods=METHODS,order=order[row],elapsed_ms=ms,k=k,f=frac,
                     start_unix=start_times,end_unix=end_times,linux_one_minute_load=load)
            if (row+1)%16==0:print(row+1,"balanced timing targets",flush=True)
    allrows=[]
    for i in range(len(ids)):
        with np.load(DST/"shards"/f"{i:03}.npz") as d:
            assert str(d["id"])==str(ids[i]);np.testing.assert_equal(d["order"],order[i])
            allrows.append(tuple(d[field] for field in ("k","f","elapsed_ms")))
    for j,m in enumerate(METHODS):
        k,f,ms=[np.stack([item[col][j] for item in allrows]) for col in range(3)]
        save_npz(DST/f"{m}-episodes.npz",ids=ids,z=z,k=k,f=f,elapsed_ms=ms)
    write_json(DST/"settings.json",dict(methods=METHODS,targets=128,draw_cap=20,order=order.tolist(),order_seed=871155,
        first_start_unix=start_unix,last_end_unix=time.time(),device=torch.cuda.get_device_name(),
        torch=torch.__version__,cpu_threads=1,tf32=False,five_warmups_per_method=True,
        direct_timing="One target-method stream of 20 consecutive calls; balanced random method ordering per target block",
        source_sha256=digest(Path(__file__)),state_sha256={m:json.loads((HERE/f"gpu/{m}-settings.json").read_text()).get("checkpoint_sha256") or digest(HERE/"prior.npz") for m in METHODS}))

if __name__=="__main__":main()
