"""Synchronized warm timings for the compact models' 128-target multielement GPU cohort."""
import argparse,json,os,time
from pathlib import Path
for var in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[var]="1"
import numpy as np
import torch
from crystalflow import ROOT,HERE,load_model,EncoderCache,sample
def decode(z,k,f):
    s=np.zeros((len(z),3,3),float)
    s[:,np.arange(3),np.arange(3)]=k[:,:3]
    for j,(a,b) in enumerate(((0,1),(0,2),(1,2))):s[:,a,b]=s[:,b,a]=k[:,3+j]/np.sqrt(2)
    w,v=np.linalg.eigh(s)
    cell=(v*np.exp(w)[:,None,:])@v.swapaxes(-1,-2)
    return z.copy(),cell,f.copy(),f@cell
def main(a):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dest=HERE/f"timing-s{a.steps}";dest.mkdir(exist_ok=True)
    with np.load(ROOT/"revision/full/mp20/cohort.npz") as d:ids=d["ids"][:128];z=d["z"][:128]
    model,emb,build=load_model(a.source,a.weights,"cuda"); cache=EncoderCache(model)
    def call(rows,seed):
        k,f=sample(model,emb,build,cache,rows,a.steps,"cuda",seed)
        output=decode(rows,k,f);torch.cuda.synchronize()
        return k,f
    for batch in (1,24,16):
        for i in range(5):call(z[:batch],81001+i)
    ks=[];fs=[];ts=[]
    torch.manual_seed(554011)
    for i in range(len(z)):
        row_k=[];row_f=[];row_t=[];torch.cuda.synchronize();start=time.perf_counter()
        for j in range(20):
            k,f=call(z[i:i+1],None);row_t.append(time.perf_counter()-start)
            row_k.append(k[0]);row_f.append(f[0])
        ks.append(row_k);fs.append(row_f);ts.append(row_t)
    np.savez_compressed(dest/"serial.npz",ids=ids,z=z,k=ks,f=fs,elapsed_ms=np.array(ts)*1000)
    ids10=np.arange(10000)%128;batches=[]
    torch.manual_seed(661011);start=time.perf_counter()
    for lo in range(0,len(ids10),24):
        tic=time.perf_counter();call(z[ids10[lo:lo+24]],None);batches.append(time.perf_counter()-tic)
    sec=time.perf_counter()-start
    np.savez_compressed(dest/"batch.npz",ids=ids[ids10],batch_seconds=batches,wall_seconds=sec)
    summary=dict(nfe=a.steps,median_b1_ms=float(np.median(np.diff(ts,axis=1,prepend=0))*1000),
        batch=24,batch_total_seconds=sec,per_second=10000/sec,gpu=torch.cuda.get_device_name(),threads=1,
        torch=torch.__version__,precision="FP32 highest, TF32 disabled",cohort=128)
    (dest/"runtime.json").write_text(json.dumps(summary,indent=2)+"\n");print(summary,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",type=Path,required=True)
    p.add_argument("--weights",type=Path,required=True);p.add_argument("--steps",type=int,default=5);main(p.parse_args())
