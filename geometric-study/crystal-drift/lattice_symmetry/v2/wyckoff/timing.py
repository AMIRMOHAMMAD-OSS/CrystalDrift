"""Matched CPU clocks for template proposals ending at decoded native host arrays."""
import json
import os
import platform
from time import perf_counter
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[v]="1"
import numpy as np
import torch
from charts import HERE
from geometry import cohort
from prior import Prior,records
from evaluate import model_and_hash
from train_template import apply,draw
from diffpp import construct,runner

def main():
    torch.set_num_threads(1);rows=cohort("test")[:16];assert all("error" not in r for r in rows)
    # IDs are available to the clock only for record keeping. Methods get two symbolic inputs.
    requests=[{k:r[k] for k in ("group","template")} for r in rows]
    prior=Prior(records());ext,ts=runner(construct("cpu"),20,"cpu");out={}
    for ai,arm in enumerate(["prior","angular-A","angular-C","orbit-C","DiffCSPpp-S20"]):
        if arm.startswith("Diff"):service=ext
        else:
            model,_=model_and_hash(arm)
            def service(batch,seed):
                rng=np.random.default_rng(seed)
                seeds=[draw(r,prior,rng) for r in batch]
                return seeds if model is None else apply(model,seeds)
        for q in range(2):service(requests[:1],878+q)
        lat=[];sizes=[]
        for i,row in enumerate(requests):
            tick=perf_counter();x=service([row],70201+i)
            lat.append(perf_counter()-tick);sizes.append([len(r[2]) for r in x])
        tick=perf_counter();result=service(requests,70955);bt=perf_counter()-tick
        out[arm]=dict(serial_ms=(1000*np.array(lat)).tolist(),median_ms=float(1000*np.median(lat)),batch16_s=bt,
                      throughput=16/bt,counts=sizes,ids=[r["id"] for r in rows])
        print(arm,out[arm]["median_ms"],bt,flush=True)
        (HERE/"template-timing.json").write_text(json.dumps(dict(torch=torch.__version__,device="Intel Core i7-12700H CPU",precision="FP32",
            threads=1,other_cpu_jobs=True,serial_warmups=2,batch_size=16,methods=out),indent=1)+"\n")
if __name__=="__main__":main()
