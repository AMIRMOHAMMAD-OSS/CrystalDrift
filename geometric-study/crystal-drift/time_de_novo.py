"""Time warm unconditional binary crystal proposals, including composition sampling."""
import csv
import hashlib
import json
import os
import platform
import statistics
from pathlib import Path
from time import perf_counter_ns

for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'

import numpy as np
import torch

from de_novo import make_prior
from model import CrystalGenerator
from speed import physical_cell


HERE=Path(__file__).resolve().parent
SEEDS=(11,23,37)
BATCHES=(1,256)


@torch.inference_mode()
def generate(model,mean,scale,support,weights,batch,rng):
    # Draw the prior with the same weighted-choice algorithm as the de novo experiment.
    chosen=rng.choice(len(support),batch,p=weights)
    z=np.zeros((batch,8),dtype=np.int64)
    for i,j in enumerate(chosen):
        a,b,na,nb=support[j]
        z[i,:na],z[i,na:na+nb]=a,b
    cell_noise=torch.from_numpy(rng.normal(size=(batch,6)).astype('float32'))
    site_noise=torch.from_numpy(rng.random((batch,8,3)).astype('float32'))
    k,f=model(torch.from_numpy(z),cell_noise,site_noise)
    cell=physical_cell((k*scale+mean).cpu().numpy())
    return z,cell,f.cpu().numpy()


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output=HERE/'de-novo'/'timing'
    output.mkdir(exist_ok=True)
    with np.load(HERE/'data'/'multi-train.npz') as archive:z_train=archive['z']
    start=perf_counter_ns()
    support,weights,observed=make_prior(z_train)
    prior_ms=(perf_counter_ns()-start)*1e-6
    order_rng=np.random.default_rng(71413)
    rows=[];hashes={}
    for seed in SEEDS:
        checkpoint=HERE/'drift-results'/'multi-exact'/f'seed{seed}.pt'
        hashes[str(seed)]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model=CrystalGenerator().eval();model.load_state_dict(state['model'])
        sample_rng={batch:np.random.default_rng(np.random.SeedSequence([71413,seed,batch])) for batch in BATCHES}
        for repeat in range(-3,15):
            for batch in order_rng.permutation(BATCHES).tolist():
                start=perf_counter_ns()
                z,cell,f=generate(model,state['mean'],state['scale'],support,weights,batch,sample_rng[batch])
                elapsed=(perf_counter_ns()-start)*1e-6
                if not (z.shape==(batch,8) and f.shape==(batch,8,3) and cell.shape==(batch,3,3) and
                        np.isfinite(cell).all() and np.isfinite(f).all() and np.all((f>=0)&(f<1)) and
                        np.all(np.linalg.det(cell)>0)):
                    raise ValueError('Invalid decoded batch')
                if repeat>=0:rows.append(dict(seed=seed,batch=batch,repeat=repeat,wall_ms=elapsed))
    with (output/'observations.csv').open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summaries={}
    for batch in BATCHES:
        byseed={str(s):statistics.median(r['wall_ms'] for r in rows if (r['batch'],r['seed'])==(batch,s))
                for s in SEEDS}
        pooled=statistics.median(r['wall_ms'] for r in rows if r['batch']==batch)
        summaries[str(batch)]=dict(three_seed_medians_ms=byseed,pooled_45_median_ms=pooled,
            mean_of_three_seed_medians_ms=statistics.mean(byseed.values()),pooled_median_ms_per_proposal=pooled/batch)
    record=dict(measurement='From prebuilt discrete prior and warm CPU model to host elemental numbers, physical lattice matrix and all fractional coordinates. Includes weighted choice, species tensor construction, independent noise, one network forward, decoding and host materialization.',
        excluded='Disk reads, checkpoint loading, prior fitting, validation assertions, CIF output, feature calculation and relaxation.',
        prior_build_ms=prior_ms,training_rows=len(z_train),training_cell_compositions=len(observed),prior_support=len(support),
        torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),torch=torch.__version__,
        python=platform.python_version(),numpy=np.__version__,processor=platform.processor(),warmups_per_seed_batch=3,
        measured_repeats_per_seed_batch=15,draw_rng='NumPy default_rng(SeedSequence([71413,fit_seed,batch]))',
        checkpoint_sha256=hashes,summary=summaries)
    (output/'summary.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'prior_build_ms':prior_ms,'summary':summaries},indent=1))


if __name__=='__main__':
    main()

