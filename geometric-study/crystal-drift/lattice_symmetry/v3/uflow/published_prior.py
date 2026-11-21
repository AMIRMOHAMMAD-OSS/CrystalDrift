"""Decode exactly the two historical ordered Gaussian streams behind the released EMA comparison."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from runtime import HERE,ROOT,MeanFlowTransport,atomic_json,formula_tokens,model,released

sys.path.insert(0,str(ROOT))
from data import unpack_gram

OLD=ROOT/'revision/full/external/mp20'
SOURCE=ROOT/'scaled/external/mp20'
POOL=HERE/'predictions/published-prior/frozen/test'
PREFIX=HERE/'published-prefix.npz'

def original_flow():
    return MeanFlowTransport(torus_coords=False,lattice_per_structure=False,proj_lattice=False,
                             proj_coords_anchor=False,proj_coords_rotavg=False)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def noise_digest(x):
    return hashlib.sha256(memoryview(x.numpy()).cast('B')).hexdigest()

def decode(x,z):
    types=torch.as_tensor(formula_tokens(z),device=x.device)
    k,f=released.structures(x,types,z)
    return k,f

def probe(model_,flow,x,z,ref_k,ref_f):
    types=torch.as_tensor(formula_tokens(z),device=x.device)
    cond=dict(atom_types=types,mask=types!=0,spacegroup=torch.zeros(len(z),device=x.device,dtype=torch.long),
              generate_atom_types=False)
    y=flow.sample(model_,x,cond,num_steps=1)
    k,f=decode(y,z)
    delta=f-ref_f;delta-=np.round(delta)
    discrepancy=dict(k_max=float(np.abs(k-ref_k).max()),fractional_max=float(np.abs(delta).max()),
                     exact_coordinates=bool(np.array_equal(f,ref_f)),cases=len(z),noise_sha256=noise_digest(x.cpu()))
    assert discrepancy['k_max']<1e-4 and discrepancy['fractional_max']<1e-4,discrepancy
    return discrepancy

@torch.inference_mode()
def prefix():
    report=json.loads((SOURCE/'replay.json').read_text())
    assert torch.__version__==report['torch'] and sha(SOURCE/'uflow-seed2026-test.npz')==report['prediction_sha256']
    torch.set_num_threads(report['threads'])
    net=model('released','cpu').eval();flow=original_flow()
    with np.load(SOURCE/'uflow-seed2026-test.npz') as original:
        ids=original['ids'];z=original['z'];ref_k=original['k'].reshape(-1,6);ref_f=original['f'].reshape(-1,20,3)
    ordered=np.repeat(z,report['draws'],axis=0)
    torch.manual_seed(report['seed'])
    cells=[];coords=[];inputs=[];probes=[]
    for start in range(0,len(ordered),report['batch']):
        zz=ordered[start:start+report['batch']]
        x=flow.sample_prior(len(zz),100,12,'cpu')
        if start==0 or start+len(zz)==len(ordered):
            probes.append(dict(first=start,**probe(net,flow,x,zz,ref_k[start:start+len(zz)],ref_f[start:start+len(zz)])))
        inputs.append(x);k,f=decode(x,zz);cells.append(np.linalg.cholesky(unpack_gram(k)));coords.append(f)
    total=torch.cat(inputs)
    np.savez_compressed(PREFIX,ids=ids,Z=z,L=np.concatenate(cells).reshape(len(z),5,3,3),
                        F=np.concatenate(coords).reshape(len(z),5,20,3),noise_sha256=noise_digest(total),
                        per_input_sha=np.array([noise_digest(one) for one in total]).reshape(len(z),5))
    atomic_json(HERE/'published-prefix-replay.json',dict(original=report,probes=probes,noise_sha256=noise_digest(total),
                                                        prior_output_sha256=sha(PREFIX)))
    print('legacy CPU prefix',probes,flush=True)

@torch.inference_mode()
def gpu(handoff=False,max_seconds=None,final_handoff=False,prepare_cpu=False):
    label='uflow-source-final' if final_handoff else 'uflow-source'
    gate=HERE.parent/'models'/(label+'-exclusive.json')
    if (handoff or final_handoff) and not gate.is_file() and not prepare_cpu:
        raise RuntimeError('Source-probe GPU handover is pending')
    report=json.loads((OLD/'manifest.json').read_text())
    assert torch.__version__==report['torch'] and report['batch_size']==24
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    flow=original_flow()
    if prepare_cpu:
        if not final_handoff:raise ValueError('CPU prestaging expects an explicit final handover')
        net=model('released','cpu').eval()
        print('Published weights ready on CPU; waiting for the final GPU handover',flush=True)
        while not gate.is_file():time.sleep(3)
        net=net.to('cuda')
    else:net=model('released','cuda').eval()
    with np.load(OLD/'targets.npz') as d:ids=d['ids'];z=d['z']
    assert len(ids)==9046
    POOL.mkdir(parents=True,exist_ok=True)
    records=HERE/'published-progress';records.mkdir(exist_ok=True)
    tic=time.monotonic()
    probes={};span=[(0,128)]+[(i,min(i+256,len(ids))) for i in range(128,len(ids),256)]
    for lo,hi in span:
        target=POOL/f'{lo:05}-{hi:05}.npz'
        receipt=records/(target.stem+'.json')
        if target.is_file() and receipt.is_file():
            saved=json.loads(receipt.read_text())
            assert sha(target)==saved['sha256'] and saved['count']==hi-lo
            probes.update(saved['probes']);continue
        if max_seconds and time.monotonic()-tic>=max_seconds:
            if handoff or final_handoff:
                atomic_json(HERE.parent/'models'/(label+'-finished.json'),
                            dict(utc=datetime.now(timezone.utc).isoformat(),complete=False,completed_targets=lo))
            print('Parked at atomic tile boundary',lo,flush=True)
            return
        tile_probes={}
        L=np.empty((hi-lo,20,3,3));F=np.empty((hi-lo,20,20,3),dtype=np.float32)
        hashes=np.empty((hi-lo,20),dtype='S64')
        for j in range(20):
            if lo==0 and j<5:
                with np.load(PREFIX) as p:
                    np.testing.assert_array_equal(ids[lo:hi],p['ids']);np.testing.assert_array_equal(z[lo:hi],p['Z'])
                    L[:,j]=p['L'][:,j];F[:,j]=p['F'][:,j];hashes[:,j]=p['per_input_sha'][:,j]
                continue
            for i in range(lo,hi,24):
                end=min(i+24,hi);rows=slice(i-lo,end-lo);zz=z[i:end]
                torch.manual_seed(2026000000+1000000*j+i)
                x=flow.sample_prior(end-i,100,12,'cpu')
                hashes[rows,j]=[noise_digest(one) for one in x]
                x=x.to('cuda')
                if (j,lo,i) in ((0,128,128),(19,8832,8832)):
                    with np.load(OLD/f'draw{j:02}-cases{lo:05}-{hi:05}.npz') as ref:
                        np.testing.assert_array_equal(ref['ids'][rows],ids[i:end])
                        tile_probes[f'{j}/{i}']=probe(net,flow,x,zz,ref['k'][rows],ref['f'][rows])
                k,f=decode(x,zz)
                L[rows,j]=np.linalg.cholesky(unpack_gram(k));F[rows,j]=f
        temp=target.with_suffix('.part.npz')
        np.savez_compressed(temp,ids=ids[lo:hi],Z=z[lo:hi],L=L,F=F,per_input_sha=hashes)
        temp.replace(target)
        atomic_json(receipt,dict(count=hi-lo,sha256=sha(target),probes=tile_probes))
        probes.update(tile_probes)
        if lo==0 or hi==len(ids) or (lo-128)%2048==0:print('zero-call archived stream',lo,hi,flush=True)
    assert len(probes)==2
    atomic_json(POOL/'ready.json',dict(split='test',count=len(ids),draws=20,model_calls=0,
                 schedule='128 x 5 sequential Windows CPU prefix; remaining draws keyed by native tile, draw and minibatch start',
                 released_crosswalk_sha256=sha(OLD/'complete-crosswalk-k20.npz')))
    atomic_json(HERE/'published-gpu-replay.json',dict(environment=report,probes=probes,total=20*len(ids),
                 cpu_prefix_draws=640,cuda_decode_draws=20*len(ids)-640,prefix_sha256=sha(PREFIX)))
    if handoff or final_handoff:
        atomic_json(HERE.parent/'models'/(label+'-finished.json'),
                    dict(utc=datetime.now(timezone.utc).isoformat(),complete=True,draws=len(ids)*20))
    print(probes,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prefix','gpu']);p.add_argument('--handoff',action='store_true')
    p.add_argument('--max-seconds',type=float);p.add_argument('--final-handoff',action='store_true');p.add_argument('--prepare-cpu',action='store_true')
    args=p.parse_args()
    if args.stage=='prefix':prefix()
    else:gpu(args.handoff,args.max_seconds,args.final_handoff,args.prepare_cpu)
