"""Single-process sequential GPU research matrix; every completed fit is resumable."""
import argparse,gc,json,os,shutil,sys,time
from pathlib import Path
if os.environ.get('CRYTALDRIFT_NUMBA_SITE'):sys.path.append(os.environ['CRYTALDRIFT_NUMBA_SITE'])
import numpy as np
import torch
from .train import one_fit
from .sampler import HERE

def run(tier,arm,affinity='laplace',seed=11,passes=1,batch=16,profile=False,local=False,published=False,commonorder=False):
    a=argparse.Namespace(tier=tier,arm=arm,affinity=affinity,seed=seed,passes=passes,
        batch=batch,lr=.0003,device='cuda',local=local,profile=profile,published=published,separate_time_rng=commonorder)
    one_fit(a)
    gc.collect();torch.cuda.empty_cache()

def wait(path):
    while not path.exists():time.sleep(5)

def note(name,data):
    p=HERE/name;t=p.with_suffix('.building');t.write_text(json.dumps(data,indent=2));os.replace(t,p)

def slot(start,end):
    if not (HERE/start).exists():
        note(start,dict(utc=time.time(),cuda='released',reason='exclusive comparator or same-hardware timing window'))
    if not (HERE/end).exists():
        print('GPU/context released for',start,'resume after',end,flush=True)
        raise SystemExit(75)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--takeover',action='store_true');p.add_argument('--coordinated',action='store_true')
    a=p.parse_args()
    if a.takeover:wait(HERE/'takeover.completed')
    for tier,arm in [('small','angular'),('medium','angular'),('small','contacts'),('small','reciprocal'),
                     ('small','meanflow'),('medium','meanflow')]:
        run(tier,arm)
    if not a.coordinated and not (HERE/'promotion.json').exists():
        from .select_weights import process
        from .sampler import Sampler
        process(Sampler(),False)
    wait(HERE/'promotion.json')
    winner=json.loads((HERE/'promotion.json').read_text())['arm']
    run('medium',winner)
    if a.coordinated:slot('uflow-exclusive.json','uflow-finished.json')
    for tier in ('small','medium'):run(tier,'meanflow',commonorder=True)
    for seed in (23,37):
        for tier in ('small','medium'):
            for arm in ('angular',winner):run(tier,arm,seed=seed)
    note('primary-replicates-complete.json',dict(utc=time.time(),winner=winner))
    for arm in ('angular','contacts'):run('small',arm,affinity='gaussian')
    for arm in ('pair','center0','variance'):run('small',arm)
    for tier,arm in [('small','angular'),('medium','angular'),('medium',winner),('medium','meanflow')]:
        run(tier,arm,passes=8,commonorder=(arm=='meanflow'))
    from .long_selection import cycle
    from .sampler import Sampler
    if len(list((HERE/'fits').glob('*-8pass/confirmation.json')))<4:
        service=Sampler();cycle(service);del service
    if a.coordinated:slot('timing-start.json','timing-finished.json')
    run('tiny','angular',published=True);run('small','angular',published=True)
    run('small','angular',local=True)
    run('small','radial4')
    run('large','angular',batch=16,profile=True)
    # Two deployable weight sets plus old and new full Adam checkpoints at the write boundary.
    scratch=os.getenv('CRYTALDRIFT_SCRATCH')
    destination=Path(scratch) if scratch else HERE
    destination.mkdir(parents=True,exist_ok=True)
    disk=shutil.disk_usage(destination).free
    large_bytes=8*4*15_011_747+4*2**20
    if disk>large_bytes:
        run('large','angular',batch=16)
    else:note('large-storage-gate.json',dict(required_free_MiB=large_bytes/2**20,observed_free_MiB=disk/2**20,
              reason='capacity for checkpoints, optimizer, atomic replacement and concurrent receipts'))
    available=torch.cuda.mem_get_info()[0]
    minimum=135_992_027*4*4 # weights, gradients and two FP32 Adam moments
    if available>minimum:
        from .capacity_gate import strongest,gate
        arm,candidates=strongest()
        try:
            run('reference',arm,batch=2,profile=True)
            prior_gate=HERE/'reference-capacity.json'
            resume_receipt=HERE/'fits'/f'reference-{arm}-laplace-11'/'resume-location.json'
            decision=json.loads(prior_gate.read_text()) if prior_gate.exists() else {}
            if not (decision.get('supported') and decision.get('arm')==arm and resume_receipt.exists()):
                decision=gate(arm)
                note('reference-capacity.json',decision)
            if decision['supported']:run('reference',arm,batch=2)
        except torch.OutOfMemoryError as exc:
            gc.collect();torch.cuda.empty_cache()
            note('reference-vram-gate.json',dict(free_MiB=available/2**20,error=str(exc)))
    else:note('reference-vram-gate.json',dict(free_MiB=available/2**20,min_parameter_and_optimizer_MiB=minimum/2**20))
    if not a.coordinated:
        from .select_weights import process
        from .long_selection import cycle
        from .sampler import Sampler
        service=Sampler();process(service,True);cycle(service)
    note('window-complete.json',dict(utc=time.time(),selected_descriptor=winner))

