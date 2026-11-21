"""Apply the registered validation rule and log both disjoint validation panels."""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import numpy as np
import torch
from .sampler import HERE,V2,V3,ROOT,Sampler,weight_path
from .train import dev,sha
torch.set_num_threads(1)

def digest(path):return sha(path)
def write(path,record):
    p=path.with_suffix('.building');p.write_text(json.dumps(record,indent=2));os.replace(p,path)

def score(source,split='val'):
    exe=os.getenv('MATCHER_PYTHON') or json.loads((V2/'csp/local-runtime.json').read_text())['matcher_executable']
    out=source.with_name(source.stem+'-scores.npz')
    if not out.exists():
        script=str(V3/'evaluation/score.py');src=str(source);dst=str(out)
        if os.name!='nt' and re.match(r'^[A-Za-z]:',exe) and shutil.which('wslpath'):
            def convert(flag,path):return subprocess.check_output(['wslpath','-a',flag,path],text=True).strip()
            exe=convert('-u',exe);script,src,dst=(convert('-w',v) for v in (script,src,dst))
        subprocess.run([exe,script,src,'--split',split,'--output',dst],check=True)
    with np.load(out) as z:
        hit=np.isfinite(z['rms']);native=hit & (np.abs(z['log_volume_ratio'])<=np.log(1.1))
        return dict(hits=int(hit.any(1).sum()),native=int(native.any(1).sum()),n=len(hit),
                    errors=int(z['error'].astype(bool).sum()),score_sha256=digest(out))

def process(s,do_confirm):
    prior=HERE/'fits/prior'
    prior.mkdir(parents=True,exist_ok=True)
    for panel in ('dev','confirm'):
        path=prior/f'{panel}.npz'
        if not path.exists():dev(s,None,path,panel)
        if not (prior/f'{panel}.json').exists():write(prior/f'{panel}.json',score(path))
    for fit in sorted((HERE/'fits').glob('*')):
        if fit==prior or not fit.is_dir():continue
        vals={}
        for name in ('half','full'):
            raw=fit/(name+'-dev.npz')
            if raw.exists():
                vals[name]=score(raw)
        if len(vals)!=2:continue
        sel=fit/'selection.json'
        if not sel.exists():
            pick=max(vals,key=lambda k:vals[k]['hits'])
            write(sel,dict(chosen=pick+'.pt',sha256=digest(fit/(pick+'.pt')),dev=vals,
                            rule='maximum five-draw development hits, earliest tie'))
            print(fit.name,'selected',pick,vals[pick]['hits'],flush=True)
        if do_confirm and not (fit/'confirmation.json').exists():
            src=json.loads(sel.read_text());raw=fit/'frozen-confirm.npz'
            if not raw.exists():dev(s,s.load(fit/src['chosen']),raw,'confirm')
            write(fit/'confirmation.json',score(raw))
    paths=[HERE/f'fits/small-{a}-laplace-11/selection.json' for a in ('contacts','reciprocal')]
    promotion=HERE/'promotion.json'
    if all(p.exists() for p in paths) and not promotion.exists():
        scores=[json.loads(p.read_text()) for p in paths]
        n=[j['dev'][Path(j['chosen']).stem]['hits'] for j in scores]
        write(promotion,dict(arm=['contacts','reciprocal'][int(n[1]>n[0])],contacts_hits=n[0],reciprocal_hits=n[1],
                  source=[digest(p) for p in paths],rule='maximum dev five-draw hits; ties to contacts'))
        print('promoted',promotion.read_text(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--confirm',action='store_true')
    args=p.parse_args();s=Sampler()
    if args.watch and args.confirm and os.name=='nt':
        write(HERE/'windows-confirm-active.json',dict(pid=os.getpid(),started=time.time(),platform='Windows CPU'))
    while True:
        process(s,args.confirm)
        if args.confirm:
            from .long_selection import cycle
            cycle(s)
        if not args.watch or (HERE/'window-complete.json').exists():break
        time.sleep(8)
    if args.watch and args.confirm and os.name=='nt':
        (HERE/'windows-confirm-active.json').unlink(missing_ok=True)

