"""Score complete, resumable target/draw crosswalks using the pinned pymatgen matcher."""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from time import perf_counter

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from importlib.metadata import version
from pymatgen.analysis.structure_matcher import StructureMatcher
from prescreen import frame_possible

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from standard_match import SETTINGS,make_structure
from data import unpack_gram

ARMS=[f'{m}-seed{s}' for m in ('drift','flow1','flow16','retrieval') for s in (11,23,37)]
TILES=HERE/'mp20'/'tiles'
MATCHER=StructureMatcher(**SETTINGS)


def pin():
    try:
        import psutil
        proc=psutil.Process();chosen=[c for c in proc.cpu_affinity() if c not in (0,2,4)]
        if chosen:proc.cpu_affinity(chosen)
    except (ImportError,AttributeError,OSError):pass


def fingerprint():
    return hashlib.sha256(Path(__file__).read_bytes()+(HERE/'prescreen.py').read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def resources():
    pin()
    with np.load(HERE/'mp20/cohort.npz') as d:ref={k:d[k] for k in d.files}
    pools=[]
    for arm in ARMS:
        folder=HERE/'mp20'/arm;info=json.loads((folder/'ready.json').read_text())
        if info['targets']!=len(ref['ids']) or info['draws']!=20:raise ValueError(arm)
        pools.append(tuple(np.load(folder/(k+'.npy'),mmap_mode='r') for k in ('k','f')))
    return ref,pools


def target_context(k,f,z):
    truth=make_structure(k,f,z)
    reduced=MATCHER._get_reduced_structure(truth,True,True)
    coord=np.ascontiguousarray(reduced.frac_coords,dtype=np.float64)
    species=np.ascontiguousarray(reduced.atomic_numbers,dtype=np.int64)
    anchor=int(np.argmin((species[:,None]==species[None,:]).sum(1)))
    g=reduced.lattice.matrix@reduced.lattice.matrix.T
    return truth,reduced,coord,species,anchor,g


def candidate(context,kg,fg,z,gg,verify):
    truth,ref,coord,species,anchor,g=context
    if len(ref)==len(z):
        possible=frame_possible(gg,np.ascontiguousarray(fg[:len(z)],dtype=np.float64),z,
                                g,coord,species,anchor)
    else:possible=True
    rms=np.nan;strict=False;official=False
    if possible:
        pred=make_structure(kg,fg[:len(z)],z)
        small=MATCHER._get_reduced_structure(pred,True,True)
        if len(small)==len(ref):
            s1,s2,fu,sc=MATCHER._preprocess(small,ref,skip_structure_reduction=True)
            result=MATCHER._match(s1,s2,fu,sc,use_rms=True,break_on_match=False)
            if result is not None:
                rms=float(result[0])
                if max(result[1])<MATCHER.stol:strict=True
                else:strict=MATCHER._match(s1,s2,fu,sc,use_rms=False,break_on_match=True) is not None
        official=True
    if verify:
        pred=make_structure(kg,fg[:len(z)],z);expected=MATCHER.get_rms_dist(pred,truth);expected_fit=MATCHER.fit(pred,truth)
        expected_rms=np.nan if expected is None else expected[0]
        if not np.isclose(rms,expected_rms,rtol=1e-8,atol=1e-8,equal_nan=True) or strict!=expected_fit:
            raise RuntimeError(f'Public method mismatch {rms},{strict} vs {expected_rms},{expected_fit}')
    return rms,strict,official


def score(task):
    lo,hi,first,last,check=task
    ref,pools=resources();path=TILES/f'cases{lo:05}-{hi:05}-draws{first:02}-{last:02}.npz'
    signature=fingerprint()
    if path.is_file():
        with np.load(path) as cache:
            if str(cache['signature'])==signature:return path.name,0.,0
    started=perf_counter()
    shape=(hi-lo,len(ARMS),last-first)
    rms=np.full(shape,np.nan);fits=np.zeros(shape,bool);vol=np.empty(shape,np.float32)
    errors=Counter();attempts=0;formal=0;public=0
    for target in range(lo,hi):
        chem=ref['z'][target];z=chem[chem>0].astype(np.int64)
        ctx=target_context(ref['k'][target],ref['f'][target],chem)
        native_logvol=float(np.sum(ref['k'][target,:3],dtype=np.float64))
        memo={}
        for arm,(ks,fs) in enumerate(pools):
            k=ks[target,first:last];f=fs[target,first:last];gg=unpack_gram(k)
            row=target-lo;vol[row,arm]=k[:,:3].sum(1,dtype=np.float64)-native_logvol
            for c in range(last-first):
                key=(k[c].tobytes(),f[c].tobytes())
                if key not in memo:
                    do_verify=check or ((target*104729+arm*331+(first+c)*1543)%2003==0)
                    try:
                        value=candidate(ctx,k[c],f[c],z,gg[c],do_verify)
                    except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as exc:
                        errors[type(exc).__name__]+=1;value=(np.nan,False,False)
                    memo[key]=value;attempts+=1;formal+=value[2];public+=do_verify
                rms[row,arm,c],fits[row,arm,c],_=memo[key]
    tmp=path.with_suffix('.writing')
    with tmp.open('wb') as out:
        np.savez_compressed(out,ids=ref['ids'][lo:hi],arms=np.array(ARMS),rms=rms,strict_fit=fits,
            log_volume_ratio=vol,first=first,last=last,signature=signature,pymatgen=version('pymatgen'),
            unique_pairs=attempts,forwarded=formal,public_api_comparisons=public,errors=json.dumps(errors),
            elapsed_seconds=perf_counter()-started)
    tmp.replace(path)
    return path.name,perf_counter()-started,formal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=5);p.add_argument('--chunk',type=int,default=64)
    p.add_argument('--stage',choices=('pilot','primary','extension','all'),default='primary')
    p.add_argument('--stop',type=int,default=9046);p.add_argument('--verify-all',action='store_true')
    args=p.parse_args();pin();TILES.mkdir(parents=True,exist_ok=True)
    if args.stage=='pilot':plans=[(0,min(128,args.stop),0,1),(0,min(128,args.stop),1,5)]
    else:
        plans=[]
        if args.stage in ('primary','all'):plans=[(0,args.stop,0,1),(0,args.stop,1,5),(0,min(128,args.stop),5,20)]
        if args.stage in ('extension','all') and args.stop>128:plans.append((128,args.stop,5,20))
    tasks=[(i,min(i+args.chunk,stop),a,b,args.verify_all) for start,stop,a,b in plans for i in range(start,stop,args.chunk)]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=pin) as ex:
        for name,t,forwarded in ex.map(score,tasks):
            if t:print(name,'seconds',round(t,1),'pymatgen pairs',forwarded,flush=True)


if __name__=='__main__':main()
