"""Prepare all registered fresh later-traversal banks without reading held-out geometry."""
import argparse
import shutil
import os,sys,hashlib,json,time
from pathlib import Path
if os.environ.get('CRYTALDRIFT_NUMBA_SITE'):sys.path.append(os.environ['CRYTALDRIFT_NUMBA_SITE'])
import numpy as np
from .prepare import BLOCK,CACHE,make_shard,safe_save
from .sampler import Sampler

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',type=int,required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--scratch-dir',type=Path)
    a=p.parse_args();s=Sampler();beg=time.time();checks={}
    for epoch in range(1,8):
        order=np.random.default_rng(55171+epoch).permutation(len(s.rows))
        for k,start in enumerate(range(0,len(order),BLOCK)):
            limit_mib=100 if a.scratch_dir else 1200
            if shutil.disk_usage(CACHE).free<limit_mib*2**20:
                raise SystemExit('Stopping optional disk cache at a shard boundary; fit can generate this shard in RAM.')
            if k%a.workers!=a.part:continue
            if a.scratch_dir:
                name=f'epoch-{epoch:02}/train-{k:03}.npz';original=CACHE/name;target=a.scratch_dir/name
                if original.exists():file=original
                else:
                    file=target
                    if not target.exists():
                        data=make_shard(s,order[start:start+BLOCK],k,epoch,in_memory=True)
                        target.parent.mkdir(parents=True,exist_ok=True);safe_save(target,**data)
                checks[name]=hashlib.sha256(file.read_bytes()).hexdigest()
                if k%25==0:print(epoch,k,round(time.time()-beg,1),shutil.disk_usage(CACHE).free//2**20,flush=True)
            else:make_shard(s,order[start:start+BLOCK],k,epoch)
        if a.scratch_dir:
            dest=CACHE.parent/'fresh-bank-sha256.json';tmp=dest.with_suffix('.building')
            tmp.write_text(json.dumps(dict(scratch=str(a.scratch_dir),through_epoch=epoch,shards=checks),sort_keys=True,indent=1));os.replace(tmp,dest)

