"""Compare cached identical-seed development replay against the archived CUDA checkpoint selections."""
import argparse,json,os,sys
if os.environ.get('CRYTALDRIFT_NUMBA_SITE'):sys.path.append(os.environ['CRYTALDRIFT_NUMBA_SITE'])
import numpy as np
import torch
from .sampler import HERE,Sampler,V3,ROOT
from .train import dev,sha,CACHE
from .select_weights import score
from .prepare import safe_save
from .network import make_model,pack

def native_cache():
    service=Sampler()
    ids=json.loads((V3/'protocol/splits.json').read_text())['dev_ids']
    with np.load(ROOT/'scaled/mp20-data/val.npz') as data:
        lookup={str(x):i for i,x in enumerate(data['ids'])};z=data['z']
    allseeds=[]
    for mid in ids:
        i=lookup[mid];s=int(np.random.SeedSequence([38471,i]).generate_state(1)[0])
        allseeds.extend(service.seeds(z[i],5,s))
    args=pack(allseeds)
    safe_save(CACHE/'dev-inputs.npz',ids=ids,**{'x'+str(i):a.numpy() for i,a in enumerate(args)})

def check(device):
    folder=HERE/'fits/small-angular-laplace-11'
    obj=torch.load(folder/'half.pt',weights_only=True,map_location='cpu')
    net=make_model(obj['architecture']).to(device);net.load_state_dict(obj['net'])
    out=HERE/f'controls/dev-inputs-{device}.npz';dev(None,net,out,force_cache=True)
    outscore=out.with_name(out.stem+'-scores.npz')
    # Give the pinned scorer its current-source digest when rerunning after native cache reconstruction.
    from .select_weights import score
    if outscore.exists():
        with np.load(outscore) as old:stale=str(old['source_sha'])!=sha(out)
        if stale:outscore.unlink()
    rec=score(out)
    with np.load(out) as new,np.load(folder/'half-dev.npz') as old:
        mask=new['Z']>0
        assert np.array_equal(new['Z'],old['Z']) and np.array_equal(new['ids'],old['ids'])
        L=float(np.max(abs(new['L']-old['L'])))
        F=float(np.max(np.abs((new['F'][mask]-old['F'][mask]+.5)%1-.5)))
    with np.load(outscore) as a,np.load(folder/'half-dev-scores.npz') as b:
        ha=np.isfinite(a['rms']);hb=np.isfinite(b['rms'])
        va=ha&(abs(a['log_volume_ratio'])<=np.log(1.1));vb=hb&(abs(b['log_volume_ratio'])<=np.log(1.1))
        index=np.argwhere(ha!=hb);disagree=[(str(a['ids'][i]),int(j+1)) for i,j in index]
        return dict(input_sha=sha(CACHE/'dev-inputs.npz'),device=device,archived_device='cuda',draws=480,
                    max_cell_A=L,max_wrapped_site=F,flag_disagreements=disagree,native_draw_disagreements=int((va!=vb).sum()),
                    recovered_replay=int(ha.any(1).sum()),recovered_archived=int(hb.any(1).sum()),
                    native_replay=int(va.any(1).sum()),native_archived=int(vb.any(1).sum()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--native-cache',action='store_true')
    a=p.parse_args();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    if a.native_cache:native_cache()
    result=check(a.device)
    (HERE/f'dev-input-parity-{a.device}.json').write_text(json.dumps(result,indent=2))
    if a.device=='cuda' and not result['flag_disagreements'] and not result['native_draw_disagreements']:
        (CACHE/'dev-gpu-parity-ok.json').write_text(json.dumps(result,indent=2))
    print(result,flush=True)
if __name__=='__main__':main()

