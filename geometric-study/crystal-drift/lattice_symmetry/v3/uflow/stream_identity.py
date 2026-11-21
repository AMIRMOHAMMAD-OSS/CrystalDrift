"""Recreate each RNG tile independently, recover its saved identity output, and link all training-seed replicates."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SALT={'family':38473,'shared':38473,'test':38479}
ARMS=('small-23','small-37','medium-23','medium-37')

def write(path,data):
    path.parent.mkdir(exist_ok=True,parents=True);temp=path.with_suffix('.writing')
    temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

@torch.no_grad()
def reconstruct(path,panel,record):
    meta=sha(path)
    if record.is_file():
        result=json.loads(record.read_text());assert result['prior_shard_sha256']==meta;return result
    with np.load(path) as d:
        n=d['native_indices'];species=d['Z'];cells=d['L'];coords=d['F'];count=cells.shape[1]
    with np.load(ROOT/'revision/full/external/chemistry.npz') as d:key=d['sort_key']
    noise=[];tokens=[]
    for ix,z in zip(n,species):
        seed=int(np.random.SeedSequence([SALT[panel],int(ix)]).generate_state(1)[0]);g=torch.Generator(device='cpu');g.manual_seed(seed)
        tok=sorted(z[z>0].tolist(),key=lambda x:(key[x],x));tokens.extend([tok+[0]*(100-len(tok))]*count)
        for _ in range(count):
            x=torch.randn((1,100,12),generator=g);noise.append((x-x.mean(1,keepdim=True))[0])
    latent=torch.stack(noise);toks=torch.tensor(tokens);digest=hashlib.sha256(memoryview(latent.numpy()).cast('B')).hexdigest()
    refL=cells.reshape(-1,3,3);refF=coords.reshape(-1,20,3);max_l=max_f=0.;off_l=off_f=0
    for i in range(0,len(latent),64):
        x=latent[i:i+64];typ=toks[i:i+64];mask=typ!=0
        L=((x[...,3:]*mask[...,None]).sum(1)/mask.sum(1)[:,None]).reshape(-1,3,3)
        F=(x[...,:3]@torch.linalg.pinv(L)).remainder(1.).masked_fill(~mask[...,None],0.)
        order=torch.argsort(torch.where(mask,typ,120),dim=1,stable=True)[:,:20]
        F=F.gather(1,order[...,None].expand(-1,-1,3));cells2=L.numpy();coords2=F.numpy()
        d=cells2-refL[i:i+len(x)];f=coords2-refF[i:i+len(x)];f=f-np.round(f)
        max_l=max(max_l,float(np.abs(d).max()));max_f=max(max_f,float(np.abs(f).max()))
        off_l+=int(np.count_nonzero(np.any(d!=0,axis=(1,2))));off_f+=int(np.count_nonzero(np.any(f!=0,axis=(1,2))))
    assert max_l<2e-6
    out=dict(prior_shard_sha256=meta,gaussian_sha256=digest,targets=len(n),draws_per_target=count,
             max_lattice_abs_error=max_l,max_fractional_torus_error=max_f,nonbitwise_cells=off_l,nonbitwise_sites_draws=off_f)
    write(record,out);return out

def main(a):
    torch.set_num_threads(1);summary={}
    for panel in SALT:
        source=HERE/'predictions/prior-11/frozen'/panel;dest=HERE/'streams'/panel
        rows=[];checked=0
        for p in sorted(source.glob('?????-?????.npz')):
            record=dest/(p.stem+'.json');row=reconstruct(p,panel,record);rows.append(row)
            if a.verify:
                for tag in ARMS:
                    q=HERE/'predictions'/tag/'frozen'/panel/p.name
                    with np.load(q) as d,np.load(p) as parent:
                        np.testing.assert_array_equal(d['ids'],parent['ids']);np.testing.assert_array_equal(d['Z'],parent['Z'])
                        assert str(d['noise_sha256'])==row['gaussian_sha256']
                    checked+=1
        summary[panel]=dict(shards=len(rows),targets=sum(r['targets'] for r in rows),
            draws=sum(r['targets']*r['draws_per_target'] for r in rows),replicate_hashes_checked=checked,
            max_lattice_abs_error=max(r['max_lattice_abs_error'] for r in rows),max_fractional_torus_error=max(r['max_fractional_torus_error'] for r in rows),
            nonbitwise_cells=sum(r['nonbitwise_cells'] for r in rows),nonbitwise_sites_draws=sum(r['nonbitwise_sites_draws'] for r in rows))
        print(panel,summary[panel],flush=True)
    write(HERE/('verified-noise-identity.json' if a.verify else 'prior-stream-replay.json'),dict(torch=torch.__version__,
          law='CPU torch.Generator separately keyed by dataset split salt and original dataset row; one centered 100 x 12 Gaussian per ordered draw',panels=summary))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');main(p.parse_args())
