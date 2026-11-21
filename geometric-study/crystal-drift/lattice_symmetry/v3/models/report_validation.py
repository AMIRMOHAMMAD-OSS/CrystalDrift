"""Target-paired validation evidence at frozen model choice, including discordant outcomes."""
import csv,json
from pathlib import Path
import numpy as np
from .sampler import HERE

def write(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)

def interval(x):
    if len(x)==0:return (float('nan'),float('nan'))
    rng=np.random.default_rng(7331)
    boot=x[rng.integers(0,len(x),(5000,len(x)))].mean(1)
    return np.quantile(boot,[.025,.975])

def features(path):
    with np.load(path) as q:
        rms=q['rms'];native=np.isfinite(rms)&(abs(q['log_volume_ratio'])<=np.log(1.1))
        return q['ids'].astype(str),rms,native

def compare(data,ref,mask,k):
    ids,rms,native=data;other,rr,rnat=ref
    assert np.array_equal(ids,other)
    a=rms[mask,:k];b=rr[mask,:k]
    h=np.isfinite(a).any(1);h0=np.isfinite(b).any(1)
    n=native[mask,:k].any(1);n0=rnat[mask,:k].any(1)
    d=h.astype(int)-h0;dn=n.astype(int)-n0
    low,high=interval(d);nlow,nhigh=interval(dn)
    best=np.where(np.isfinite(a),a,np.inf).min(1);base=np.where(np.isfinite(b),b,np.inf).min(1)
    both=h&h0
    return dict(targets=int(mask.sum()),hits=int(h.sum()),prior_hits=int(h0.sum()),
        gained=int((d>0).sum()),lost=int((d<0).sum()),delta_pp=100*d.mean(),lo_pp=100*low,hi_pp=100*high,
        native_hits=int(n.sum()),prior_native_hits=int(n0.sum()),native_delta_pp=100*dn.mean(),
        native_lo_pp=100*nlow,native_hi_pp=100*nhigh,rms=float(best[h].mean()) if h.any() else np.nan,
        common_matched=int(both.sum()),common_rms=float(best[both].mean()) if both.any() else np.nan,
        common_control_rms=float(base[both].mean()) if both.any() else np.nan)

def main():
    fits=HERE/'fits';out=[];cost=[];kernels=[]
    for panel in ('dev','confirm'):
        ref=features(fits/'prior'/(panel+'-scores.npz'))
        with np.load(fits/'prior'/(panel+'.npz')) as q:sites=(q['Z'][:,0]>0).sum(1)
        for path in sorted(fits.iterdir()):
            selection=path/'selection.json'
            if not selection.exists():continue
            r=json.loads(selection.read_text());pick=path/r['chosen']
            dev=pick.with_name(pick.stem+'-dev-scores.npz')
            scores=dev if panel=='dev' else path/'frozen-confirm-scores.npz'
            if not scores.exists():continue
            d=features(scores)
            for lo,hi in [(1,20),(1,4),(5,8),(9,12),(13,20),(9,20)]:
                mask=(sites>=lo)&(sites<=hi)
                for k in (1,5):
                    out.append(dict(panel=panel,model=path.name,checkpoint=pick.name,lo=lo,hi=hi,k=k,
                                    **compare(d,ref,mask,k)))
            if panel=='dev':
                c=json.loads((path/'complete.json').read_text())
                cost.append(dict(model=path.name,parameters=c['parameters'],updates=c['updates'],eligible=c['distinct_anchors'],
                    active_seed_structures=c['query_exposures'],positive_exposures=c['reference_exposures'],
                    negative_exposures=c['negative_exposures'],atoms=c['training_atoms'],elapsed_s=c['seconds'],
                    peak_cuda_MiB=c['peak_allocated_MiB'],selected=pick.name,selected_sha=r['sha256']))
            seq=path.name.split('-')
            same_backbone_extra=path.name.endswith('-systemaux')
            if len(seq)<4 or (seq[1] not in ('contacts','reciprocal','variance','meanflow','radial4') and not same_backbone_extra):continue
            control=fits/f'{seq[0]}-angular-{seq[2]}-{seq[3]}'
            if path.name.endswith('-8pass'):control=control.with_name(control.name+'-8pass')
            if not (control/'selection.json').exists():continue
            cp=json.loads((control/'selection.json').read_text())['chosen']
            control_weight=control/cp
            sc=control_weight.with_name(control_weight.stem+'-dev-scores.npz') if panel=='dev' else control/'frozen-confirm-scores.npz'
            if sc.exists():kernels.append(dict(panel=panel,model=path.name,control=control.name,k=5,**compare(d,features(sc),sites>0,5)))
    write(HERE/'validation-contrast.csv',out);write(HERE/'kernel-contrast.csv',kernels);write(HERE/'training-footprint.csv',cost)

if __name__=='__main__':main()

