"""Average independent fit seeds within each target and bootstrap paired outcomes."""
import json
import numpy as np
from results import HERE,ROOT,read,reorder,ci,by_draw

def summary(flag,base,groups,subset):
    f=flag[:,subset];b=base[subset];g=groups[subset];delta=100*(f.mean(0)-b)
    lo,hi=ci(delta,g)
    return dict(hits_by_seed=f.sum(1).tolist(),prior_hits=int(b.sum()),mean_mr=float(f.mean()),
                gained_by_seed=(f&~b).sum(1).tolist(),lost_by_seed=(~f&b).sum(1).tolist(),delta_pp=float(delta.mean()),
                paired_95_pp=[lo,hi],clusters=len(set(g)),targets=len(b))

def main():
    registry=json.loads((HERE.parent/'protocol/splits.json').read_text());out={}
    key=lambda row:tuple(sorted(row[row>0]))
    with np.load(ROOT/'scaled/mp20-data/train.npz') as d:
        selected=np.isin(d['ids'],registry['train_ids']);chemistry={key(r) for r in d['z'][selected]}
    for panel in ('confirm','family','shared','test'):
        ids=registry[panel+'_ids'];groups=np.array([registry['family_by_challenge_id'][m] for m in ids] if panel in ('family','shared') else ids)
        split='train' if panel in ('family','shared') else ('val' if panel=='confirm' else 'test')
        with np.load(ROOT/f'scaled/mp20-data/{split}.npz') as d:
            lookup={str(m):i for i,m in enumerate(d['ids'])};zs=d['z'][[lookup[m] for m in ids]]
        size=(zs>0).sum(1);seen=np.array([key(r) in chemistry for r in zs]);all_rows=np.ones(len(ids),bool)
        subsets={'large':size>=9,'seen':seen,'unseen':~seen,**{f'{lo}-{hi}':(size>=lo)&(size<=hi) for lo,hi in ((1,4),(5,8),(9,12),(13,20))}}
        prior=reorder(read(HERE/f'scores/prior-11/frozen/{panel}/outcomes.npz'),ids)
        arms={}
        for tier in ('small','medium'):
            files=[HERE/f'scores/{tier}-{seed}/frozen/{panel}/outcomes.npz' for seed in (11,23,37)]
            if not all(p.is_file() for p in files):continue
            seeds=[reorder(read(p),ids) for p in files];arms[tier]={}
            for k in (1,5,20):
                if k>prior['rms'].shape[1]:continue
                pool=[by_draw(r,k) for r in seeds];p0=by_draw(prior,k)
                out_row={'subsets':{label:{} for label,mask in subsets.items() if mask.any()}};arms[tier][k]=out_row
                for endpoint,flag,base in [('scale',np.array([x[0] for x in pool]),p0[0]),
                                           ('native',np.array([x[2] for x in pool]),p0[2]),
                                           ('maximum_site',np.array([x['strict_fit'][:,:k].any(1) for x in seeds]),prior['strict_fit'][:,:k].any(1))]:
                    out_row[endpoint]=summary(flag,base,groups,all_rows)
                    for label,mask in subsets.items():
                        if mask.any():out_row['subsets'][label][endpoint]=summary(flag,base,groups,mask)
                value=[float(x[1][x[0]].mean()) if x[0].any() else None for x in pool]
                out_row['normalized_rms_by_seed']=value
                out_row['mean_normalized_rms_across_seeds']=float(np.mean([v for v in value if v is not None])) if any(v is not None for v in value) else None
        if arms:out[panel]=arms
    dst=HERE/'native-three-seed.json';tmp=dst.with_suffix('.writing');tmp.write_text(json.dumps(dict(seeds=[11,23,37],
        uncertainty='5,000 percentile resamples of targets; family and shared panels resample operational families; average seeds within target first',panels=out),indent=2)+'\n');tmp.replace(dst)
    print({p:list(v) for p,v in out.items()})
if __name__=='__main__':main()
