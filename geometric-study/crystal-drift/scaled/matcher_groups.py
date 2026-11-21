"""Report full-composition recovery separately for held-out chemistry and multielement structures."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();folder=args.run
    reference=Path(__file__).resolve().parent/(folder.name.split('-to-')[-1]+'-data')
    overlap=json.loads((folder/'formula-overlap.json').read_text())
    allrows=[]
    for partition in ('val','test'):
        fits=defaultdict(dict);ordered=None
        for f in sorted(folder.glob(f'*-seed*-{partition}-matching.json')):
            r=json.loads(f.read_text());s=r['summary'];ids=r['material_ids']
            if ordered is not None and ids!=ordered:raise ValueError('ID order changed')
            ordered=ids;fits[s['method']][s['seed']]=np.isfinite(np.asarray(r['rms_by_candidate'],float))
        if ordered is None:continue
        lookup={mid:i for i,mid in enumerate(ordered)}
        with np.load(reference/(partition+'.npz')) as d:
            ref=dict(zip(d['ids'],d['z']))
            cardinal=np.asarray([len(set(ref[mid][ref[mid]>0])) for mid in ordered])
        unknown=set(overlap[partition]['cohort_novel_reduced_ids'])
        if not unknown.issubset(lookup):raise ValueError('Unknown formula flags do not align to IDs')
        masks={'all':np.ones(len(ordered),bool),'at_least_three_elements':cardinal>=3,
               'unseen_reduced_formula':np.asarray([mid in unknown for mid in ordered])}
        for label,mask in masks.items():
            n=mask.sum()
            if not n:continue
            boot=np.random.default_rng(741123).multinomial(n,np.full(n,1/n),size=10000)
            for m,found in sorted(fits.items()):
                ss=sorted(found)
                for metric,k in (('match_at_1',1),('match_at_5',5)):
                    values=np.mean([v[:,:k].any(1)[mask] for v in found.values()],axis=0)
                    lo,hi=np.quantile((boot@values)/n,[.025,.975])
                    allrows.append(dict(split=partition,group=label,n=int(n),method=m,metric=metric,seeds=ss,
                                        estimate=float(values.mean()),low=float(lo),high=float(hi)))
    target=folder/'structure-recovery-subgroups.json'
    target.write_text(json.dumps(allrows,indent=1)+'\n')


if __name__=='__main__':
    main()

