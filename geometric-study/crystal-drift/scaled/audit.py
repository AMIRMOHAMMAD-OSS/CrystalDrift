"""Audit material-ID and chemical-formula overlap on all rows and the fixed cohorts."""
import argparse
import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiment import select


@lru_cache(maxsize=100000)
def reduced(key):
    counts=Counter(key);g=math.gcd(*counts.values())
    return tuple((z,v//g) for z,v in counts.items())


def keys(row):
    key=tuple(map(int,row[row>0]));return key,reduced(key)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--train',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();report={'training_directory':args.train.name,'test_directory':args.data.name}
    with np.load(args.train/'train.npz') as d:
        train_ids=set(d['ids'].tolist()); exact,forms=map(set,zip(*(keys(r) for r in np.unique(d['z'],axis=0))))
    for split in ('val','test'):
        with np.load(args.data/(split+'.npz')) as d:
            z=d['z'];ids=d['ids']; idx=select(ids,128)
        overlap=len(train_ids&set(ids.tolist()));assert overlap==0
        labels=[keys(r) for r in z];known=np.array([(k in exact, f in forms) for k,f in labels]);
        report[split]=dict(n=len(ids),id_overlap=overlap,exact_composition_seen=int(known[:,0].sum()),
                           reduced_formula_seen=int(known[:,1].sum()),cohort_n=len(idx),
                           cohort_exact_seen=int(known[idx,0].sum()),cohort_reduced_seen=int(known[idx,1].sum()),
                           cohort_novel_reduced_ids=ids[idx[~known[idx,1]]].tolist())
    args.out.mkdir(exist_ok=True,parents=True)
    (args.out/'formula-overlap.json').write_text(json.dumps(report,indent=2)+'\n')
    for key in ('val','test'):print(key,{k:v for k,v in report[key].items() if not isinstance(v,list)})


if __name__=='__main__':main()
