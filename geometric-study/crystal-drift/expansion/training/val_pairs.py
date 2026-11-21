"""Measure paired validation differences between sampling and training objectives."""
import csv
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    bootstrap=np.random.default_rng(340754).integers(0,128,(10000,128))
    rows=[]
    for epoch in (1,3,10):
        raw={}
        for arm,nfe in (("drift",1),("iid",1),("meanflow",1),("flow",16)):
            items=[]
            for s in (11,23,37):
                path=HERE/f"fits/{arm}-seed{s}/pass{epoch}-val-{nfe}.npz"
                if path.exists():
                    with np.load(path) as f:items.append((f["ids"],f["energy"]))
            if len(items)==3:
                for ids,_ in items:np.testing.assert_array_equal(ids,items[0][0])
                raw[arm]=np.mean([r[1] for r in items],0)
        for m in ("iid","meanflow","flow"):
            if "drift" not in raw or m not in raw:continue
            delta=raw[m]-raw["drift"]
            low,high=np.quantile(delta[bootstrap].mean(1),[.025,.975])
            rows.append(dict(passes=epoch,first=m,second="drift",contrast="fair validation energy, lower is better",
                         n=128,mean=delta.mean(),lo=low,hi=high))
    if rows:
        with (HERE/"validation-paired.csv").open("w",newline="") as fp:
            writer=csv.DictWriter(fp,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=="__main__":main()
