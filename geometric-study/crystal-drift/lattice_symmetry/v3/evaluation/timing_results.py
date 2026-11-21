"""Retrospective recovery among native arrays returned before wall-clock deadlines."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from score import score
from analysis import ci
HERE=Path(__file__).resolve().parent;CLOCK=HERE/"timing"
BUDGETS=(5,10,20,50,100,250,500,1000)

def write_csv(path,rows):
    with path.open("w",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    info=json.loads((CLOCK/"settings.json").read_text())
    cases={};rows=[];pairs=[];draw_rows=[];lat=[]
    for m in info["method_order"]:
        source=CLOCK/(m+".npz");out=CLOCK/(m+"-scores.npz");score(source,"test",out)
        with np.load(source) as p,np.load(out) as d:
            np.testing.assert_array_equal(p["ids"],d["ids"]);times=p["elapsed_ms"];rms=d["rms"];ids=d["ids"]
        assert (np.diff(times,axis=1)>=0).all()
        increments=np.diff(times,axis=1,prepend=0)
        b1=times[:,0]
        lat.append(dict(method=m,targets=len(ids),draw_cap=times.shape[1],first_median=float(np.median(b1)),
                        first_p95=float(np.percentile(b1,95)),serial_median=float(np.median(increments)),serial_p95=float(np.percentile(increments,95))))
        for ms in BUDGETS:
            delivered=times<=ms;hit=(delivered&np.isfinite(rms)).any(1)
            cases[m,ms]=hit
            low,high=ci(hit)
            rows.append(dict(method=m,ms=ms,n=len(ids),completed_draws=int(delivered.sum()),hits=int(hit.sum()),mr=float(hit.mean()),low=low,high=high))
        for k in range(1,21):
            hit=np.isfinite(rms[:,:k]).any(1);lo,hi=ci(hit)
            draw_rows.append(dict(method=m,k=k,n=len(ids),hits=int(hit.sum()),mr=float(hit.mean()),low=lo,high=hi))
    for m in info["method_order"]:
        for other in info["method_order"]:
            if m==other:continue
            for ms in BUDGETS:
                a,b=cases[m,ms],cases[other,ms];delta=a.astype(float)-b
                lo,hi=ci(delta)
                pairs.append(dict(first=m,second=other,ms=ms,first_only=int((a&~b).sum()),second_only=int((b&~a).sum()),
                                  pp=100*float(delta.mean()),low=100*lo,high=100*hi))
    for filename,r in (("latency",lat),("elapsed",rows),("paired",pairs),("draws",draw_rows)):write_csv(CLOCK/(filename+".csv"),r)
if __name__=="__main__":main()
