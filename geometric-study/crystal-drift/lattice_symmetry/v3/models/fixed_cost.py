"""Observed development recovery at or below common cumulative training-job clocks."""
import csv
from .learning_curves import HERE,rows

def main():
    records=rows();out=[]
    for tier,model in sorted(set((r['tier'],r['method']) for r in records)):
        candidates=sorted((r for r in records if r['tier']==tier and r['method']==model),key=lambda r:r['seconds'])
        for limit in (30,60,120,240,480,960,1920,3840):
            allowed=[r for r in candidates if r['seconds']<=limit]
            if not allowed:continue
            best=max(allowed,key=lambda r:r['validation_hits'])
            out.append(dict(training_job_s=limit,**best))
    with (HERE/'fixed-training-cost.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
if __name__=='__main__':main()

