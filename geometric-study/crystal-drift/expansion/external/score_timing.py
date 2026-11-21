"""Score the exact elapsed-time episodes and report observed recovery versus budget."""
import argparse,csv,json,os,sys
from pathlib import Path
for var in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS","OMP_NUM_THREADS"):os.environ[var]="1"
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"revision/full"))
from match_full import target_context,candidate,unpack_gram
def main(folder):
    with np.load(ROOT/"revision/full/mp20/cohort.npz") as d:
        ids=d["ids"][:128];z=d["z"][:128];rk=d["k"][:128];rf=d["f"][:128]
    with np.load(folder/"serial.npz") as d:
        np.testing.assert_array_equal(d["ids"],ids);np.testing.assert_array_equal(d["z"],z)
        t=d["elapsed_ms"];k=d["k"];f=d["f"]
    assert t.shape==(128,20) and (np.diff(t,axis=1)>0).all()
    rms=np.full(t.shape,np.nan);strict=np.zeros(t.shape,bool);errors=[]
    for i in range(len(ids)):
        ctx=target_context(rk[i],rf[i],z[i]);grams=unpack_gram(k[i])
        for j in range(20):
            try:
                rms[i,j],strict[i,j],_=candidate(ctx,k[i,j],f[i,j],z[i,z[i]>0],grams[j],i%47==0 and j==0)
            except (ValueError,np.linalg.LinAlgError,OverflowError,FloatingPointError) as exc:
                errors.append((i,j,type(exc).__name__))
    np.savez_compressed(folder/"serial-scores.npz",ids=ids,rms=rms,fit=strict,exceptions=json.dumps(errors))
    with (folder/"elapsed-recovery.csv").open("w",newline="") as stream:
        w=csv.writer(stream);w.writerow(["budget_ms","mean_completed","hit_targets","mr"])
        for b in (1,2,5,10,20,30,50,75,100,150,200,250,500,1000,2000,5000,10000,20000):
            n=t<=b;hits=(n&np.isfinite(rms)).any(1)
            w.writerow([b,n.sum(1).mean(),hits.sum(),hits.mean()])
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("folder",type=Path);main(p.parse_args().folder)
