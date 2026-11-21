"""Calculate paired fixed-formula differences and retain losses as well as gains."""
import csv
from pathlib import Path
import numpy as np
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"evaluation"))
from analysis import ci

def main():
    datasets=HERE/"results"
    with np.load(datasets/"prior-cases.npz") as p:
        prior={k:p[k] for k in p}
    with np.load(HERE/"pools/prior.npz") as ref:
        formulas=ref["z"];source=ref["chemistry_source"]
    out=[]
    for f in sorted(datasets.glob("*-cases.npz")):
        name=f.name[:-len("-cases.npz")]
        with np.load(HERE/"pools"/(name+".npz")) as d:
            np.testing.assert_array_equal(d["chemistry_source"],source)
            np.testing.assert_array_equal(np.sort(d["z"],axis=1),np.sort(formulas,axis=1))
        with np.load(f) as d:
            np.testing.assert_array_equal(d["ids"],prior["ids"])
            for field in ("valid","unique","novel","valid_unique_novel"):
                if field=="valid_unique_novel":
                    b=prior["valid"]&prior["novel"]&prior["unique"];a=d["valid"]&d["novel"]&d["unique"]
                else:a,b=d[field],prior[field]
                delta=a.astype(float)-b;low,high=ci(delta)
                out.append(dict(method=name,endpoint=field,n=len(a),hits=int(a.sum()),prior_hits=int(b.sum()),
                    gained=int((a&~b).sum()),lost=int((b&~a).sum()),difference_pp=100*delta.mean(),low_pp=100*low,high_pp=100*high))
    with (datasets/"paired.csv").open("w",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=out[0]);w.writeheader();w.writerows(out)

if __name__=="__main__":main()
