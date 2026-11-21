"""Check primitive-cell chart multiplicities and each orbit against independent operation sets."""
import json
from pathlib import Path
import string
import numpy as np
from ..models.sampler import chart,orbit_chart,GROUP,group_ops

def main():
    here=Path(__file__).resolve().parent
    with np.load(here/"native-operations.npz") as db:rows,off=db["rows"],db["offsets"]
    rng=np.random.default_rng(25811);results=[]
    for g in range(1,231):
        R,T=rows[off[g-1]:off[g],:9].reshape(-1,3,3),rows[off[g-1]:off[g],9:]/24
        _,A,I=group_ops(g)
        for letter in string.ascii_letters[:GROUP[g]-GROUP[g-1]]:
            J,b=orbit_chart(g,letter)
            c,d=chart(g,letter)
            u=rng.uniform(.03,.97,3)
            # Verify two chart parameters and a generic tangent throughout one active chart.
            worst=0.
            for params in (u,u+.014*rng.normal(size=3)):
                F=params@J+b
                conventional=(params@c+d)@I
                actual=F[0]@R+T
                for reference in (actual,conventional):
                    dx=F[:,None]-reference[None]
                    dmax=np.abs(dx-np.rint(dx)).max(-1)
                    worst=max(worst,dmax.min(0).max(),dmax.min(1).max())
                ds=F[:,None]-F[None];sep=np.abs(ds-np.rint(ds)).max(-1)
                assert (sep<1e-8).sum()==len(F),(g,letter)
            dof=int(np.linalg.matrix_rank(c[0]))
            assert np.linalg.matrix_rank(J[0])==dof and len(J)==round(len(c)*abs(np.linalg.det(A)))
            assert worst<5e-13,(g,letter,worst)
            results.append(dict(group=g,letter=letter,conventional=len(c),native=len(J),rank=dof,worst=float(worst)))
    out=dict(groups=230,charts=len(results),seed=25811,errors="bidirectional max wrapped fractional component",
             ranked={k:sum(x["rank"]==k for x in results) for k in range(4)},
             worst=max(x["worst"] for x in results),records=results)
    (here/"primitive-charts.json").write_text(json.dumps(out,separators=(",",":"))+"\n",encoding="utf-8")
    print(out["charts"],out["worst"])

if __name__=="__main__":main()
