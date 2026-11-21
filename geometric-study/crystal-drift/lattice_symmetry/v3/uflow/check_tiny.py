"""Compare the positive-volume scale-free exclusion against both public matcher APIs."""
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"evaluation"))
from score import MATCHER, Structure, target_context, tiny_cell, truth
from data import unpack_gram

def main():
    lookup,ref=truth("val")
    path=HERE/"predictions/prior-11/frozen/confirm/00000-00096.npz"
    rng=np.random.default_rng(314159)
    rows=[]
    with np.load(path) as source:
        ids,zs,Ls,fs=[source[k] for k in ("ids","Z","L","F")]
        # Deterministic native prior examples, plus well-conditioned positives and
        # unimodular shears of a positive. Deliberately bypass certification at high condition.
        for i in range(24):
            ix=lookup[str(ids[i])]; z=zs[i][zs[i]>0].astype(np.int64)
            ctx=target_context(ref["k"][ix],ref["f"][ix],ref["z"][ix])
            orig=np.linalg.cholesky(unpack_gram(ref["k"][ix]))
            truth_f=ref["f"][ix,:len(z)].astype(float)
            for kind,L,f in [("native",Ls[i,0].astype(float),fs[i,0,:len(z)].astype(float)),
                             ("target",orig,truth_f)]:
                scale=(.045/abs(np.linalg.det(L)))**(1/3)
                fixtures=[(kind,L*scale,f)]
                if kind=="target" and i<4:
                    for shear in (7,39):
                        U=np.array([[1,shear,0],[0,1,0],[0,0,1]])
                        fixtures.append((f"shear-{shear}",U@(scale*L),(f@np.linalg.inv(U))%1))
                for desc,A,v in fixtures:
                    start=time.perf_counter()
                    a,b,forwarded=tiny_cell(ctx,A,v,z)
                    pred=Structure(A,z,v)
                    pub=MATCHER.get_rms_dist(pred,ctx[0]);p=np.nan if pub is None else pub[0]
                    fit=bool(MATCHER.fit(pred,ctx[0]))
                    assert np.isclose(a,p,atol=1e-9,rtol=1e-9,equal_nan=True) and bool(b)==fit
                    rows.append(dict(id=str(ids[i]),fixture=desc,volume=float(abs(np.linalg.det(A))),
                                     condition=float(np.linalg.cond(A)),forwarded=bool(forwarded),
                                     hit=bool(pub is not None),seconds=time.perf_counter()-start))
    # Very small determinant at well-conditioned shape, and a stretched high-condition
    # synthetic with a single atom to keep the stock primitive finder inexpensive.
    # For the latter choose diagonal lengths with a strongly anisotropic normalized shape.
    one=int(np.flatnonzero((ref["z"]>0).sum(1)==1)[0])
    z=ref["z"][one,:1].astype(np.int64);f=np.zeros((1,3))
    ctx=target_context(ref["k"][one],ref["f"][one],ref["z"][one])
    for L in (np.eye(3)*.03,np.diag([.04,.05,1.4])):
        a,b,sent=tiny_cell(ctx,L,f,z);pred=Structure(L,z,f)
        p=MATCHER.get_rms_dist(pred,ctx[0]);fit=MATCHER.fit(pred,ctx[0])
        assert np.isclose(a,np.nan if p is None else p[0],atol=1e-9,equal_nan=True) and b==fit
        rows.append(dict(id=str(ref["ids"][one]),fixture="tiny-determinant",volume=float(abs(np.linalg.det(L))),
                         condition=float(np.linalg.cond(L)),forwarded=bool(sent),hit=bool(p is not None)))
    (HERE/"tiny-cell-parity.json").write_text(json.dumps(dict(total=len(rows),disagreements=0,
        excluded=sum(not r["forwarded"] for r in rows),positive=sum(r["hit"] for r in rows),fixtures=rows),indent=2)+"\n")
if __name__=="__main__":main()
