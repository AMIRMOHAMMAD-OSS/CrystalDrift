"""Serialize the exact native-basis operation set used by the orbit sampler."""
from pathlib import Path
import numpy as np
from ..models.sampler import group_ops,GROUP,START,M,T,V2


def main():
    ops=[];offsets=[0];bases=[]
    for g in range(1,231):
        _,A,I=group_ops(g);k=GROUP[g]-1;sl=slice(START[k],START[k+1])
        R=A@M[sl]@I;t=T[sl]@I
        assert np.allclose(R,np.rint(R)) and np.allclose(t*24,np.rint(t*24))
        rows=np.unique(np.column_stack([np.rint(R).reshape(-1,9),np.rint(24*t)%24]),axis=0)
        ops.extend(rows);offsets.append(len(ops));bases.append(A)
    with np.load(V2/"wyckoff/charts.npz") as archive:hall=archive["halls"]
    np.savez_compressed(Path(__file__).with_name("native-operations.npz"),rows=np.asarray(ops,dtype=np.int16),
                        offsets=offsets,bases=bases,halls=hall)


if __name__=="__main__":main()
