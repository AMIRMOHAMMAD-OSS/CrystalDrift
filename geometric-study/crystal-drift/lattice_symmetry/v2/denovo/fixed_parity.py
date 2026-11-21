"""Compare the 10,000 newly computed fixed-basis fingerprints with the released panel."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parents[2]/"expansion/denovo"


def main():
    with np.load(OLD/"pools/learned-drift.npz") as source,np.load(HERE/"pools/fixed-basis.npz") as other:
        keys=("ids","z","f","k","lattice","valid_output")
        for field in keys:
            if field=="valid_output" and field not in source:
                assert other[field].all();continue
            np.testing.assert_array_equal(source[field],other[field])
    src=OLD/"scores/fp/learned-drift"; dst=HERE/"scores/fp/fixed-basis"
    num=0;maximum={"comp":0.,"struc":0.};above=0
    for p in sorted(src.glob("?????-?????.npz")):
        with np.load(dst/p.name) as a,np.load(p) as b:
            for k in ("error","ids"):np.testing.assert_array_equal(a[k],b[k])
            for k in maximum:
                d=np.abs(a[k]-b[k]);np.testing.assert_array_equal(np.isnan(a[k]),np.isnan(b[k]))
                maximum[k]=max(maximum[k],float(np.nanmax(d)))
                if k=="struc":above+=int((np.nanmax(d,axis=1)>1e-10).sum())
            num+=len(a["ids"])
    assert num==10000
    with np.load(OLD/"panel/learned-drift-coverage.npz") as a,np.load(HERE/"panel/fixed-basis-coverage.npz") as b:
        changed={}
        for group in ("ref","generated"):
            x=(a[group+"_chem"]<=10)&(a[group+"_struc"]<=.4)
            y=(b[group+"_chem"]<=10)&(b[group+"_struc"]<=.4)
            changed[group]=int(np.count_nonzero(x!=y))
    result=dict(fresh_cases=num,input_fields=keys,maximum_absolute=maximum,structure_profiles_above_1e_10=above,
            coverage_flag_changes=changed,source="expansion/denovo/pools/learned-drift.npz")
    (HERE/"fixed-fingerprint-replay.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)


if __name__=="__main__":main()
