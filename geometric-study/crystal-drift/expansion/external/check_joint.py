"""Validate an independently hash-selected sample against pymatgen's POSCAR reader."""
import argparse,hashlib,json
from pathlib import Path
import re
import numpy as np
from pymatgen.io.vasp.inputs import Poscar
HERE=Path(__file__).resolve().parent
def main(path):
    records=re.split(r"(?m)^-+\s+(?:POSCAR|CONTCAR)\s+\d+\s+-+\s*$",path.read_text())
    with np.load(HERE/"crystalflow-joint.npz") as pool:
        ix=sorted(np.flatnonzero(pool["present"]),key=lambda j:hashlib.sha256(f"source-poscar-audit/{j}".encode()).digest())[:100]
        maximum=0.
        for j in ix:
            c=Poscar.from_str(records[1+j].strip()).structure
            z=np.array(c.atomic_numbers);order=np.argsort(z,kind="stable");n=len(z)
            np.testing.assert_array_equal(z[order],pool["z"][j,:n])
            np.testing.assert_allclose(c.lattice.matrix,pool["lattice"][j],atol=1e-10)
            delta=(c.frac_coords[order]-pool["f"][j,:n]+.5)%1-.5
            maximum=max(maximum,float(abs(delta).max()))
    assert maximum<1e-10
    result=dict(checked=100,failed=0,maximum_wrapped_site_error=maximum,indexes=list(map(int,ix)))
    (HERE/"joint-parity.json").write_text(json.dumps(result,indent=2)+"\n")
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("path",type=Path);main(p.parse_args().path)
