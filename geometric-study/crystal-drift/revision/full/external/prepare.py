"""Export the exact native pymatgen chemistry data for GPU replay."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import warnings

import numpy as np

HERE=Path(__file__).resolve().parent
STUDY=HERE.parents[2]
sys.path.insert(0,str(STUDY))
from uflowcsp_replay import stage_sources


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--site",nargs="*",default=[]);a=p.parse_args()
    sys.path.extend(a.site)
    from pymatgen.core import Element
    with tempfile.TemporaryDirectory() as temp:
        stage_sources(Path(temp))
        from src.models.denoisers.atomic_features import build_atomic_feature_table
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chemistry=build_atomic_feature_table(120).numpy()
            key=[0.]
            for zz in range(1,119):
                v=Element.from_Z(zz).X
                key.append(float(v) if v is not None and np.isfinite(v) else 100+zz*.001)
        assert np.abs(chemistry).sum()>0
        np.savez(HERE/"chemistry.npz",table=chemistry,sort_key=np.asarray(key))
