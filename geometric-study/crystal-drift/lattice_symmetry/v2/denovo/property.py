"""Evaluate the released MP-20 DimeNet++ property network on each frozen valid subset."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/"expansion/denovo"
sys.path.insert(0,str(OLD))
import numpy as np
import score
import property_graphs
score.HERE=property_graphs.HERE=HERE


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("graphs","energy"))
    p.add_argument("arms",nargs="+")
    p.add_argument("--workers",type=int,default=1)
    args=p.parse_args()
    if args.phase=="graphs":
        sys.argv=["property_graphs",*args.arms,"--workers",str(args.workers)]
        property_graphs.main()
        return
    import property_eval
    property_eval.HERE=HERE
    property_eval.PROP=OLD/"property"
    dest=HERE/"energies"
    dest.mkdir(exist_ok=True)
    ref=OLD/"energies/reference.npz"
    if not (dest/ref.name).exists():
        shutil.copyfile(ref,dest/ref.name)
    assert sha(ref)==sha(dest/ref.name)
    with np.load(ref) as e,np.load(ROOT/"scaled/mp20-data/test.npz") as split:
        assert len(e["ids"])==9046
        np.testing.assert_array_equal(e["ids"],split["ids"])
    sys.argv=["property_eval",*args.arms,"--threads","1"]
    property_eval.main()


if __name__=="__main__":main()
