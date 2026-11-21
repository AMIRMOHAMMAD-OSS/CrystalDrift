"""Retrospective checks use the released pinned matcher and the actual timed candidates."""
import os
for k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMBA_NUM_THREADS"):os.environ[k]="1"
import importlib.util
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"expansion/training"))
spec=importlib.util.spec_from_file_location("archived_gpu_score",ROOT/"expansion/training/gpu_score.py")
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
original.GPU=HERE/"gpu"
if __name__=="__main__":
    for path in sorted(original.GPU.glob("*-episodes.npz")):original.one(path)
    original.reports()
