"""Recompute manuscript inputs from the research evidence, then build a minimal submission."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
V3 = HERE.parent

def run(script, *args):
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    subprocess.run([sys.executable, str(script), *args], cwd=V3, env=env, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", action="store_true", help="rebuild the paired reductions from ordered score files")
    parser.add_argument("--preview", action="store_true", help="allow incomplete secondary-seed tables and mark outputs working")
    args = parser.parse_args()
    if args.analysis:
        run(V3/"evaluation/analysis.py", "test","family","shared","arrangement","arrangement-control")
    run(HERE/"figures.py","--results","contacts-medium-long","Contacts, 2M (selected)")
    for name in ("operating_points","capacity_points","experiments","kernels","scale_details","clock_detail","physical_diagnostics"):
        run(HERE/(name+".py"))
    run(HERE/"replications.py", *(["--working"] if args.preview else []))
    run(HERE/"compose.py", *(["--working"] if args.preview else []))
    run(V3/"release/author_report.py")
    if not args.preview:
        subprocess.run([sys.executable,"-m","lattice_symmetry.v3.geometry.check_replication_tables"],cwd=V3.parents[1],check=True)
    run(V3/"release/author_brief.py")
    run(HERE/"build_submission.py", *(["--preview"] if args.preview else []))
    run(HERE/"check_claims.py", "working" if args.preview else "CrytalDrift")

if __name__=="__main__":
    main()
