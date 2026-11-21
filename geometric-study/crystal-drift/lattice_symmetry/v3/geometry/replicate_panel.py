"""Freeze and score one already registered pair as soon as both development selections exist."""
import argparse
import subprocess
import sys
from ..evaluation.register import freeze,model

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("seed",type=int,choices=(23,37));p.add_argument("--matcher",required=True)
    a=p.parse_args()
    cfg=freeze(f"challenge-seed{a.seed}-frozen",[model(f"{arm}-small-{a.seed}",f"small-{arm}-laplace-{a.seed}")
                                  for arm in ("angular","contacts")])
    subprocess.run([sys.executable,"-m","lattice_symmetry.v3.geometry.evaluate_challenges","--matcher",a.matcher,
                    "--matrix",str(cfg),"--panels","arrangement","arrangement-control"],check=True)
