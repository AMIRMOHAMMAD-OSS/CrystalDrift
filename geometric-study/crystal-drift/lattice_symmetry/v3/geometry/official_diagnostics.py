"""Refresh the official geometry tables using complete, exactly matched symbolic proposal streams."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
for name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[name]="1"
from .raw_templates import audit
from .volume_strata import TAGS,E,OUT,study
from ..evaluation.storage import pool

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-architecture",action="store_true",
                        help="Wait for the frozen architecture registry and its released official outcomes.")
    args=parser.parse_args()
    registry=E/"scale-frozen.json"
    if args.after_architecture:
        while not registry.is_file():time.sleep(45)
    scale=[] if not registry.is_file() else [v["tag"] for v in json.loads(registry.read_text())["methods"]]
    # The independently launched seven-system control has its own development-only freeze.
    auxiliary=E/"weight-angular-small-systemaux.json"
    if auxiliary.is_file():
        scale.extend(v["tag"] for v in json.loads(auxiliary.read_text())["methods"])
    if args.after_architecture:
        while not all((E/f"scores/test/{tag}/outcomes.npz").is_file() for tag in scale):time.sleep(45)
    verified=json.loads((OUT/"raw-templates-test.json").read_text())
    current=[t for t in dict.fromkeys([*TAGS,*scale]) if (E/f"scores/test/{t}/outcomes.npz").is_file() and (pool("test",t)/"settings.json").is_file()]
    if args.after_architecture:assert set(scale)<=set(current),"Missing registered architecture raw metadata"
    pending=[t for t in current if t!="prior" and t not in verified["matching_metadata_methods"]]
    if pending:audit("test",pending)
    study(current)
    for module,args in (("conditional",["test",*current]),("selector_outcomes",[]),("write_coverage_section",[])):
        subprocess.run([sys.executable,"-m","lattice_symmetry.v3.geometry."+module,*args],check=True)
    print("Official diagnosis completed",current,flush=True)
if __name__=="__main__":main()
