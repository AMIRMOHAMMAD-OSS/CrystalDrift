"""Reduce existing family and chemistry holdout outcomes without creating new draws."""
import argparse
import os
import time

for key in ("MKL_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="1"

from .family_effects import V3,summarize


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--wait",action="store_true",help="Poll for the scheduled three-seed outcomes.")
    args=parser.parse_args()
    pending=[(panel,f"{arm}-{size}") for panel in ("family","shared")
             for size in ("small","medium") for arm in ("angular","contacts")]
    while pending:
        for panel,group in pending.copy():
            tags=[f"{group}-{seed}" for seed in (11,23,37)]
            if not all((V3/f"evaluation/scores/{panel}/{t}/outcomes.npz").is_file() for t in tags):
                continue
            row=summarize(panel,group+"-three-seeds",tags)["all"]["scale-5"]
            print(panel,group,row,flush=True)
            pending.remove((panel,group))
        if not args.wait:break
        if pending:time.sleep(40)
    if pending:print("Missing replicate outcomes:",pending,flush=True)


if __name__=="__main__":main()
