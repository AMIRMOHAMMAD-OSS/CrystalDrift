"""Finish protected formula-pool metrics, surrogate relaxations and paired reductions at frozen weights."""
import json
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
staged=["prior","contacts-small","released-angular","released-uflow","angular-small"]
remaining=["contacts-medium-long","angular-medium-long"]
if __name__=="__main__":
    for tag in remaining:
        receipt=HERE/"pools"/(tag+".json")
        while not receipt.exists():time.sleep(15)
        assert json.loads(receipt.read_text())["draws"]==10000
        subprocess.run([sys.executable,str(HERE/"evaluate.py"),tag,"--workers","1"],cwd=ROOT,check=True)
        staged.append(tag)
        subprocess.run([sys.executable,str(HERE/"physical.py"),*staged,"--workers","1"],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(HERE/"paired_metrics.py")],cwd=ROOT,check=True)

