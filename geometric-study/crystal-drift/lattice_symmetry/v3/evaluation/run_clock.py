"""Run the three dedicated timing stages sequentially from the project root."""
import os
from pathlib import Path
import runpy
import subprocess
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="--child":
        lib=os.environ.get("CRYTALDRIFT_NUMBA_SITE")
        if lib:sys.path.append(lib)
        sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT))
        filename=sys.argv[2];sys.argv=[filename,*sys.argv[3:]]
        runpy.run_path(str(HERE/filename),run_name="__main__")
    else:
        matrix=HERE/"clock-frozen.json"
        stages=[("clock.py",[str(matrix)]),("clock.py",[str(matrix),"--throughput"]),
                ("cpu_latency.py",[str(matrix),"--uflow"]),("selector_cost.py",[])]
        for script,args in stages:
            print("Timing",script,*args,flush=True)
            subprocess.run([sys.executable,"-u",str(Path(__file__).resolve()),"--child",script,*args],cwd=ROOT,check=True)

