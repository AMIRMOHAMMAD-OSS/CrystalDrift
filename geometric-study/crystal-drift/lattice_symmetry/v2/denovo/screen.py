"""Run the frozen 16-draw CHGNet protocol with checked MP-20 competing phases."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
STORE=HERE/"screen"
PRIMARY=ROOT/"expansion/physical"
sys.path.insert(0,str(PRIMARY))
import run as physical
import summarize


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize():
    physical.HERE=STORE
    physical.initialize()


def main(workers):
    os.chdir(ROOT)
    complete=STORE/"settings.json"
    if complete.is_file():
        settings=json.loads(complete.read_text())
        assert settings["design_sha256"]==sha(HERE/"DESIGN.md")
        for arm,saved_hash in settings["pool_sha256"].items():
            assert sha(HERE/"pools"/(arm+".npz"))==saved_hash
        assert (STORE/"by-arm.csv").is_file() and (STORE/"per-candidate.csv").is_file()
        assert settings["phase_library"]==sha(STORE/"reference-library.json")
        print("Verified existing complete physical-screen result.")
        return
    for name in ("inputs","records","cifs"):
        (STORE/name).mkdir(parents=True,exist_ok=True)
    physical.HERE=summarize.HERE=STORE
    arms=("angular","radial")
    pools={arm:HERE/"pools"/(arm+".npz") for arm in arms}
    names=physical.prepare([f"{arm}={path.relative_to(ROOT).as_posix()}" for arm,path in pools.items()])
    reused={}
    for name in names:
        src=STORE/"inputs"/(name+".json")
        data=json.loads(src.read_text())
        dest=STORE/"records"/(name+".json")
        if data["role"]=="reference":
            prior_in=PRIMARY/"inputs"/src.name
            prior_out=PRIMARY/"records"/dest.name
            if prior_in.is_file() and prior_out.is_file() and src.read_bytes()==prior_in.read_bytes():
                assert json.loads(prior_out.read_text())["id"]==name
                if dest.exists():
                    assert dest.read_bytes()==prior_out.read_bytes()
                else:
                    shutil.copyfile(prior_out,dest)
                reused[name]={"input":sha(src),"record":sha(dest)}
    for arm,path in pools.items():
        with np.load(path) as pool:
            for i in range(16):
                name=f"{arm}-{i+1:04d}"
                row=json.loads((STORE/"inputs"/(name+".json")).read_text())
                assert row["source_id"]==pool["ids"][i] and row["index"]==i
                if not pool["valid_output"][i]:
                    fail={"id":name,"converged":False,"status":"invalid_raw","wall_seconds":0.}
                    (STORE/"records"/(name+".json")).write_text(json.dumps(fail)+"\n")
    pending=[name for name in names if not (STORE/"records"/(name+".json")).exists()]
    if pending:
        with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as exe:
            for i,_ in enumerate(exe.map(physical.run,pending)):
                if i%8==0:print(i+1,"/",len(pending),flush=True)
    summarize.main()
    settings=dict(design_sha256=sha(HERE/"DESIGN.md"),reused_phases=reused,
        pool_sha256={key:sha(p) for key,p in pools.items()},
        evaluator_sha256={n:sha(PRIMARY/n) for n in ("run.py","summarize.py")},
        phase_library=sha(STORE/"reference-library.json"),
        packages={n:version(n) for n in ("chgnet","pymatgen","spglib","ase","torch")},
        protocol="CHGNet 0.3.0, FIRE/FrechetCellFilter, 0.10 eV/Å, 100 steps, zero pressure",
        workers=workers, threads_per_worker=1)
    (STORE/"settings.json").write_text(json.dumps(settings,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers",type=int,default=2)
    main(p.parse_args().workers)
