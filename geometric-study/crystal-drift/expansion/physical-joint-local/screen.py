"""Apply the original physical screen in an isolated secondary output folder."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from time import sleep

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIMARY = ROOT / "expansion/physical"
sys.path.insert(0, str(PRIMARY))
import run as primary
import numpy as np

ARM = "crystalflow-joint-local"
POOL = ROOT / f"expansion/denovo/pools/{ARM}.npz"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    temp = path.with_suffix(".building")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def initialize():
    primary.HERE = HERE
    primary.initialize()


def prepare(workers):
    primary.HERE = HERE
    for name in ("inputs", "records", "cifs"):
        (HERE / name).mkdir(exist_ok=True)
    with np.load(POOL, allow_pickle=False) as data:
        assert len(data["ids"]) == 10000 and len(set(data["ids"])) == 10000
        assert data["z"].shape == (10000,20) and data["k"].shape == (10000,6)
        assert data["f"].shape == (10000,20,3)
        ids = data["ids"][:16].tolist()
        z = data["z"][:16]
        raw_ok = (data["valid_output"][:16] & np.isfinite(data["k"][:16]).all(1)
                  & np.isfinite(data["f"][:16]).all((1,2)) & ((z>0).sum(1)==data["num_sites_drawn"][:16])
                  & (z>0).any(1) & (z>=0).all(1) & (z<=118).all(1))
        formulas_defined = (z>0).any(1) & (z>=0).all(1) & (z<=118).all(1)
    names = primary.prepare([f"{ARM}={POOL.relative_to(ROOT).as_posix()}"])
    candidates = [f"{ARM}-{i+1:04d}" for i in range(16)]
    assert names[:16] == candidates
    reused = {}
    new = []
    for name in names[16:]:
        input_path = HERE / "inputs" / f"{name}.json"
        from_input = PRIMARY / "inputs" / input_path.name
        from_record = PRIMARY / "records" / input_path.name
        dest = HERE / "records" / input_path.name
        if from_input.is_file() and from_record.is_file() and from_input.read_bytes() == input_path.read_bytes():
            record = json.loads(from_record.read_text())
            assert record["id"] == name and record["status"] in {"converged","step_cap","error"}
            if dest.exists():
                assert dest.read_bytes() == from_record.read_bytes()
            else:
                shutil.copyfile(from_record, dest)
            reused[name] = dict(input_sha256=digest(input_path),record_sha256=digest(from_record),
                                charged_worker_seconds=record["wall_seconds"])
        else:
            new.append(name)
    invalid = []
    for i,name in enumerate(candidates):
        source = json.loads((HERE / "inputs" / f"{name}.json").read_text())
        assert source["source_id"] == ids[i] and source["index"] == i
        if not raw_ok[i]:
            invalid.append(i+1)
            dest=HERE / "records" / f"{name}.json"
            failure=dict(id=name,converged=False,status="invalid_raw_input",wall_seconds=0.,
                         error="Raw generator returned incomplete species or nonfinite geometry.")
            if dest.exists():assert json.loads(dest.read_text())==failure
            else:save(dest,failure)
    settings=dict(design_sha256=digest(HERE/"DESIGN.md"),pool=POOL.relative_to(ROOT).as_posix(),
                  pool_sha256=digest(POOL),draws=list(range(1,17)),source_ids=ids,
                  defined_formulas=formulas_defined.tolist(),invalid_draws=invalid,
                  primary_code_sha256={n:digest(PRIMARY/n) for n in ("run.py","summarize.py")},
                  packages={p:version(p) for p in ("chgnet","ase","torch","numpy","pymatgen","spglib")},
                  model="CHGNet 0.3.0",device="CPU",torch_threads_per_worker=1,workers=workers,python=platform.python_version(),
                  reused_references=reused,new_references=new)
    import chgnet
    weight=next((Path(chgnet.__file__).parent/"pretrained").rglob("*0.3.0*.pth.tar"))
    settings["weights_sha256"]=digest(weight)
    save(HERE/"settings.json",settings)
    return candidates + new


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--wait-for-pool",action="store_true")
    args=parser.parse_args()
    os.chdir(ROOT)
    if args.wait_for_pool:
        print("Waiting for the closed 10,000-draw joint CrystalFlow pool.",flush=True)
        while not POOL.is_file():sleep(3)
    names=prepare(args.workers)
    pending=[n for n in names if not (HERE/"records"/f"{n}.json").exists()]
    print("Secondary screen",len(pending),"new candidate or reference jobs",flush=True)
    if pending:
        # Spawn-safe initializer points each process at the secondary output folder.
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as executor:
            for i,_ in enumerate(executor.map(primary.run,pending,chunksize=1)):
                if i%20==0 or i+1==len(pending):print(i+1,"/",len(pending),flush=True)
    import summarize
    primary.HERE=HERE
    summarize.HERE=HERE
    summarize.main()


if __name__=="__main__":
    main()
