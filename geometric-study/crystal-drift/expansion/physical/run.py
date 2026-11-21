"""Relax ordered proposals and their finite MP-20 phase reference using one CHGNet protocol."""
import argparse
import bz2
import csv
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from time import perf_counter

for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="1"
import numpy as np
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from data import unpack_gram, pack_gram


def reduced(z):
    elements, counts=np.unique(z[z>0],return_counts=True)
    return tuple(zip(elements.tolist(),(counts//np.gcd.reduce(counts)).tolist()))


def read_refs():
    raw={}
    for split in ("train","val","test"):
        path=ROOT/f"data/mp20_{split}.csv.bz2"
        if not path.exists(): path=ROOT.parent/"crystal/data"/path.name
        with bz2.open(path,"rt",encoding="utf8",newline="") as stream:
            for row in csv.DictReader(stream):
                raw[row["material_id"]]=(float(row["formation_energy_per_atom"]),float(row["e_above_hull"]))
    phases={}; train=set()
    for split in ("train","val","test"):
        with np.load(ROOT/f"scaled/mp20-data/{split}.npz") as source:
            ids,z,k,f=(source[n] for n in ("ids","z","k","f"))
        for i,mid in enumerate(ids):
            key=reduced(z[i])
            if split=="train":train.add(key)
            form,hull=raw[mid]
            if len(key)>1 and hull>.10+1e-12: continue
            rank=(form,str(mid))
            if key not in phases or rank<phases[key][0]:
                phases[key]=(rank,dict(id=str(mid),role="reference",z=z[i,z[i]>0].tolist(),
                    k=k[i].tolist(),f=f[i,z[i]>0].tolist(),split=split,source_hull=hull,source_formation=form))
    return phases,train


def prepare(pools):
    phases,train=read_refs()
    selected={}
    refs=set()
    for spec in pools:
        arm,path=spec.split("=",1)
        with np.load(path) as data:
            for i in range(16):
                z=data["z"][i]
                key=reduced(z)
                elements={e for e,count in key}
                eligible=[v[1]["id"] for comp,v in phases.items() if all(e in elements for e,count in comp)]
                refs.update(eligible)
                name=f"{arm}-{i+1:04d}"
                selected[name]=dict(id=name,role="candidate",arm=arm,index=i,source_id=str(data["ids"][i]),
                    z=z[z>0].tolist(),k=data["k"][i].tolist(),f=data["f"][i,z>0].tolist(),seen=key in train,
                    reference_ids=sorted(eligible),pool=str(Path(path).as_posix()))
    all_refs={v[1]["id"]:v[1] for v in phases.values() if v[1]["id"] in refs}
    records={**selected,**all_refs}
    for name,value in records.items():
        path=HERE/"inputs"/(name+".json")
        if path.exists():
            assert json.loads(path.read_text())==value, name
        else:path.write_text(json.dumps(value)+"\n")
    print(len(selected),"draws",len(all_refs),"unique competing phases",flush=True)
    return list(records)


def initialize():
    import relax_candidates
    relax_candidates.load_model()


def snapshot(atoms, filt):
    from ase.data import covalent_radii
    from ase.geometry import minkowski_reduce
    z=atoms.numbers
    radii=covalent_radii[z]
    d=atoms.get_all_distances(mic=True)
    lattice,_=minkowski_reduce(atoms.cell.array)
    np.fill_diagonal(d,np.linalg.norm(lattice,axis=1).min())
    return dict(energy=float(atoms.get_potential_energy()/len(atoms)),
                atom_fmax=float(np.linalg.norm(atoms.get_forces(),axis=1).max()),
                filter_fmax=float(np.linalg.norm(filt.get_forces(),axis=1).max()),
                min_contact=float(d.min()),min_ratio=float((d/(radii[:,None]+radii[None,:])).min()),
                volume_per_atom=float(atoms.get_volume()/len(atoms)))


def run(name):
    import relax_candidates as library
    from ase import Atoms
    from ase.io import write
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from chgnet.model.dynamics import CHGNetCalculator
    source=json.loads((HERE/"inputs"/(name+".json")).read_text())
    out=HERE/"records"/(name+".json")
    if out.exists():return
    started=perf_counter();record=dict(id=name,converged=False)
    try:
        atoms=Atoms(numbers=source["z"],scaled_positions=source["f"],
                    cell=np.linalg.cholesky(unpack_gram(source["k"])),pbc=True)
        if source["role"]=="candidate":write(HERE/"cifs"/(name+"-raw.cif"),atoms)
        atoms.calc=CHGNetCalculator(model=library.MODEL,use_device="cpu",on_isolated_atoms="error")
        filt=FrechetCellFilter(atoms)
        record["initial"]=snapshot(atoms,filt)
        optimizer=FIRE(filt,logfile=None)
        tic=perf_counter()
        converged=bool(optimizer.run(fmax=.10,steps=100))
        record.update(optimizer_seconds=perf_counter()-tic,steps=optimizer.nsteps,converged=converged,
                      status="converged" if converged else "step_cap")
        record["final"]=snapshot(atoms,filt)
        record["final_k"]=pack_gram(atoms.cell.array@atoms.cell.array.T).tolist()
        record["final_f"]=atoms.get_scaled_positions(wrap=True).tolist()
        if source["role"]=="candidate":write(HERE/"cifs"/(name+"-relaxed.cif"),atoms)
    except Exception as exc:
        record.update(status="error",error=type(exc).__name__+": "+str(exc),converged=False)
    record["wall_seconds"]=perf_counter()-started
    temp=out.with_suffix(".part");temp.write_text(json.dumps(record,allow_nan=False)+"\n");temp.replace(out)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pools",nargs="*");parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--prepare",action="store_true")
    args=parser.parse_args()
    for part in ("inputs","records","cifs"):(HERE/part).mkdir(parents=True,exist_ok=True)
    names=prepare(args.pools) if args.pools else [p.stem for p in sorted((HERE/"inputs").glob("*.json"))]
    if not args.prepare:
        names=[n for n in names if not (HERE/"records"/(n+".json")).exists()]
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:
            for i,_ in enumerate(pool.map(run,names,chunksize=1)):
                if i%30==0:print(i+1,len(names),flush=True)
