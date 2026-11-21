"""Independent CPU CHGNet relaxation of a fixed generated cohort and MP-20 controls."""
import argparse
import bz2
import csv
import hashlib
import json
import os
import platform
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import data

HERE = Path(__file__).resolve().parent
ROOT = HERE / "de-novo"
OUTPUT = ROOT / "relaxation"
EXPECTED = "b01ccf21e60da1a2919bc5e4baf2c146047f6eac97dc8088420eb77d885dba56"
MODEL = None
FMAX = .1
STEPS = 100


def formula_key(z):
    numbers,counts=np.unique(np.asarray(z,dtype=int),return_counts=True)
    return tuple(zip(map(int,numbers), map(int,counts//np.gcd.reduce(counts))))


def selection():
    path=ROOT/"candidates.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==EXPECTED
    batch=np.load(path); held=np.load(data.DATA/"multi-test.npz")
    with (ROOT/"manifest.csv").open(encoding="utf-8") as f:
        labels={r["id"]:r for r in csv.DictReader(f)}
    entries={}
    def add(mid,role,species,coord,G,split=""):
        entries[mid]=dict(sample_id=mid,role=role,z=np.asarray(species,int).tolist(),
                          f=np.asarray(coord,float).tolist(),G=np.asarray(G,float).tolist(),split=split)
    wanted={}
    for mid,zz,kk,ff in zip(batch["ids"],batch["z"],batch["k"],batch["f"]):
        mask=zz>0
        add(str(mid),"generated",zz[mask],ff[mask],data.unpack_gram(kk))
        wanted.setdefault(formula_key(zz[mask]),[]).append(str(mid))
        entries[mid]["generator_formula"]=labels[mid]["formula"]
    control_indices=np.random.default_rng(20261120).choice(len(held["ids"]),size=12,replace=False)
    for i in control_indices:
        zz=held["z"][i];mask=zz>0
        add(str(held["ids"][i]),"random_test_control",zz[mask],held["f"][i][mask],held["G"][i],"test")
    best={}; counts=Counter()
    for rank,split in enumerate(("test","val","train")):
        p=data.DATA/f"mp20_{split}.csv.bz2"
        if not p.exists():p=HERE.parent/"crystal"/"data"/p.name
        with bz2.open(p,"rt",encoding="utf-8",newline="") as f:
            for row in csv.DictReader(f):
                head,_,atoms=row["cif"].partition("_atom_site_occupancy\n")
                sites=[line.split() for line in atoms.strip().splitlines()]
                if len({v[0] for v in sites})!=2:continue
                sites.sort(key=lambda v:data.Z[v[0]])
                zz=[data.Z[v[0]] for v in sites]; key=formula_key(zz)
                if key not in wanted:continue
                counts[key]+=1
                priority=(rank,len(zz),row["material_id"])
                if key in best and best[key][0]<=priority:continue
                fields={v.split(maxsplit=1)[0]:v.split(maxsplit=1)[1] for v in head.splitlines() if v.startswith("_cell_")}
                xyz=[[float(t) for t in v[3:6]] for v in sites]
                best[key]=(priority,row["material_id"],split,zz,xyz,data.cell_gram(fields))
    for key,mid_list in wanted.items():
        if key not in best:continue
        _,mid,split,zz,ff,G=best[key]
        if mid not in entries:add(mid,"same_formula_reference",zz,ff,G,split)
        for m in mid_list:
            entries[m]["same_formula_reference"]=mid
            entries[m]["available_reference_structures"]=counts[key]
    selection_path=OUTPUT/"selection.json"
    selection_path.write_text(json.dumps(list(entries.values()),indent=1)+"\n")
    return list(entries.values())


def load_model():
    global MODEL
    import torch
    from chgnet.model.model import CHGNet
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    MODEL=CHGNet.load(model_name="0.3.0", use_device="cpu")
    MODEL.eval()


def physical_snapshot(atoms, filt):
    from ase import units
    from ase.data import covalent_radii
    from ase.geometry import minkowski_reduce
    z=atoms.get_atomic_numbers()
    G=atoms.cell.array@atoms.cell.array.T
    k=data.pack_gram(G)
    x=atoms.get_scaled_positions(wrap=True)
    elements=np.unique(z)
    if len(elements)==1:
        lattice,_=minkowski_reduce(atoms.cell.array)
        shortest=float(np.linalg.norm(lattice,axis=1).min())
        if len(atoms)>1:
            d=atoms.get_all_distances(mic=True);np.fill_diagonal(d,np.inf)
            shortest=min(shortest,float(d.min()))
        distances=np.array([shortest]);reference=np.array([2*covalent_radii[elements[0]]])
    else:
        distances=data.geometry_features(k[None],x[None],z[None])[0,1:]
        radii=covalent_radii[elements]
        reference=np.array([2*radii[0],radii.sum(),2*radii[1]])
    return dict(energy_eV_per_atom=float(atoms.get_potential_energy()/len(atoms)),
                max_force_eV_per_A=float(np.linalg.norm(atoms.get_forces(),axis=1).max()),
                generalized_fmax_eV_per_A=float(np.linalg.norm(filt.get_forces(),axis=1).max()),
                max_abs_stress_GPa=float(np.abs(atoms.get_stress(voigt=False)).max()/units.GPa),
                volume_per_atom_A3=float(atoms.get_volume()/len(atoms)),
                min_contact_A=float(distances.min()), min_covalent_radius_ratio=float((distances/reference).min()))


def relax(item):
    from ase import Atoms
    from ase.filters import FrechetCellFilter
    from ase.geometry import find_mic
    from ase.io import write
    from ase.optimize import FIRE
    from chgnet.model.dynamics import CHGNetCalculator
    key=item["sample_id"]; recorded=OUTPUT/"records"/f"{key}.json"
    if recorded.exists():
        old=json.loads(recorded.read_text())
        if old.get("protocol_version")==1:return old
    t=perf_counter(); z=np.array(item["z"],int)
    record=dict(protocol_version=1,sample_id=key,role=item["role"],split=item["split"],
                species_formula=item.get("generator_formula",""),reference_id=item.get("same_formula_reference",""),
                available_reference_structures=item.get("available_reference_structures",0),n_atoms=len(z))
    try:
        initial=np.asarray(item["f"])
        atoms=Atoms(numbers=z,scaled_positions=initial,cell=np.linalg.cholesky(item["G"]),pbc=True)
        initial_cell=atoms.cell.array.copy()
        write(OUTPUT/"cifs"/f"{key}-initial.cif",atoms)
        calc=CHGNetCalculator(model=MODEL,use_device="cpu",on_isolated_atoms="error")
        atoms.calc=calc
        filt=FrechetCellFilter(atoms)
        t0=perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            first=physical_snapshot(atoms,filt);record["initial_evaluation_seconds"]=perf_counter()-t0
            record.update({"initial_"+k:v for k,v in first.items()})
            optimizer=FIRE(filt,logfile=None)
            t1=perf_counter();done=bool(optimizer.run(fmax=FMAX,steps=STEPS));duration=perf_counter()-t1
            last=physical_snapshot(atoms,filt)
            if not all(np.isfinite(list(last.values()))):raise FloatingPointError("Nonfinite observable")
            record.update({"final_"+k:v for k,v in last.items()})
            record["warning_messages"]=sorted({str(v.message) for v in captured})
        cell=atoms.cell.array
        frac=atoms.get_scaled_positions(wrap=True)
        displacement,_=find_mic((frac-initial)@cell,cell,pbc=True)
        displacement-=displacement.mean(axis=0)
        stretches=np.linalg.svd(np.linalg.solve(initial_cell,cell),compute_uv=False)
        record.update(status="converged" if done else "step_cap",converged=done,steps=optimizer.nsteps,
                      relaxation_seconds=duration, energy_change_eV_per_atom=last["energy_eV_per_atom"]-first["energy_eV_per_atom"],
                      fractional_volume_change=last["volume_per_atom_A3"]/first["volume_per_atom_A3"]-1.,
                      rms_internal_displacement_A=float(np.linalg.norm(displacement)/np.sqrt(len(z))),
                      max_abs_log_principal_stretch=float(np.abs(np.log(stretches)).max()),
                      final_k=data.pack_gram(cell@cell.T).tolist(),final_f=frac.tolist())
        write(OUTPUT/"cifs"/f"{key}-relaxed.cif",atoms)
    except Exception as exc:
        record.update(status="error",converged=False,exception=f"{type(exc).__name__}: {exc}")
    record["wall_seconds"]=perf_counter()-t
    tmp=recorded.with_suffix(".writing");tmp.write_text(json.dumps(record,indent=1,allow_nan=False)+"\n");tmp.replace(recorded)
    print(key,record["role"],record["status"],record.get("steps"),round(record["wall_seconds"],2),flush=True)
    return record


def save_results(records):
    byid={r["sample_id"]:r for r in records}
    for r in records:
        ref=byid.get(r["reference_id"])
        if ref and r.get("final_energy_eV_per_atom") is not None and ref.get("final_energy_eV_per_atom") is not None:
            r["delta_vs_reference_final_eV_per_atom"]=r["final_energy_eV_per_atom"]-ref["final_energy_eV_per_atom"]
            r["reference_converged"]=ref["converged"]
    keys=sorted(set().union(*(r.keys() for r in records))-{"final_k","final_f","warning_messages"})
    with (OUTPUT/"results.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore");w.writeheader();w.writerows(records)
    b=np.load(ROOT/"candidates.npz");k=np.full_like(b["k"],np.nan);x=np.full_like(b["f"],np.nan)
    good=np.zeros(len(k),bool);converged=good.copy()
    for i,m in enumerate(b["ids"]):
        r=byid.get(m)
        if r and "final_k" in r:
            k[i]=r["final_k"];x[i]=0.;x[i,:r["n_atoms"]]=r["final_f"];good[i]=True;converged[i]=r["converged"]
    np.savez_compressed(OUTPUT/"relaxed-candidates.npz",k=k,f=x,z=b["z"],ids=b["ids"],successful=good,converged=converged)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    for d in (OUTPUT/"records",OUTPUT/"cifs"):d.mkdir(parents=True,exist_ok=True)
    items=selection()
    print("Fixed cohort",Counter(r["role"] for r in items),flush=True)
    if args.prepare:return
    if args.smoke:
        load_model()
        control=next(r for r in items if r["role"]=="random_test_control")
        result=relax(control)
        if result["status"]=="error":raise RuntimeError(result["exception"])
        return
    t=perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=load_model) as exe:
        records=list(exe.map(relax,items))
    save_results(records)
    import chgnet
    weights=next((Path(chgnet.__file__).parent/"pretrained").rglob("*0.3.0*.pth.tar"))
    run_info=dict(packages={p:version(p) for p in ("chgnet","ase","torch","numpy","pymatgen","spglib")},
                  weights_name=weights.name,weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                  source="https://github.com/CederGroupHub/chgnet/tree/v0.3.8",device="cpu",
                  requested_workers=args.workers,torch_threads_per_worker=1,python=platform.python_version(),computer=platform.processor(),
                  run_wall_seconds=perf_counter()-t,candidate_sha256=EXPECTED,optimizer="ASE FIRE, FrechetCellFilter",
                  fmax=FMAX,max_steps=STEPS,random_control_seed=20261120)
    (OUTPUT/"run.json").write_text(json.dumps(run_info,indent=2)+"\n")


if __name__=="__main__":
    main()
