"""Prepare the downsized uFlowCSP Cartesian training set from the locked original CIF rows."""
import argparse
import bz2
import csv
import hashlib
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from importlib.machinery import SourceFileLoader
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
warnings.filterwarnings("ignore")
import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
order = SourceFileLoader("official_canonical_order", str(HERE/"atom_ordering.py.txt")).load_module()

def process(item):
    mid, cif, original_group = item
    # Author preprocessing_utils.build_crystal, niggli=True and primitive=False.
    crystal = Structure.from_str(cif, fmt="cif").get_reduced_structure()
    crystal = Structure(Lattice.from_parameters(*crystal.lattice.parameters), crystal.species,
                        crystal.frac_coords, coords_are_cartesian=False)
    cell = np.asarray(crystal.lattice.matrix, dtype=np.float32)
    frac = np.asarray(crystal.frac_coords, dtype=np.float32)
    z = np.asarray(crystal.atomic_numbers, dtype=np.int64)
    # Coarse auxiliary class from spglib (0.1 Angstrom); original group kept for audit.
    sg = SpacegroupAnalyzer(crystal, symprec=0.1).get_space_group_number()
    en, orbits, ranks = order.compute_group_labels(z, frac, cell, mode="symmetry", symprec=0.1)
    perm = order.canonical_order(en, orbits, ranks, frac)
    x = np.zeros((20, 12), np.float32)
    tokens = np.zeros(20, np.int64)
    tokens[:len(z)] = z[perm]
    x[:len(z), :3] = frac[perm] @ cell
    x[:len(z), 3:] = cell.ravel()
    return mid, x, tokens, sg, original_group

def verify_source():
    tree = json.loads((ROOT/"uflowcsp-results/cartesian.json").read_text())
    source = {r["path"].split("/")[-1]:r["sha"] for r in tree["tree"]}
    for p in HERE.glob("*.py.txt"):
        data = p.read_bytes()
        assert hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest() == source[p.stem]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    verify_source()
    assert hashlib.sha256(args.csv.read_bytes()).hexdigest() == json.loads(
        (ROOT/"scaled/mp20-data/manifest.json").read_text())["splits"]["train"]["source_sha256"]
    ids = json.loads((HERE.parent/"protocol/splits.json").read_text())["train_ids"]
    with bz2.open(args.csv, "rt", encoding="utf8") as handle:
        raw = {r["material_id"]: (r["material_id"], r["cif"], int(r["spacegroup.number"]))
               for r in csv.DictReader(handle)}
    (HERE/"train-cache").mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for lo in range(0, len(ids), 512):
            path = HERE/f"train-cache/{lo:05}.npz"
            if path.exists():
                continue
            rows = list(pool.map(process, [raw[i] for i in ids[lo:lo+512]], chunksize=12))
            keys, x, z, sg, source_sg = zip(*rows)
            target = path.with_suffix(".tmp.npz")
            np.savez_compressed(target, ids=keys, x=x, z=z, sg=sg, original_group=source_sg)
            target.replace(path)
            print("prepared", lo+len(rows), "/", len(ids), flush=True)
