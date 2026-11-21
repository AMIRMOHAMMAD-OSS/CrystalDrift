"""Prepare the exact published CrystalNN bond graphs for the formation-energy predictor."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import numpy as np
from score import HERE, make, arrays
import warnings


@lru_cache(maxsize=2)
def neighbors(cutoff):
    from pymatgen.analysis.local_env import CrystalNN

    return CrystalNN(
        distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False, search_cutoff=cutoff
    )


def batch(task):
    from pymatgen.core import Lattice, Structure
    from pymatgen.analysis.graphs import StructureGraph

    warnings.filterwarnings("ignore")
    arm, indices = task
    root = HERE / "graphs" / arm
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{indices[0]:05}-{indices[-1]:05}.npz"
    if path.exists():
        with np.load(path) as d:
            if np.array_equal(d["indices"], indices):
                return
    edges = []
    images = []
    counts = []
    errors = []
    for ix in indices:
        e = []
        o = []
        err = ""
        try:
            original = make(arm, ix)
            crystal = Structure(
                Lattice.from_parameters(*original.lattice.parameters),
                original.species,
                original.frac_coords,
            )
            try:
                graph = StructureGraph.with_local_env_strategy(crystal, neighbors(7.0))
            except Exception:
                graph = StructureGraph.with_local_env_strategy(crystal, neighbors(10.0))
            for i, j, image in graph.graph.edges(data="to_jimage"):
                e.extend(((j, i), (i, j)))
                o.extend((image, tuple(-v for v in image)))
        except Exception as exc:
            err = type(exc).__name__ + ": " + str(exc)[:180]
        edges.extend(e)
        images.extend(o)
        counts.append(len(e))
        errors.append(err)
    temp = path.with_suffix(".writing.npz")
    np.savez_compressed(
        temp,
        indices=indices,
        counts=counts,
        edges=np.array(edges, np.int16).reshape(-1, 2),
        images=np.array(images, np.int16).reshape(-1, 3),
        errors=errors,
    )
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("arms", nargs="+")
    args = p.parse_args()
    jobs = []
    for name in args.arms:
        if name == "reference":
            ix = np.arange(len(arrays(name)["z"]))
        else:
            with np.load(HERE / f"panel/{name}-cases.npz") as s:
                ix = np.sort(s["w1_sample"])
        jobs.extend((name, ix[j : j + args.chunk]) for j in range(0, len(ix), args.chunk))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for j, _ in enumerate(ex.map(batch, jobs)):
            if j % 40 == 0:
                print(1 + j, len(jobs), flush=True)


if __name__ == "__main__":
    main()
