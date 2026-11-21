"""Compute released DiffCSP fingerprints, raw per-draw validity, and train-set novelty."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import warnings

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get("CRYSTAL_METRICS_SITE"):
    sys.path.insert(0, os.environ["CRYSTAL_METRICS_SITE"])
from standard_match import make_structure, SETTINGS
from prior import split_formula
from math import gcd, prod
from functools import reduce
from itertools import product


def formula(row):
    z, n = split_formula(row)
    g = reduce(gcd, n)
    return tuple(zip(z, (k // g for k in n)))


@lru_cache(maxsize=None)
def valid_chem(form):
    from pymatgen.core import Element
    import smact
    from smact.screening import pauling_test

    symbols = tuple(Element.from_Z(z).symbol for z, n in form)
    if len(form) == 1 or all(s in smact.metals for s in symbols):
        return True
    elements = smact.element_dictionary(symbols)
    nums = [v[1] for v in form]
    states = [elements[s].oxidation_states for s in symbols]
    ens = [elements[s].pauling_eneg for s in symbols]
    if prod(map(len, states)) > 10**7:
        return False
    for charges in product(*states):
        if sum(v * n for v, n in zip(charges, nums)):
            continue
        try:
            if pauling_test(charges, ens):
                return True
        except TypeError:
            return True
    return False


@lru_cache(maxsize=1)
def fp_tools():
    from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
    from matminer.featurizers.composition import ElementProperty

    if os.environ.get("CRYSTAL_COMPILED_FINGERPRINT") == "1":
        from fast_ops import install
        from fast_faces import install_faces

        install()
        install_faces()
    return CrystalNNFingerprint.from_preset("ops"), ElementProperty.from_preset("magpie")


@lru_cache(maxsize=30000)
def magpie(form):
    from pymatgen.core import Composition, Element

    return fp_tools()[1].featurize(Composition({Element.from_Z(z): n for z, n in form}))


@lru_cache(maxsize=3)
def arrays(name):
    path = (
        ROOT / "expansion/external/crystalflow-joint.npz"
        if name == "crystalflow-joint"
        else (
            ROOT / "scaled/mp20-data/test.npz"
            if name == "reference"
            else (
                ROOT / "scaled/mp20-data/train.npz"
                if name == "training"
                else HERE / f"pools/{name}.npz"
            )
        )
    )
    with np.load(path) as bundle:
        data = {key: bundle[key] for key in ("z", "k", "f", "ids")}
        output_flag = next((key for key in ("valid_output", "present") if key in bundle), None)
        data["valid_output"] = (
            bundle[output_flag].astype(bool) if output_flag else np.ones(len(data["ids"]), bool)
        )
        return data


def make(name, index):
    a = arrays(name)
    if not a["valid_output"][index]:
        raise ValueError("The archived generator marked this draw as failed")
    return make_structure(a["k"][index], a["f"][index], a["z"][index])


def worker(item):
    name, start, stop, kind = item
    a = arrays(name)
    out = HERE / "scores" / kind / name
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{start:05}-{stop:05}.npz"
    if path.exists():
        return
    begin = time.perf_counter()
    z = a["z"]
    rows = []
    for i in range(start, stop):
        failure = ""
        if not np.any(z[i] > 0) or not a["valid_output"][i]:
            failure = "empty released candidate" if not np.any(z[i] > 0) else "failed generator output"
            rows.append(
                (False, False, np.nan, np.nan, np.nan, 0, failure)
                if kind == "raw"
                else (np.full(132, np.nan), np.full(61, np.nan), failure)
            )
            continue
        if kind == "raw":
            form = formula(z[i])
            try:
                chem = valid_chem(form)
            except (ValueError, TypeError, KeyError) as exc:
                chem = False
                failure = f"chemistry: {type(exc).__name__}: {str(exc)[:100]}"
            try:
                cell = make(name, i)
                dist = cell.distance_matrix.copy()
                np.fill_diagonal(dist, np.inf)
                shortest = float(min(10.5, dist.min()))
                density = float(cell.density)
                volume = float(cell.volume)
                struct = shortest >= 0.5 and volume >= 0.1
            except Exception as e:
                failure = type(e).__name__
                shortest = density = volume = np.nan
                struct = False
            rows.append((chem, struct, density, volume, shortest, len(form), failure))
        else:
            try:
                cell = make(name, i)
                cf = magpie(formula(z[i]))
                extractor = fp_tools()[0]
                sf = np.mean([extractor.featurize(cell, j) for j in range(len(cell))], axis=0)
                if not np.isfinite(sf).all():
                    raise ValueError("nonfinite site features")
            except Exception as e:
                failure = type(e).__name__ + ": " + str(e)[:140]
                sf = np.full(61, np.nan)
                cf = np.full(132, np.nan)
            rows.append((cf, sf, failure))
    keys = (
        ("comp_valid", "struct_valid", "density", "volume", "shortest", "nelem", "error")
        if kind == "raw"
        else ("comp", "struc", "error")
    )
    data = {k: np.array(x) for k, x in zip(keys, zip(*rows))}
    tmp = path.with_suffix(f".writing-{os.getpid()}.npz")
    np.savez_compressed(tmp, ids=a["ids"][start:stop], elapsed=time.perf_counter() - begin, **data)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=("raw", "fp"), required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument(
        "--arms",
        nargs="+",
        default=["reference", "learned-drift", "empirical-drift", "learned-flow16"],
    )
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--limit", type=int)
    p.add_argument(
        "--compiled", action="store_true", help="Use the parity-tested compiled CrystalNN kernels"
    )
    p.add_argument(
        "--reverse", action="store_true", help="Consume tiles from the end of each ordered bank"
    )
    args = p.parse_args()
    jobs = []
    if args.compiled:
        os.environ["CRYSTAL_COMPILED_FINGERPRINT"] = "1"
    for name in args.arms:
        n = len(arrays(name)["z"])
        n = min(n, args.limit or n)
        jobs.append(
            [
                (name, i, min(n, i + args.chunk), args.kind)
                for i in range(0, n, args.chunk)
                if not (
                    HERE / "scores" / args.kind / name / f"{i:05}-{min(n,i+args.chunk):05}.npz"
                ).exists()
            ]
        )
    # Interleave arms to reveal method-specific featurization failures early.
    from itertools import zip_longest

    todo = [j for pack in zip_longest(*jobs) for j in pack if j]
    if args.reverse:
        todo.reverse()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, _ in enumerate(executor.map(worker, todo, chunksize=1)):
            if i % 20 == 0:
                print(i + 1, "/", len(todo), flush=True)


if __name__ == "__main__":
    main()
