"""Prepare a formula-disjoint split of the DFT-relaxed WBM dataset.

Run python scaled/wbm.py from the study. The selected native unit cells have up
to 20 sites. Output arrays are the same format as the full MP-20 extension.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from math import gcd
from pathlib import Path
import shutil
from urllib.request import urlopen

import numpy as np

from prepare_mp20 import Z

HERE = Path(__file__).resolve().parent

SOURCE = "https://ndownloader.figshare.com/files/53161832"
FILE = "2022-10-19-wbm-computed-structure-entries.jsonl.gz"
PUBLISHED_MD5 = "655b7a9c368e136dd8747f1ef8002e7a"
N_RELEASE = 256963
WIDTH = 20
SPLITS = ("train", "val", "test")


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def source_file(directory, source):
    path = source or directory / FILE
    if not path.is_file() and source is None:
        partial = path.with_suffix(".gz.partial")
        with urlopen(SOURCE, timeout=180) as src, partial.open("wb") as out:
            shutil.copyfileobj(src, out, length=2**20)
        if digest(partial, "md5") != PUBLISHED_MD5:
            raise ValueError(f"Download failed its release checksum: {partial}")
        partial.replace(path)
    if digest(path, "md5") != PUBLISHED_MD5:
        raise ValueError(f"Source checksum does not match Matbench Discovery: {path}")
    return path


def formula_partition(species, assignments):
    counts = Counter(species)
    divisor = gcd(*counts.values())
    key = ",".join(f"{number}:{count // divisor}" for number, count in sorted(counts.items()))
    if key not in assignments:
        bucket = int.from_bytes(hashlib.sha256(("crytaldrift-wbm-v1:" + key).encode()).digest()[:8], "big") % 100
        assignments[key] = 0 if bucket < 80 else (1 if bucket < 90 else 2)
    return assignments[key]


def pack_cells(matrices):
    gram = matrices @ matrices.transpose(0, 2, 1)
    values, vectors = np.linalg.eigh(gram)
    if np.any(values <= 0) or not np.isfinite(values).all():
        raise ValueError("Invalid released metric")
    logs = (vectors * (0.5 * np.log(values))[:, None, :]) @ vectors.transpose(0, 2, 1)
    offdiag = logs[:, [0, 0, 1], [1, 2, 2]] * np.sqrt(2)
    return np.column_stack((logs.diagonal(axis1=1, axis2=2), offdiag)).astype(np.float32)


def prepare(source, out):
    ids = np.empty(N_RELEASE, dtype="U24")
    z = np.zeros((N_RELEASE, WIDTH), dtype=np.int16)
    coords = np.zeros((N_RELEASE, WIDTH, 3), dtype=np.float32)
    cells = np.empty((N_RELEASE, 3, 3), dtype=np.float64)
    split = np.empty(N_RELEASE, dtype=np.uint8)
    assignments, skips, seen, n = {}, Counter(), set(), 0
    with gzip.open(source, "rt", encoding="utf-8") as entries:
        for line in entries:
            row = json.loads(line)
            mid = row["material_id"]
            if mid in seen or len(mid) > 24:
                raise ValueError(f"Invalid or duplicate material ID {mid}")
            seen.add(mid)
            structure = row["computed_structure_entry"]["structure"]
            sites = structure["sites"]
            if not 1 <= len(sites) <= WIDTH:
                skips["site_count_outside_1_to_20"] += 1
                continue
            if any(len(site["species"]) != 1 or site["species"][0]["occu"] != 1 for site in sites):
                skips["partial_or_mixed_occupancy"] += 1
                continue
            atoms = sorted(((Z[site["species"][0]["element"]], site["abc"]) for site in sites), key=lambda x: x[0])
            species, positions = zip(*atoms)
            ids[n] = mid
            z[n, :len(sites)] = species
            coords[n, :len(sites)] = np.asarray(positions, dtype=np.float32) % 1
            cells[n] = structure["lattice"]["matrix"]
            split[n] = formula_partition(species, assignments)
            n += 1
    if len(seen) != N_RELEASE or n + sum(skips.values()) != N_RELEASE:
        raise ValueError("Incomplete source release")
    if not np.isfinite(coords[:n]).all():
        raise ValueError("Nonfinite positions")
    k = pack_cells(cells[:n])
    meta = {"source": SOURCE, "source_md5": PUBLISHED_MD5, "source_sha256": digest(source, "sha256"),
            "release_rows": len(seen), "selected_rows": n, "exclusions": dict(skips), "native_site_cap": WIDTH,
            "geometry": "DFT-relaxed native WBM cells, ordered full occupancy, no energy or stability filter",
            "split": "Group by reduced elemental formula. SHA256('crytaldrift-wbm-v1:' + sorted Z:reduced_count pairs), first 8 bytes big endian modulo 100; 0..79 train, 80..89 validation, 90..99 test. This is a new generative-model partition, independent of the published stability-classification task.",
            "splits": {}}
    for i, label in enumerate(SPLITS):
        mask = np.flatnonzero(split[:n] == i)
        dest = out / f"{label}.npz"
        partial = dest.with_suffix(".partial.npz")
        np.savez_compressed(partial, ids=ids[mask], z=z[mask], k=k[mask], f=coords[mask])
        partial.replace(dest)
        meta["splits"][label] = {"rows": len(mask), "distinct_reduced_formulas": sum(s == i for s in assignments.values()),
                                 "npz_sha256": digest(dest, "sha256")}
    path = out / "manifest.json"
    path.with_suffix(".partial").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    path.with_suffix(".partial").replace(path)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "wbm-data")
    parser.add_argument("--source", type=Path, help="An existing copy of the official compressed release.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prepare(source_file(args.out, args.source), args.out)
