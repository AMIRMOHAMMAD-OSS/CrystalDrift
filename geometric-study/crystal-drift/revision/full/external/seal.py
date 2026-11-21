"""Seal a complete published-model replay with candidate-level scores and hashes."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from report import HERE, OUT, collect, archive_check


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    with np.load(HERE.parent / "mp20/cohort.npz") as d:
        ids, z, release = d["ids"], d["z"], d["release_index"]
    n = len(ids)
    rows = collect(ids, 20)
    if rows is None:
        raise SystemExit("Some matched target/draw tiles are still pending.")
    rms, strict, logv, audit = rows
    parity = archive_check(ids, 20, rms)
    found = np.zeros((n,20), bool)
    hashes = {}
    secs = Counter()
    for path in sorted(OUT.glob("draw*-cases*.npz")):
        if path.name.endswith(".part.npz"):
            continue
        with np.load(path) as d:
            start, draw = int(d["first"]), int(d["draw"])
            end = start + len(d["ids"])
            np.testing.assert_array_equal(d["ids"], ids[start:end])
            np.testing.assert_array_equal(d["z"], z[start:end])
            assert np.isfinite(d["k"]).all() and np.isfinite(d["f"]).all()
            assert not found[start:end, draw].any()
            found[start:end, draw] = True
            key = "copied_cpu_prefix" if end <= 128 and draw < 5 else "gpu_inferred"
            secs[key] += float(d["seconds"])
        hashes[path.relative_to(OUT).as_posix()] = sha(path)
    assert found.all() and len(hashes) == 720
    for path in sorted((OUT / "matches").glob("cases*.npz")):
        hashes[path.relative_to(OUT).as_posix()] = sha(path)
    assert len(hashes) == 828
    crosswalk = OUT / "complete-crosswalk-k20.npz"
    np.savez_compressed(crosswalk,ids=ids,z=z,release_index=release,draw_index=np.arange(20),
                        rms=rms,strict_fit=strict,log_volume_ratio=logv)
    manifest = dict(targets=n, draws=20, candidates=n*20, copied_cpu_candidates=640,
                    additional_gpu_candidates=n*20-640, finite_geometry=int(found.sum()),
                    case_id_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                    chemistry_npz_sha256=sha(HERE/"chemistry.npz"), crosswalk_sha256=sha(crosswalk),
                    prefix_parity=parity, audit=audit, summed_tile_seconds=secs, tile_sha256=hashes)
    path = OUT / "complete-manifest.json"
    path.write_text(json.dumps(manifest,indent=2)+"\n")
    print(path, manifest["candidates"], manifest["audit"])


if __name__ == "__main__":
    main()
