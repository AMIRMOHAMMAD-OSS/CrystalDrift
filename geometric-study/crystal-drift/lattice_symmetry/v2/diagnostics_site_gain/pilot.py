"""Fixed validation intervention on the Cartesian site step of the released generator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[var] = "1"
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
from data import pack_gram


def generate():
    import torch
    from lattice_symmetry.v2.csp.model import Prior, checkpoint_path, make_cell, new_model
    from lattice_symmetry.v2 import physical

    class SiteGain(physical.Generator):
        """The frozen model computes the exact Cartesian site tangent on the original graph.

        Capture its fractional linear solve before wrapping; apply one scalar gain.
        The returned cell and all learned weights are those of the original model.
        """
        def forward(self, cell, frac, z):
            from unittest.mock import patch

            solve = torch.linalg.solve
            solutions = []

            def save_solve(a, b, **kwargs):
                solution = solve(a, b, **kwargs)
                solutions.append(solution)
                return solution

            with patch.object(torch.linalg, "solve", save_solve):
                output_cell, base = super().forward(cell, frac, z)
            matching = [v for v in solutions if v.shape == (len(z), 3, z.shape[1])]
            if len(matching) != 1:
                raise RuntimeError("Expected exactly one unwrapped fractional site solve.")
            tangent = matching[0].transpose(-1, -2)
            mask = (z > 0).unsqueeze(-1)
            torch.testing.assert_close(base, (frac + tangent).remainder(1) * mask)
            return output_cell, base, (frac + 2 * tangent).remainder(1) * mask

    torch.set_num_threads(1)
    with np.load(ROOT / "scaled/mp20-data/val.npz") as data:
        ids, elements = data["ids"], data["z"]
    indices = np.array(sorted(range(len(ids)), key=lambda i: hashlib.sha256(str(ids[i]).encode()).hexdigest())[640:832])
    checkpoint = checkpoint_path("angular", 11)
    model = new_model("angular")
    gain_model = SiteGain(cutoff=model.cutoff, width=24, n_radial=5).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    gain_model.load_state_dict(state["model"], strict=True)
    prior = Prior.load(checkpoint.parents[2] / "prior.npz")
    generated_cell, original, double = [], [], []
    rng = np.random.default_rng(113320)
    with torch.inference_mode():
        for group in np.array_split(indices, 24):
            z = elements[np.repeat(group, 5)].astype("int64")
            k, f = prior.draw(z, rng)
            cell, base, altered = gain_model(make_cell(torch.from_numpy(k)), torch.from_numpy(f), torch.from_numpy(z))
            L = cell.double().numpy()
            generated_cell.extend(pack_gram(L @ L.transpose(0, 2, 1)))
            original.extend(base.numpy())
            double.extend(altered.numpy())
    np.savez_compressed(OUT / "proposals.npz", indices=indices, ids=ids[indices],
                        k=np.asarray(generated_cell).reshape(len(indices), 5, 6),
                        base=np.asarray(original).reshape(len(indices), 5, 20, 3),
                        double=np.asarray(double).reshape(len(indices), 5, 20, 3),
                        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())


def score():
    from importlib.metadata import version
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from standard_match import SETTINGS, make_structure
    if version("pymatgen") != "2024.4.13":
        raise ValueError("Use the release's pinned pymatgen.")
    matcher = StructureMatcher(**SETTINGS)
    with np.load(OUT / "proposals.npz") as p, np.load(ROOT / "scaled/mp20-data/val.npz") as d:
        ix, ids, k, f, f2 = (p[key] for key in ("indices", "ids", "k", "base", "double"))
        kr, fr, zs = (d[key][ix] for key in ("k", "f", "z"))
        np.testing.assert_array_equal(d["ids"][ix], ids)
        digest = str(p["checkpoint_sha256"])
    errors = np.full((len(ix), 5, 2), np.nan)
    volume = np.abs(k[:, :, :3].sum(-1) - kr[:, :3].sum(-1)[:, None]) <= np.log(1.1)
    for i, z in enumerate(zs):
        truth = make_structure(kr[i], fr[i], z)
        for j in range(5):
            for arm, coords in enumerate((f, f2)):
                result = matcher.get_rms_dist(make_structure(k[i, j], coords[i, j], z), truth)
                if result is not None:
                    errors[i, j, arm] = result[0]
    np.savez_compressed(OUT / "outcomes.npz", ids=ids, rms=errors, volume_pass=volume)
    summary = dict(validation_targets=len(ix), draws=5, checkpoint_sha256=digest, inference_calls=1,
                   rng=113320, comparison="gain one versus two on the unwrapped physical site shift only", metrics={})
    for n in (1, 5):
        for name, flags in (("scaled", np.isfinite(errors)), ("native", np.isfinite(errors) & volume[:, :, None])):
            matched = flags[:, :n].any(1)
            delta = matched[:, 1].astype(float) - matched[:, 0]
            boot = np.random.default_rng(443).choice(delta, size=(5000, len(ix)), replace=True).mean(1)
            summary["metrics"][f"{name}@{n}"] = dict(counts=matched.sum(0).tolist(), delta=float(delta.mean()),
                                                     paired95=np.quantile(boot, [.025, .975]).tolist())
    (OUT / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("generate", "score"))
    if p.parse_args().stage == "generate":
        generate()
    else:
        score()
