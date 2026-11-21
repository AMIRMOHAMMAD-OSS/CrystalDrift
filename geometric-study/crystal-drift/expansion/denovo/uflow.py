"""Generate the released one-step Cartesian model on a train-fitted composition prior."""

import argparse
from pathlib import Path
import json
import sys
import tempfile
import time

import numpy as np
import torch

from generate import HERE, ROOT, cells
from prior import CompositionPrior

sys.path.insert(0, str(ROOT / "revision/full/external"))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "external_pinned", ROOT / "revision/full/external/generate.py"
)
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--site", nargs="*", default=[])
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    sys.path.extend(args.site)
    out = HERE / "pools"
    ext_dir = tempfile.TemporaryDirectory()
    ext.official_sources(Path(ext_dir.name))
    import timm
    from src.models.denoisers.meanflow_sit_raw_formula_atomwise_system import (
        MeanFlowSiTRawFormulaAtomwiseSystem,
        build_sg_to_crystal_system,
    )
    from src.models.meanflow_transport import MeanFlowTransport

    assert timm.__version__ == "1.0.21"
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.device("meta"):
        model = MeanFlowSiTRawFormulaAtomwiseSystem(
            input_size=100,
            atom_type_vocab_size=120,
            hidden_size=768,
            depth=12,
            num_heads=12,
            predict_atom_types=False,
            use_spacegroup_conditioning=False,
            use_formula_embedding=True,
            use_atomwise_features=True,
            use_crystal_system_embedding=True,
        )
    model.load_state_dict(ext.safe_weights(args.weights), strict=True, assign=True)
    with np.load(ROOT / "revision/full/external/chemistry.npz") as t:
        model.atomic_feature_table = torch.from_numpy(t["table"].copy())
        key = t["sort_key"]
    model.sg_to_crystal_system = build_sg_to_crystal_system(230)
    model = model.to("cuda").eval()
    trans = MeanFlowTransport(
        torus_coords=False,
        lattice_per_structure=False,
        proj_lattice=False,
        proj_coords_anchor=False,
        proj_coords_rotavg=False,
    )
    with np.load(out / "learned-drift.npz") as pool:
        z, ids = pool["z"], pool["ids"]
    with np.load(ROOT / "scaled/mp20-data/train.npz") as tr:
        prior = CompositionPrior(tr["z"], 4)

    # Reproduce the original 128-composition request stream to preserve its draw order.
    def prior_stream():
        rng = np.random.default_rng(22112026)
        for i in range(0, 10000, 128):
            yield prior.sample(rng, min(128, 10000 - i), "learned")

    with torch.inference_mode():

        def compute(zb):
            a = np.zeros((len(zb), 100), np.int64)
            for i, row in enumerate(zb):
                s = row[row > 0]
                a[i, : len(s)] = sorted(s, key=lambda j: (key[j], j))
            types = torch.as_tensor(a, device="cuda")
            noise = trans.sample_prior(len(zb), 100, 12, "cpu").to("cuda")
            conditions = dict(
                atom_types=types,
                mask=types != 0,
                spacegroup=torch.zeros(len(a), dtype=torch.long, device="cuda"),
                generate_atom_types=False,
            )
            output = trans.sample(model, noise, conditions, num_steps=1)
            k, f = ext.structures(output, types, zb)
            L = cells(k)
            X = f @ L
            return k, f, L

        for _ in range(3):
            compute(z[: args.batch])
        torch.manual_seed(241126)
        t0 = time.perf_counter()
        generated_z = np.concatenate(list(prior_stream()))
        np.testing.assert_array_equal(generated_z, z)
        sampled = time.perf_counter() - t0
        parts = [compute(generated_z[j : j + args.batch]) for j in range(0, len(z), args.batch)]
        k, f, L = map(np.concatenate, zip(*parts))
        elapsed = time.perf_counter() - t0
        if args.replay:
            with np.load(out / "learned-uflow.npz") as bank:
                dk = abs(k - bank["k"]).max(1)
                delta = f - bank["f"]
                df = (abs(delta - np.rint(delta)) * (z > 0)[:, :, None]).max((1, 2))
            np.savez_compressed(
                out / "learned-uflow-gpu-parity.npz",
                lattice_chart_max_difference=dk,
                fractional_max_difference=df,
            )
        else:
            np.savez_compressed(
                out / "learned-uflow.npz",
                ids=ids,
                z=z,
                k=k,
                f=f,
                lattice=L,
                valid_output=np.isfinite(k).all(1) & np.isfinite(f).all((1, 2)),
            )
        rng = np.random.default_rng(20262026)
        ms = []
        for _ in range(131):
            t = time.perf_counter()
            compute(prior.sample(rng, 1, "learned"))
            ms.append((time.perf_counter() - t) * 1000)
    record = dict(
        n=10000,
        elapsed_s=elapsed,
        composition_seconds=sampled,
        batch=args.batch,
        b1_ms=ms[3:],
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        timm=timm.__version__,
        seed=241126,
        weight_sha256=ext.WEIGHT_SHA,
        source_commit=ext.COMMIT,
        padded_tokens=100,
        geometry_output="FP32 network with species, log lattice, fractional and Cartesian decoded host arrays",
        nfe=1,
    )
    if args.replay:
        record.update(
            matched_latents=True,
            chart_linf=float(dk.max()),
            site_periodic_linf=float(df.max()),
            above_tolerance=int(((dk > 1e-4) | (df > 1e-4)).sum()),
            tolerance=1e-4,
        )
    path = out / ("learned-uflow-gpu-matched.json" if args.replay else "learned-uflow.json")
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(record["elapsed_s"])


if __name__ == "__main__":
    main()
