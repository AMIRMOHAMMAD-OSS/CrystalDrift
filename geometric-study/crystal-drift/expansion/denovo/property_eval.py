"""Run the verified DiffCSP MP-20 supervised encoder and its own scaler on frozen graphs."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import zipfile

import numpy as np
import torch
from score import HERE, arrays
from generate import cells

PROP = HERE / "property"


def encoder():
    if os.environ.get("CRYSTAL_PYG_SITE"):
        sys.path.insert(0, os.environ["CRYSTAL_PYG_SITE"])
    import sympy
    import torch.nn as nn
    from torch_scatter import scatter
    from torch_sparse import SparseTensor
    from torch_geometric.nn.models.dimenet import (
        BesselBasisLayer,
        EmbeddingBlock,
        ResidualLayer,
        SphericalBasisLayer,
    )
    from torch_geometric.nn.inits import glorot_orthogonal
    import torch.nn.functional as F

    # The swish activation in the released PyG version equals x * sigmoid(x).
    def swish(x):
        return x * torch.sigmoid(x)

    env = dict(
        torch=torch,
        nn=nn,
        np=np,
        swish=swish,
        scatter=scatter,
        SparseTensor=SparseTensor,
        sym=sympy,
        glorot_orthogonal=glorot_orthogonal,
        BesselBasisLayer=BesselBasisLayer,
        EmbeddingBlock=EmbeddingBlock,
        ResidualLayer=ResidualLayer,
        SphericalBasisLayer=SphericalBasisLayer,
    )
    manifest = json.loads((PROP / "manifest.json").read_text())["sha256"]
    for f, names in (
        (
            "diffcsp/common/data_utils.py",
            {"lattice_params_to_matrix_torch", "frac_to_cart_coords", "get_pbc_distances"},
        ),
        (
            "diffcsp/pl_modules/gnn.py",
            {"DimeNetPlusPlusWrap", "DimeNetPlusPlus", "InteractionPPBlock", "OutputPPBlock"},
        ),
    ):
        path = PROP / f
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[f]
        tree = ast.parse(path.read_text())
        extracted = ast.Module(
            body=[
                n
                for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
            ],
            type_ignores=[],
        )
        assert {n.name for n in extracted.body} == names
        exec(compile(extracted, str(path), "exec"), env)
    model = env["DimeNetPlusPlusWrap"](
        num_targets=1,
        hidden_channels=128,
        num_blocks=4,
        int_emb_size=64,
        basis_emb_size=8,
        out_emb_channels=256,
        num_spherical=7,
        num_radial=6,
        cutoff=7.0,
        max_num_neighbors=20,
        readout="mean",
    )
    folder = PROP / "diffcsp/prop_models/mp20"
    path = folder / "epoch=839-step=89039.ckpt"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == manifest[str(path.relative_to(PROP)).replace(os.sep, "/")]
    )
    weights = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict({k.removeprefix("encoder."): v for k, v in weights.items()}, strict=True)
    with zipfile.ZipFile(folder / "prop_scaler.pt") as q:
        mean, sd = (
            float(np.frombuffer(q.read("archive/data/" + str(i)), dtype="<f4")[0]) for i in (0, 1)
        )
    return model.eval(), mean, sd


def graph_rows(name):
    records = {}
    for path in sorted((HERE / "graphs" / name).glob("*.npz")):
        if path.name.endswith(".writing.npz"):
            continue
        with np.load(path) as d:
            ed, im = d["edges"], d["images"]
            offset = 0
            for i, n, err in zip(d["indices"], d["counts"], d["errors"]):
                records[int(i)] = ed[offset : offset + n], im[offset : offset + n], str(err)
                offset += n
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arms", nargs="+")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--limit", type=int)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--wait-for-graphs", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    model, mean, sd = encoder()
    for name in args.arms:
        data = arrays(name)
        ix = (
            np.arange(len(data["z"]))
            if name == "reference"
            else np.sort(np.load(HERE / f"panel/{name}-cases.npz")["w1_sample"])
        )
        ix = ix[: args.limit]
        graphs = graph_rows(name)
        while args.wait_for_graphs and not all(i in graphs for i in ix):
            time.sleep(20)
            graphs = graph_rows(name)
        if not all(i in graphs for i in ix):
            print(name, "waiting for graphs", len(graphs), "/", len(ix))
            continue
        vals = []
        errors = []
        for start in range(0, len(ix), args.batch):
            ids = ix[start : start + args.batch]
            rows = data["z"][ids]
            n = (rows > 0).sum(1)
            L = cells(data["k"][ids])
            length = np.linalg.norm(L, axis=2)
            unit = L / length[:, :, None]
            angles = np.rad2deg(
                np.arccos(
                    np.clip(
                        np.column_stack(
                            [
                                np.sum(unit[:, (j + 1) % 3] * unit[:, (j + 2) % 3], 1)
                                for j in range(3)
                            ]
                        ),
                        -1,
                        1,
                    )
                )
            )
            bonds = []
            images = []
            counts = []
            offset = 0
            for i, num in zip(ids, n):
                e, im, err = graphs[int(i)]
                if err:
                    raise ValueError(f"{name} graph failure at {i}: {err}")
                bonds.append(e + offset)
                images.append(im)
                counts.append(len(e))
                offset += num
            batch = SimpleNamespace(
                atom_types=torch.LongTensor(rows[rows > 0]),
                frac_coords=torch.FloatTensor(data["f"][ids][rows > 0]),
                num_atoms=torch.LongTensor(n),
                num_bonds=torch.LongTensor(counts),
                lengths=torch.FloatTensor(length),
                angles=torch.FloatTensor(angles),
                edge_index=torch.LongTensor(np.concatenate(bonds).T),
                to_jimages=torch.LongTensor(np.concatenate(images)),
                batch=torch.arange(len(ids)).repeat_interleave(torch.LongTensor(n)),
            )
            with torch.inference_mode():
                out = model(batch)[:, 0].numpy() * sd + mean
            vals.extend(out)
        dest = HERE / "energies"
        dest.mkdir(exist_ok=True)
        assert np.isfinite(vals).all()
        path = dest / (name + ("-partial" if args.limit else "") + ".npz")
        np.savez_compressed(path, index=ix, ids=data["ids"][ix], energy_eV_per_atom=vals)
        print(name, len(vals), float(np.mean(vals)), flush=True)


if __name__ == "__main__":
    main()
