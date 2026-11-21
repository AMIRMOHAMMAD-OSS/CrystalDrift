"""Draw complete unrelaxed cells as CIF files from a trained crystal drifting model."""
import argparse
import re
from pathlib import Path

import numpy as np
import torch

from data import ELEMENTS, Z, unpack_gram
from model import CrystalGenerator


HERE = Path(__file__).resolve().parent


def atom_list(formula):
    parts = re.findall(r"([A-Z][a-z]?)([1-9][0-9]*)?", formula)
    if "".join(s+n for s, n in parts) != formula or len(parts) != 2:
        raise ValueError("Supply two distinct elements and their atom counts in the unit cell, e.g. Mg2Zn4.")
    if any(s == "X" or s not in Z for s, _ in parts) or parts[0][0] == parts[1][0]:
        raise ValueError("Expected two distinct chemical elements.")
    total = sum(int(n or 1) for _, n in parts)
    if not 2 <= total <= 8:
        raise ValueError("This checkpoint supports cells containing two to eight atoms.")
    return sorted(a for s, n in parts for a in [Z[s]] * int(n or 1))


def cif_text(name, z, k, f):
    gram = unpack_gram(k)
    lengths = np.sqrt(np.diag(gram))
    cos = gram / np.outer(lengths, lengths)
    angles = np.rad2deg(np.arccos(np.clip(cos[[1,0,0], [2,2,1]], -1, 1)))
    lines = [f"data_{name}", "_audit_creation_method 'Direct crystal drifting model'",
             "_symmetry_space_group_name_H-M 'P 1'", "_symmetry_Int_Tables_number 1"]
    for label, v in zip(("length_a","length_b","length_c","angle_alpha","angle_beta","angle_gamma"),
                        [*lengths, *angles]):
        lines.append(f"_cell_{label} {v:.9f}")
    lines.extend(["loop_", "_symmetry_equiv_pos_as_xyz", "'x,y,z'", "loop_", "_atom_site_label",
                  "_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y",
                  "_atom_site_fract_z", "_atom_site_occupancy"])
    # Choose the first atom as origin for presentation. P1 lists every generated atom explicitly.
    relative = (f - f[0]) % 1
    for i, (a, xyz) in enumerate(zip(z, relative), 1):
        symbol = ELEMENTS[a]
        lines.append(f"{symbol}{i} {symbol} " + " ".join(f"{v:.9f}" for v in xyz) + " 1")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("formula", help="Exact cell composition, e.g. NaCl, MgZn2, or Mg2Zn4")
    p.add_argument("--checkpoint", type=Path, default=HERE/"drift-results/multi-exact/seed11.pt")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    if args.count < 1:
        p.error("--count must be positive")
    atoms = atom_list(args.formula)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not state["method"].startswith("multi") and len(atoms) != 2:
        p.error("The selected checkpoint was trained on two-atom cells only.")
    model = CrystalGenerator().eval()
    model.load_state_dict(state["model"])
    rng = torch.Generator().manual_seed(args.seed)
    z = torch.tensor(atoms).expand(args.count, -1)
    with torch.inference_mode():
        k, f = model(z, torch.randn(args.count, 6, generator=rng),
                     torch.rand(args.count, len(atoms), 3, generator=rng))
        k = k * state["scale"] + state["mean"]
    args.out.mkdir(exist_ok=True, parents=True)
    paths = [args.out / f"{args.formula}-{args.seed}-{i+1}.cif" for i in range(args.count)]
    if any(path.exists() for path in paths):
        p.error("An output with that formula, seed and index exists. Choose a different folder or seed.")
    for i, path in enumerate(paths):
        path.write_text(cif_text(path.stem, atoms, k[i].numpy(), f[i].numpy()), encoding="ascii")
        print(path)


if __name__ == "__main__":
    main()
