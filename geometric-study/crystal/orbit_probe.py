"""Audit the WyckoffDiff MP-20 training release and a two-state count relaxation."""
import argparse
import bz2
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import torch


SOURCE = "https://public.openmaterialsdb.se/wyckoffdiff/data/mp20/raw/mp20_train.csv.bz2"


def read_training(path):
    compressed = path.read_bytes()
    decompressor = bz2.BZ2Decompressor()
    decoded = decompressor.decompress(compressed)
    if not decompressor.eof or decompressor.unused_data:
        raise ValueError("Expected one complete bzip2 member; the file may still be downloading.")
    rows = csv.DictReader(io.StringIO(decoded.decode("utf-8"), newline=""))
    for ordinal, row in enumerate(rows, 1):
        formula_line = next(line for line in row["cif.conv"].splitlines() if line.startswith("_chemical_formula_sum"))
        formula = formula_line.removeprefix("_chemical_formula_sum").strip().strip("'").replace(" ", "")
        components = re.findall(r"([A-Z][a-z]?)([0-9.]*)", formula)
        if "".join(e + n for e, n in components) != formula:
            raise ValueError(f"Unparsed conventional formula in row {ordinal}: {formula}")
        atom_counts = {e: int(float(n or 1)) for e, n in components}
        assert all(atom_counts[e] == float(n or 1) for e, n in components)
        descriptor, element_order = row["wyckoff_spglib"].split(":")
        _, pearson, sg, *token_strings = descriptor.split("_")
        elements = element_order.split("-")
        assert int(sg) == int(row["spacegroup.number.conv"])
        assert set(elements) == set(atom_counts) and len(elements) == len(token_strings)
        assert sum(atom_counts.values()) == int(pearson[2:])
        counts = {}
        for element, tokens in zip(elements, token_strings):
            pairs = re.findall(r"(\d*)([A-Za-z])", tokens)
            assert "".join(count + letter for count, letter in pairs) == tokens
            counts[element] = sum(int(count or 1) for count, _ in pairs)
        yield {
            "train_record": ordinal,
            "material_id": row["material_id"],
            "conventional_composition": atom_counts,
            "space_group": int(sg),
            "wyckoff_spglib": row["wyckoff_spglib"],
            "orbit_counts": counts,
        }


def collision_summary(rows):
    groups = defaultdict(Counter)
    for row in rows:
        key = row["space_group"], tuple(sorted(row["conventional_composition"].items()))
        profile = tuple(sorted(row["orbit_counts"].items()))
        groups[key][profile] += 1
    different_profiles = [group for group in groups.values() if len(group) > 1]
    return {
        "train_records": len(rows),
        "composition_and_space_group_keys": len(groups),
        "keys_with_multiple_records": sum(group.total() > 1 for group in groups.values()),
        "keys_with_different_orbit_count_profiles": len(different_profiles),
        "record_pairs_with_different_profiles": sum(a * b for g in different_profiles for a, b in combinations(g.values(), 2)),
    }


def gradient_probe(witness):
    # States list counts of the two Zn orbits, while Mg stays on one 4c.
    # Each row obeys composition exactly; the c=1 state cannot occur.
    states = torch.tensor([(c, g) for c in range(3) for g in range(2) if 4*c + 8*g == 8])
    assert states.tolist() == [[0, 1], [2, 0]]
    dtype = torch.float64
    zero = torch.zeros(5, dtype=dtype, requires_grad=True)
    materials = ["mp-1094198", "mp-1094439"]  # (0,1) then (2,0)
    observed_features = torch.tensor([[witness[k]["orbit_counts"]["Mg"], witness[k]["orbit_counts"]["Zn"]]
                                      for k in materials], dtype=dtype)
    def composite_scores(z):
        return z[states[:, 0]] + z[3 + states[:, 1]]

    def feature(state_probabilities):
        expected_counts = state_probabilities @ states.to(dtype)
        return torch.stack((state_probabilities.sum(), expected_counts.sum()))

    for hard_state, expected in zip(torch.eye(2, dtype=dtype), observed_features):
        torch.testing.assert_close(feature(hard_state), expected, rtol=0, atol=0)
    def loss(z):
        probabilities = composite_scores(z).softmax(0)
        return 0.5 * (feature(probabilities) - observed_features[1]).square().sum()

    p = composite_scores(zero).softmax(0)
    f = feature(p)
    objective = loss(zero)
    grad, = torch.autograd.grad(objective, zero)
    eps = 1e-5
    finite_diff = torch.stack([(loss(zero + eps*d) - loss(zero - eps*d))/(2*eps) for d in torch.eye(5, dtype=dtype)])
    torch.testing.assert_close(grad, finite_diff, rtol=1e-8, atol=1e-10)
    pooled_atoms = torch.tensor([[4, 4*c+8*g] for c, g in states.tolist()], dtype=dtype)
    assert torch.equal(pooled_atoms[0], pooled_atoms[1])
    def composition_feature(z):
        return composite_scores(z).softmax(0) @ pooled_atoms

    composition_jacobian = torch.autograd.functional.jacobian(composition_feature, zero)
    assert not torch.count_nonzero(composition_jacobian)
    return {
        "state_material_ids": materials,
        "allowed_zinc_counts_4c_8g": states.tolist(),
        "logit_order": ["4c:0", "4c:1", "4c:2", "8g:0", "8g:1"],
        "initial_state_probabilities": p.tolist(),
        "hard_species_orbit_features_Mg_Zn": observed_features.tolist(),
        "max_error_one_hot_vs_observed_hard_features": 0.0,
        "hard_species_atom_counts_Mg_Zn": pooled_atoms.tolist(),
        "soft_species_orbit_feature_Mg_Zn": f.tolist(),
        "half_squared_orbit_feature_distance_to_second_state": objective.item(),
        "gradient_wrt_count_logits": grad.tolist(),
        "autograd_vs_central_difference_max_abs_error": (grad - finite_diff).abs().max().item(),
        "composition_only_gradient": composition_jacobian.sum(0).tolist(),
        "composition_feature_jacobian": composition_jacobian.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_bz2", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("orbit_probe.json"))
    args = parser.parse_args()
    rows = list(read_training(args.train_bz2))
    witness = {r["material_id"]: r for r in rows if r["material_id"] in {"mp-1094198", "mp-1094439"}}
    assert len(witness) == 2
    assert all(r["space_group"] == 63 and r["conventional_composition"] == {"Mg": 4, "Zn": 8} for r in witness.values())
    assert witness["mp-1094198"]["wyckoff_spglib"] == "AB2_oC12_63_c_g:Mg-Zn"
    assert witness["mp-1094439"]["wyckoff_spglib"] == "AB2_oC12_63_c_2c:Mg-Zn"
    result = {
        "source": SOURCE, "compressed_sha256": hashlib.sha256(args.train_bz2.read_bytes()).hexdigest(),
        "bzip2_eof": True, "release_split": "train", "torch_version": torch.__version__,
        "initial_2000": collision_summary(rows[:2000]), "entire_training_split": collision_summary(rows),
        "witness": list(witness.values()), "categorical_gradient_probe": gradient_probe(witness),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["initial_2000", "entire_training_split", "categorical_gradient_probe"]}, indent=2))


if __name__ == "__main__":
    main()
