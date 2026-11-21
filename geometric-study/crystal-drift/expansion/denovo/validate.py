"""Independently reconcile reported counts, benchmark inputs, energy scores and rates."""

import bz2
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from report import ARMS, FRESH_JOINT
from score import HERE, ROOT, arrays


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def main():
    result = {"arms": {}}
    ref = arrays("reference")
    train = arrays("training")
    assert len(train["ids"]) == 27136 and len(ref["ids"]) == 9046
    assert not set(train["ids"]) & set(ref["ids"])
    with bz2.open(ROOT / "data/mp20_test.csv.bz2", "rt", encoding="utf8") as f:
        truth = {r["material_id"]: float(r["formation_energy_per_atom"]) for r in csv.DictReader(f)}
    with np.load(HERE / "energies/reference.npz") as e:
        np.testing.assert_array_equal(e["ids"], ref["ids"])
        ref_energy = e["energy_eV_per_atom"]
        residual = ref_energy - np.array([truth[i] for i in ref["ids"]])
        result["reference_energy_check"] = {
            "n": len(residual),
            "mae_eV_per_atom": float(abs(residual).mean()),
            "median_absolute_error_eV_per_atom": float(np.median(abs(residual))),
            "mean_signed_error_eV_per_atom": float(residual.mean()),
        }
    arms = (*ARMS, FRESH_JOINT) if (HERE / f"panel/{FRESH_JOINT}.json").exists() else ARMS
    for name in arms:
        bank = arrays(name)
        panel = json.loads((HERE / f"panel/{name}.json").read_text())
        assert panel["reference_fingerprints_complete"] and panel["fingerprints_complete"]
        with np.load(HERE / f"panel/{name}-cases.npz") as d:
            assert len(d["ids"]) == 10000 and len(set(d["ids"])) == 10000
            np.testing.assert_array_equal(d["ids"], bank["ids"])
            for field in ("comp_valid", "struct_valid", "unique", "structural_novel"):
                assert d[field].sum() == panel[field]
            assert d["valid"].sum() == panel["joint_valid"]
            assert (d["valid"] & d["unique"] & d["structural_novel"]).sum() == panel[
                "valid_unique_novel"
            ]
            chosen = d["w1_sample"]
            assert len(set(chosen)) == 1000 and d["valid"][chosen].all()
            np.testing.assert_array_equal(
                chosen,
                np.random.RandomState(7121).choice(np.flatnonzero(d["valid"]), 1000, replace=False),
            )
            with np.load(HERE / f"energies/{name}.npz") as ef:
                np.testing.assert_array_equal(ef["index"], np.sort(chosen))
                np.testing.assert_array_equal(ef["ids"], d["ids"][ef["index"]])
                np.testing.assert_allclose(
                    wasserstein_distance(ref_energy, ef["energy_eV_per_atom"]),
                    panel["w1_formation_energy"],
                    rtol=1e-12,
                )
            if "gpu_wall_s" in panel:
                np.testing.assert_allclose(
                    panel["valid_unique_per_min"],
                    60 * (d["valid"] & d["unique"]).sum() / panel["gpu_wall_s"],
                )
                if panel.get("gpu_replay_matches"):
                    with np.load(HERE / f"pools/{name}-gpu-parity.npz") as parity:
                        for field, target in (
                            ("lattice_chart_max_difference", "max_chart_difference"),
                            ("fractional_max_difference", "max_periodic_site_difference"),
                        ):
                            assert len(parity[field]) == 10000
                            assert parity[field].max() == panel[target] and panel[target] <= 1e-4
        with np.load(HERE / f"panel/{name}-coverage.npz") as c:
            assert len(c["ref_chem"]) == len(c["ref_struc"]) == 9046
            np.testing.assert_allclose(
                ((c["ref_chem"] <= 10) & (c["ref_struc"] <= 0.4)).mean(), panel["coverage_recall"]
            )
            np.testing.assert_allclose(
                ((c["generated_chem"] <= 10) & (c["generated_struc"] <= 0.4)).sum() / 10000,
                panel["coverage_precision"],
            )
        path = (
            HERE / f"pools/{name}.npz"
            if name != "crystalflow-joint"
            else ROOT / "expansion/external/crystalflow-joint.npz"
        )
        result["arms"][name] = {
            "cases": len(bank["ids"]),
            "bank_sha256": digest(path),
            "case_file_sha256": digest(HERE / f"panel/{name}-cases.npz"),
            "panel_sha256": digest(HERE / f"panel/{name}.json"),
        }
    for name in ("learned-flow16", "learned-uflow"):
        np.testing.assert_array_equal(arrays(name)["z"], arrays("learned-drift")["z"])
    (HERE / "panel/audit.json").write_text(json.dumps(result, indent=2) + "\n")
    fp = HERE / "manifest.json"
    include = [
        p for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p != fp
    ]
    include.extend(
        [
            ROOT / "paper/exp-denovo.tex",
            ROOT / "paper/exp-denovo-strata.tex",
            ROOT / "expansion/external/crystalflow-joint.npz",
            *(ROOT / f"scaled/mp20-data/{split}.npz" for split in ("train", "val", "test")),
            *(ROOT / f"scaled/runs/mp20/{fit}-seed11.pt" for fit in ("drift", "flow")),
        ]
    )
    metadata = {
        "candidate_arms": list(arms),
        "draws_per_arm": 10000,
        "reference_count": 9046,
        "training_count": 27136,
        "composition_seed": 22112026,
        "compact_geometry_seed": 71831,
        "uflow_geometry_seed": 241126,
        "property_subset_seed": 7121,
        "sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(include)},
    }
    fp.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(result["reference_energy_check"]))


if __name__ == "__main__":
    main()
