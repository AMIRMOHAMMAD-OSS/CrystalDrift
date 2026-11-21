"""Reduce ordered per-crystal measurements to the released de novo metrics and their denominators."""

import argparse, ast, csv, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from prior import split_formula
from score import ROOT, HERE, arrays, formula

ARMS = ("empirical-drift", "learned-drift", "learned-flow16", "learned-uflow", "crystalflow-joint")
FRESH_JOINT = "crystalflow-joint-local"


def load_tiles(name, kind):
    a = arrays(name)
    tiles = sorted((HERE / f"scores/{kind}/{name}").glob("?????-?????.npz"))
    # A resumed run may use a different tile size. Select a complete, non-overlapping path.
    n = len(a["ids"])
    next_tile = {n: None}
    for t in reversed(tiles):
        first, last = map(int, t.stem.split("-"))
        if first < last and last in next_tile and first not in next_tile:
            next_tile[first] = (last, t)
    if 0 not in next_tile:
        return
    content = {}
    first = 0
    while first < n:
        last, t = next_tile[first]
        with np.load(t) as d:
            np.testing.assert_array_equal(d["ids"], a["ids"][first:last])
            for k in d.files:
                if k != "elapsed":
                    content.setdefault(k, []).append(d[k])
        first = last
    return {k: np.concatenate(v) for k, v in content.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    default_arms = (*ARMS, FRESH_JOINT) if (HERE / f"pools/{FRESH_JOINT}.npz").exists() else ARMS
    ap.add_argument("--arms", nargs="+", default=default_arms)
    ns = ap.parse_args()
    dest = HERE / "panel"
    dest.mkdir(exist_ok=True)
    ref = load_tiles("reference", "raw")
    r_fp = load_tiles("reference", "fp")
    train = arrays("training")["z"]
    ftrain = set(map(formula, train))
    ctrain = set(map(tuple, train.tolist()))
    reference_seen = np.array([formula(r) in ftrain for r in arrays("reference")["z"]])
    const = {
        s.targets[0].id: np.array(ast.literal_eval(s.value))
        for s in ast.parse((ROOT / "revision/denovo/diffcsp_constants.txt").read_text()).body
        if isinstance(s, ast.Assign)
    }

    def features(fp):
        good = fp["error"] == ""
        comp = np.nan_to_num(
            (fp["comp"][good] - const["CompScalerMeans"]) / const["CompScalerStds"], nan=0.0
        )
        return comp, fp["struc"][good]

    if r_fp:
        rc, rs = features(r_fp)
        assert len(rc) == len(ref["ids"])
    overview = {}
    for name in ns.arms:
        raw = load_tiles(name, "raw")
        if not raw:
            continue
        data = arrays(name)
        z = data["z"]
        n = len(z)
        present = (z > 0).any(1) & data["valid_output"]
        seen = np.array([formula(row) in ftrain if row.any() else False for row in z])
        exact = np.array([tuple(r) in ctrain for r in z])
        fp = load_tiles(name, "fp")
        valid = raw["struct_valid"] & raw["comp_valid"]
        if fp:
            valid &= fp["error"] == ""
        chosen = np.random.RandomState(7121).choice(np.flatnonzero(valid), size=1000, replace=False)
        finite = present & np.isfinite(data["k"]).all(1) & np.isfinite(data["f"]).all((1, 2))
        result = dict(
            draws=n,
            finite=int(finite.sum()),
            struct_valid=int(raw["struct_valid"].sum()),
            comp_valid=int(raw["comp_valid"].sum()),
            joint_valid=int(valid.sum()),
            formula_unseen_training=int((present & ~seen).sum()),
            cell_composition_unseen_training=int((present & ~exact).sum()),
            distinct_reduced_formulas=len(set(formula(r) for r in z[present])),
            distinct_unseen_reduced_formulas=len({formula(row) for row in z[present & ~seen]}),
            draws_above_property_atomic_number_limit=int((z.max(axis=1) > 94).sum()),
            property_sample_above_atomic_number_limit=int((z[chosen].max(axis=1) > 94).sum()),
            w1_density=float(wasserstein_distance(ref["density"], raw["density"][chosen])),
            w1_nelems=float(wasserstein_distance(ref["nelem"], raw["nelem"][chosen])),
            selected_seed=7121,
            fingerprints_complete=bool(fp),
            reference_fingerprints_complete=bool(r_fp),
        )
        addition = {}
        energy_ref = HERE / "energies/reference.npz"
        energy_gen = HERE / f"energies/{name}.npz"
        if energy_ref.exists() and energy_gen.exists():
            with np.load(energy_ref) as re, np.load(energy_gen) as ge:
                np.testing.assert_array_equal(re["ids"], arrays("reference")["ids"])
                if np.array_equal(ge["index"], np.sort(chosen)):
                    np.testing.assert_array_equal(ge["ids"], data["ids"][ge["index"]])
                    result["w1_formation_energy"] = float(
                        wasserstein_distance(re["energy_eV_per_atom"], ge["energy_eV_per_atom"])
                    )
        novfile = HERE / f"novelty/{name}.csv"
        if novfile.exists():
            with novfile.open() as stream:
                nov = list(csv.DictReader(stream))
            assert len(nov) == n and [v["id"] for v in nov] == data["ids"].tolist()
            reps = np.array(
                [str(i) == v["representative_index"] and not v["error"] for i, v in enumerate(nov)]
            )
            new = np.array([not r["matched_train_id"] and not r["error"] for r in nov])
            addition.update(unique=reps, structural_novel=new)
            timing = HERE / ("pools/" + name + ("-gpu.json" if name in ARMS[:3] else ".json"))
            if timing.exists():
                t = json.loads(timing.read_text())
                replay = HERE / f"pools/{name}-gpu-matched.json"
                if replay.exists():
                    original = t
                    t = json.loads(replay.read_text())
                    assert t["above_tolerance"] == 0
                    result.update(
                        gpu_first_wall_s=original["elapsed_s"],
                        gpu_replay_matches=True,
                        max_chart_difference=t["chart_linf"],
                        max_periodic_site_difference=t["site_periodic_linf"],
                    )
                result.update(
                    gpu_wall_s=t["elapsed_s"],
                    batch_one_median_ms=float(np.median(t["b1_ms"])),
                    valid_unique_per_min=float((valid & reps).sum() * 60 / t["elapsed_s"]),
                )
            result.update(
                unique=int(sum(reps)),
                structural_novel=int(sum(new)),
                valid_unique=int((valid & reps).sum()),
                valid_unique_novel=int((valid & reps & new).sum()),
                novelty_reference_rows=len(train),
            )
        if fp and r_fp:
            gc, gs = features(fp)
            rcs = cKDTree(gc).query(rc, workers=1)[0]
            rss = cKDTree(gs).query(rs, workers=1)[0]
            gcs = cKDTree(rc).query(gc, workers=1)[0]
            gss = cKDTree(rs).query(gs, workers=1)[0]
            # Check individual complete nearest-neighbour searches against the released dense-distance expression.
            rng = np.random.default_rng(7121)
            rcheck = rng.choice(len(rc), 16, replace=False)
            gcheck = rng.choice(len(gc), 16, replace=False)
            for left, right, expected in (
                (rc[rcheck], gc, rcs[rcheck]),
                (rs[rcheck], gs, rss[rcheck]),
                (gc[gcheck], rc, gcs[gcheck]),
                (gs[gcheck], rs, gss[gcheck]),
            ):
                np.testing.assert_allclose(
                    cdist(left, right).min(1), expected, rtol=1e-10, atol=1e-10
                )
            result.update(
                coverage_precision=float(((gcs <= 10) & (gss <= 0.4)).sum() / n),
                coverage_recall=float(((rcs <= 10) & (rss <= 0.4)).mean()),
                reference_within_chemistry_cutoff=int((rcs <= 10).sum()),
                reference_within_structure_cutoff=int((rss <= 0.4).sum()),
                generated_within_chemistry_cutoff=int((gcs <= 10).sum()),
                generated_within_structure_cutoff=int((gss <= 0.4).sum()),
                generated_fp_errors=int(sum(fp["error"] != "")),
            )
            np.savez_compressed(
                dest / f"{name}-coverage.npz",
                ref_chem=rcs,
                ref_struc=rss,
                generated_chem=gcs,
                generated_struc=gss,
                generated_rows=np.flatnonzero(fp["error"] == ""),
            )
        strata = {}
        for label, selected in (("seen", present & seen), ("unseen", present & ~seen)):
            if selected.any():
                rows = selected
                strata[label] = dict(
                    proposals=int(rows.sum()),
                    struct_valid=int(raw["struct_valid"][rows].sum()),
                    comp_valid=int(raw["comp_valid"][rows].sum()),
                    joint_valid=int(valid[rows].sum()),
                    median_shortest_A=float(np.nanmedian(raw["shortest"][rows])),
                    median_volume_per_atom=float(
                        np.nanmedian(raw["volume"][rows] / (z[rows] > 0).sum(1))
                    ),
                )
                rv = reference_seen if label == "seen" else ~reference_seen
                defined = rows & np.isfinite(raw["density"])
                strata[label].update(
                    reference_count=int(rv.sum()),
                    finite_density_count=int(defined.sum()),
                    raw_density_w1=(
                        float(wasserstein_distance(raw["density"][defined], ref["density"][rv]))
                        if defined.any()
                        else None
                    ),
                )
        result["by_reduced_formula"] = strata
        np.savez_compressed(
            dest / f"{name}-cases.npz",
            **raw,
            seen_reduced_formula=seen,
            seen_cell_formula=exact,
            w1_sample=chosen,
            valid=valid,
            **addition,
        )
        overview[name] = result
        (dest / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    if set(("learned-drift", "learned-flow16", "learned-uflow")) <= set(overview):
        base = "learned-drift"
        paired = {}
        rng = np.random.default_rng(7121)
        with np.load(dest / f"{base}-cases.npz") as a:
            for name in ("learned-flow16", "learned-uflow"):
                np.testing.assert_array_equal(arrays(base)["z"], arrays(name)["z"])
                with np.load(dest / f"{name}-cases.npz") as b:
                    out = {}
                    for group, mask in (
                        ("all", np.ones(len(a["ids"]), bool)),
                        ("seen", a["seen_reduced_formula"]),
                        ("unseen", ~a["seen_reduced_formula"]),
                    ):
                        scores = {}
                        for field in ("struct_valid", "valid"):
                            delta = a[field][mask].astype(int) - b[field][mask].astype(int)
                            counts = np.bincount(delta + 1, minlength=3)
                            boot = rng.multinomial(len(delta), counts / len(delta), size=10000)
                            interval = np.quantile(
                                100 * (boot[:, 2] - boot[:, 0]) / len(delta), [0.025, 0.975]
                            )
                            scores[field] = dict(
                                n=len(delta),
                                first_only=int(counts[2]),
                                second_only=int(counts[0]),
                                difference_pp=100 * delta.mean(),
                                paired_bootstrap95_pp=interval.tolist(),
                            )
                        out[group] = scores
                    paired[name] = out
        (dest / "paired-validity.json").write_text(json.dumps(paired, indent=2) + "\n")
    print(
        json.dumps(
            {
                a: {
                    k: v
                    for k, v in o.items()
                    if k
                    in (
                        "joint_valid",
                        "w1_density",
                        "formula_unseen_training",
                        "coverage_recall",
                        "valid_unique_novel",
                    )
                }
                for a, o in overview.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
