"""Measure when v2's orbit-variance ablation can carry geometric information."""
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
WYCKOFF = HERE.parent / "wyckoff"
sys.path.insert(0, str(WYCKOFF))
from charts import chart
from labels import read
from orbit_power import orbit_ids, variance
from prior import Prior, records
from train_template import batch, real, draw
from lattice_symmetry.v2.physical import features
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from standard_match import SETTINGS


def eligible(record):
    """A variance across orbits of the same species and multiplicity needs at least two."""
    classes = [(z, len(chart(record["group"], letter)[0])) for z, letter in record["template"]]
    return len(classes) > len(set(classes))


def matched_rows(arm):
    targets = defaultdict(list)
    with gzip.open(WYCKOFF / "scores" / f"{arm}.jsonl.gz", "rt", encoding="utf8") as f:
        for line in f:
            row = json.loads(line)
            targets[row["id"]].append((row["draw"], row["rms"] is not None))
    return {key: [hit for _, hit in sorted(draws)] for key, draws in targets.items()}


def run():
    torch.set_num_threads(1)
    split_counts = {}
    for split in ("train", "val", "test"):
        counts = Counter()
        for record in read(split):
            if "error" in record:
                counts["rejected"] += 1
                continue
            counts["accepted"] += 1
            counts["repeated_element_and_multiplicity"] += int(eligible(record))
        split_counts[split] = dict(counts)

    test_cohort = json.loads((WYCKOFF / "test-cohort.json").read_text())
    plain, orbit = (matched_rows(arm) for arm in ("angular-C", "orbit-C"))
    panel = {}
    for flag in (True, False):
        ids = [r["id"] for r in test_cohort if "error" not in r and eligible(r) == flag]
        cells = {}
        for budget in (1, 5, 20):
            a = np.asarray([any(plain[key][:budget]) for key in ids])
            b = np.asarray([any(orbit[key][:budget]) for key in ids])
            cells[budget] = dict(ordinary_hits=int(a.sum()), orbit_hits=int(b.sum()),
                                 orbit_only=int((b & ~a).sum()), ordinary_only=int((a & ~b).sum()))
        panel[str(flag)] = dict(targets=len(ids), recovery=cells)

    validation = [r for r in json.loads((WYCKOFF / "val-cohort.json").read_text()) if "error" not in r]
    raw = []
    for start in range(0, len(validation), 4):
        selected = validation[start:start + 4]
        layout = [orbit_ids(r["group"], r["template"]) for r in selected]
        lattice, frac, species = batch([real(r) for r in selected])
        lattice, frac = lattice.double(), frac.double()
        with torch.no_grad():
            ref = variance(lattice, frac, species, layout)
            expanded = variance(lattice * 1.05, frac, species, layout)
        for row, val, diff in zip(selected, ref, (expanded - ref).abs()):
            raw.append((row["id"], eligible(row), float(val.abs().max()), float(diff.max())))
    data = {}
    for flag in (True, False):
        rows = [r for r in raw if r[1] == flag]
        data[str(flag)] = dict(targets=len(rows), nonzero_at_1e_10=sum(r[2]>1e-10 for r in rows),
                               respond_to_5pct_length_increase_at_1e_10=sum(r[3]>1e-10 for r in rows))

    fitted_prior = Prior(records())
    rng = np.random.default_rng(87201)
    with np.load(WYCKOFF.parent / "csp" / "scaling.npz") as archive:
        base_scale = torch.from_numpy(archive["scale"]).double()
    with np.load(WYCKOFF / "training-orbit" / "orbit-scaling.npz") as archive:
        orbit_scale = float(archive["scale"])
    active_cases = []
    channel_weights = (0.0, 0.1, 1.0)
    nearest_hits = Counter()
    at_least_one = 0
    matcher = StructureMatcher(**SETTINGS)
    for record in validation:
        if not eligible(record):
            continue
        # Compare a reference descriptor with alternative draws from its supplied template.
        # Ranking by the target geometry is an influence diagnostic, unavailable at deployment.
        bank = [real(record)] + [draw(record, fitted_prior, rng) for _ in range(4)]
        lattice, frac, species = batch(bank)
        lattice, frac = lattice.double(), frac.double()
        layout = [orbit_ids(record["group"], record["template"])] * len(bank)
        with torch.no_grad():
            base = features(lattice, frac, species)
            var = variance(lattice, frac, species, layout)
            base_d2 = ((base[1:] - base[:1]) / base_scale).square().sum(dim=1).numpy()
            orbit_d2 = ((var[1:] - var[:1]) / orbit_scale).square().sum(dim=1).numpy()
        total = base_d2 + orbit_d2
        fractions = np.divide(orbit_d2, total, out=np.zeros_like(total), where=total>0)
        active_cases.append((np.argmin(base_d2) != np.argmin(total), fractions))
        ref_cell, ref_frac, ref_z = bank[0]
        reference_crystal = Structure(ref_cell, ref_z, ref_frac)
        matches = [matcher.get_rms_dist(reference_crystal, Structure(cell, z, frac)) is not None
                   for cell, frac, z in bank[1:]]
        at_least_one += any(matches)
        for weight in channel_weights:
            chosen = np.argmin(base_d2 + weight ** 2 * orbit_d2)
            nearest_hits[weight] += matches[chosen]
    shares = np.concatenate([values for _, values in active_cases])
    influence = dict(eligible_targets=len(active_cases), candidates_per_target=4, seed=87201,
                     changed_closest_descriptor=int(sum(flip for flip, _ in active_cases)),
                     targets_with_any_match=at_least_one,
                     nearest_to_known_reference_hits_by_orbit_channel_multiplier=dict(nearest_hits),
                     share_of_squared_distance_quantiles=dict(zip(
                         ["q25", "q50", "q75", "q95", "max"], np.quantile(shares, [.25,.5,.75,.95,1]).tolist())))
    return {"definition":"An accepted template can have nonzero orbit-variance channels only "
            "when it contains at least two occupied orbits of the same species and multiplicity.",
            "official_splits":split_counts, "stored_128_target_test":panel,
            "fresh_FP64_reference_and_uniform_dilation_on_128_validation_targets":data,
            "fresh_validation_influence_on_distance_to_reference_conditional_template_draws":influence}


if __name__ == "__main__":
    output = run()
    (HERE / "orbit_coverage.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf8")
    print(json.dumps(output, indent=2))
