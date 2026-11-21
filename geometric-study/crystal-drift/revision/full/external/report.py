"""Audit and summarize released-model draws on complete common target cohorts."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "mp20"
CORE = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location("compact_report", HERE.parent / "report.py")
COMPACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPACT)
METRICS = ("rms_recovery", "strict_recovery", "rms_and_volume_1.10",
           "mean_best_rms_among_recovered")


def collect(ids, k):
    n = len(ids)
    rms = np.full((n, k), np.nan)
    strict = np.zeros((n, k), bool)
    volumes = np.full((n, k), np.nan)
    present = np.zeros((n, k), bool)
    checks = 0
    exceptions = Counter()
    for path in sorted((OUT / "matches").glob("cases*-draws*.npz")):
        if path.name.endswith(".part.npz"):
            continue
        interval, draw = path.stem.split("-draws")
        lo, hi = map(int, interval[5:].split("-"))
        first, last = map(int, draw.split("-"))
        if lo >= n or first >= k:
            continue
        if hi > n or last > k:
            raise ValueError(f"Requested cohort is not aligned with stored tiles: {path}")
        with np.load(path) as d:
            np.testing.assert_array_equal(d["ids"], ids[lo:hi])
            assert (first, last) == (int(d["first"]), int(d["last"]))
            sl = np.s_[lo:hi, first:last]
            assert not present[sl].any()
            present[sl] = True
            rms[sl], strict[sl], volumes[sl] = d["rms"], d["strict_fit"], d["log_volume_ratio"]
            checks += int(d["public_api_checks"])
            exceptions.update(json.loads(str(d["exceptions"])))
    if not present.all():
        return None
    assert np.isfinite(volumes).all()
    return rms, strict, volumes, {"public_api_checks": checks, "exceptions": dict(exceptions)}


def archive_check(ids, k, rms):
    prior = CORE / "scaled/external/mp20"
    saved = json.loads((prior / "uflow-seed2026-test-matching.json").read_text())
    count = min(k, 5)
    np.testing.assert_array_equal(ids[:128], saved["material_ids"])
    old_rms = np.asarray(saved["rms_by_candidate"], float)[:, :count]
    np.testing.assert_allclose(rms[:128, :count], old_rms, atol=1e-8, rtol=1e-8, equal_nan=True)
    finite = np.isfinite(old_rms)
    difference = abs(rms[:128, :count][finite] - old_rms[finite])
    with np.load(prior / "uflow-seed2026-test.npz") as old:
        for i in range(count):
            with np.load(OUT / f"draw{i:02}-cases00000-00128.npz") as d:
                for name in ("k", "f"):
                    assert d[name].dtype == old[name].dtype
                    np.testing.assert_array_equal(d[name], old[name][:, i])
    return dict(archived_candidates=128 * count, identical_geometry_arrays=True,
                finite_rms_hits=int(finite.sum()), max_abs_rms_difference=float(difference.max()))


def summarize(ext, other, iterations):
    rms, strict, volume = [v[:, None, :] for v in ext]
    arms = ["uFlowCSP"]
    groups = [[0]]
    if other is not None:
        rms = np.concatenate((rms, other[0]), axis=1)
        strict = np.concatenate((strict, other[1]), axis=1)
        volume = np.concatenate((volume, other[2]), axis=1)
        arms += COMPACT.METHODS
        groups += [list(range(1 + 3*i, 4 + 3*i)) for i in range(len(COMPACT.METHODS))]
    n, n_arms, _ = rms.shape
    case = COMPACT.by_case(rms, strict, volume)
    # Retain fitted arms until after conditioning the mean RMS on their own hits.
    input_rows = np.stack(tuple(case.values()), axis=2).reshape(n, -1).astype(float)
    def estimates(totals):
        totals = totals.reshape(-1, n_arms, 4)
        rms_sum, hit_count = totals[:, :, 3].copy(), totals[:, :, 0].copy()
        totals /= n
        totals[:, :, 3] = np.divide(rms_sum, hit_count, out=np.full_like(rms_sum, np.nan),
                                    where=hit_count > 0)
        return np.stack([totals[:, ix].mean(1) for ix in groups], axis=1)
    point = estimates(input_rows.sum(0))[0]
    sample = np.empty((iterations, len(arms), 4))
    rng = np.random.default_rng(741123)
    for start in range(0, iterations, 60):
        size = min(60, iterations - start)
        weights = rng.multinomial(n, np.full(n, 1/n), size=size)
        sample[start:start+size] = estimates(weights @ input_rows)
    def intervals(p, s):
        return {key: COMPACT.interval(p[j], s[:, j]) for j, key in enumerate(METRICS)}
    result = {"uFlowCSP": intervals(point[0], sample[:, 0])}
    if other is not None:
        result["compact_minus_uFlowCSP"] = {
            name: intervals(point[i] - point[0], sample[:, i] - sample[:, 0])
            for i, name in enumerate(arms) if i}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--k", type=int, choices=(1,5,20), nargs="+", default=(1,5,20))
    args = parser.parse_args()
    with np.load(HERE.parent / "mp20/cohort.npz") as ref:
        all_ids = ref["ids"]
    for tag, ids in (("prefix", all_ids[:128]), ("full", all_ids)):
        for k in args.k:
            result = OUT / f"{tag}-k{k}.json"
            prior = json.loads(result.read_text()) if result.exists() else {}
            if "compact_minus_uFlowCSP" in prior and prior.get("bootstrap") == args.bootstrap:
                continue
            target = collect(ids, k)
            if target is None:
                continue
            other = COMPACT.read_mp(ids, k)
            if prior and other is None:
                continue
            rms, strict, volume, audit = target
            assert not audit["exceptions"], audit
            stats = summarize(target[:3], other, args.bootstrap)
            stats.update(targets=len(ids), k=k, candidates=len(ids)*k, bootstrap=args.bootstrap,
                         bootstrap_seed=741123, target_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                         prefix_parity=archive_check(ids, k, rms), audit=audit)
            good = np.isfinite(rms)
            native = good & (abs(volume) <= np.log(1.10))
            hits = [good.any(1), strict.any(1), native.any(1)]
            stats["counts"] = dict(zip(METRICS[:3], [int(v.sum()) for v in hits]))
            best = np.where(good, rms, np.inf).min(1)
            with (OUT / f"{tag}-k{k}-cases.csv").open("w", newline="") as file:
                w = csv.writer(file)
                w.writerow(["material_id", "rms_match", "max_site_fit", "rms_and_native_volume",
                            "best_normalized_rms_among_matches"])
                for row, mat_id in enumerate(ids):
                    w.writerow([mat_id, *(int(col[row]) for col in hits),
                                best[row] if np.isfinite(best[row]) else ""])
            COMPACT.write(result, stats)
            print(result.name, stats["counts"], "paired comparison", other is not None, flush=True)


if __name__ == "__main__":
    main()
