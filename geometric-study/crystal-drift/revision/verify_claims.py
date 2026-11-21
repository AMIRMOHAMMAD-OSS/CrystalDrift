"""Rebuild manuscript endpoints from ordered per-case arrays and check the released tables."""
import csv
import json
from pathlib import Path

import numpy as np

REV = Path(__file__).resolve().parent
ROOT = REV.parent
SEEDS = (11, 23, 37)
METHODS = ("exact", "flow1", "flow16", "retrieval", "uflow")
FULL_METHODS = ("drift", "flow1", "flow16", "retrieval")


def records(path):
    with path.open(encoding="utf8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def endpoint(rms):
    finite = np.isfinite(rms)
    recovered = finite.any(axis=1)
    best = np.where(finite, rms, np.inf).min(axis=1)
    return [float(recovered.mean()), float(best[recovered].mean())]


def binary():
    base = REV / "binary"
    fixed = records(base / "headline-prefix.csv")
    clock = records(base / "clock-budgets.csv")
    runtime = records(base / "runtime.csv")
    result = {}
    for method in METHODS:
        seeds = (2026,) if method == "uflow" else SEEDS
        fixed_by_seed, deadline_by_seed, intervals = [], [], []
        for seed in seeds:
            with np.load(base / f"rms-headline-{method}-{seed}.npz") as scores:
                fixed_by_seed.append([endpoint(scores["rms"][:, :k]) for k in (1, 5, 20)])
                ids = scores["ids"]
            with np.load(base / f"episode-{method}-{seed}.npz") as timed, np.load(
                    base / f"rms-episode-{method}-{seed}.npz") as scored:
                np.testing.assert_array_equal(timed["ids"], ids)
                np.testing.assert_array_equal(scored["ids"], ids)
                complete = timed["elapsed_ms"] <= 5
                hit = (np.isfinite(scored["rms"]) & complete).any(axis=1)
                deadline_by_seed.append([float(hit.mean()), float(complete.sum(axis=1).mean())])
                intervals.extend(np.diff(timed["elapsed_ms"], axis=1, prepend=0).ravel())
        fixed_means = np.mean(fixed_by_seed, axis=0)
        for k, values in zip((1, 5, 20), fixed_means):
            row = next(r for r in fixed if r["method"] == method and int(r["draws"]) == k)
            np.testing.assert_allclose(values, [float(row["mr"]), float(row["rmse"])])
        observed = np.mean(deadline_by_seed, axis=0)
        row = next(r for r in clock if r["method"] == method and float(r["budget_ms"]) == 5)
        np.testing.assert_allclose(observed, [float(row["mr"]), float(row["mean_completed"])])
        timing = next(r for r in runtime if r["method"] == method)
        with np.load(base / f"ten-thousand-{method}-{seeds[0]}.npz") as job:
            np.testing.assert_array_equal(job["ids"], ids[np.arange(10000) % 128])
            assert job["counts"].sum() == 10000
            seconds = float(job["elapsed_s"])
            np.testing.assert_allclose([np.median(intervals), seconds, 10000/seconds],
                [float(timing[k]) for k in ("median_b1_ms","seconds_10000","structures_per_second")])
        result[method] = dict(mr_and_rms_at_1_5_20=fixed_means.tolist(),
            mr_and_completed_at_5ms=observed.tolist(), median_b1_ms=float(np.median(intervals)),
            seconds_10000=seconds)
    return result


def full():
    base = REV / "full"
    data = {}
    for study, budgets in (("mp20",(1,5,20)), ("wbm",(1,5))):
        file = "complete-crosswalk-k20.npz" if study == "mp20" else "test-crosswalk.npz"
        size = 9046 if study == "mp20" else 128
        with np.load(base/study/file) as pool:
            ids = pool["ids"]
            assert len(ids) == len(set(ids)) == size
            arms = pool["arms"].tolist()
            rms, strict, volume = (pool[k] for k in ("rms","strict_fit","log_volume_ratio"))
        external = None
        if study == "mp20":
            with np.load(base/"external/mp20/complete-crosswalk-k20.npz") as other:
                np.testing.assert_array_equal(other["ids"], ids)
                external = tuple(other[k] for k in ("rms","strict_fit","log_volume_ratio"))
        for k in budgets:
            outputs = {}
            published = (load_json(base/study/f"full-k{k}.json") if study == "mp20"
                         else load_json(base/study/"revision-summary.json")[str(k)])
            for method in (*FULL_METHODS, "uflow") if external else FULL_METHODS:
                if method == "uflow":
                    group = [external]
                    summary = load_json(base/f"external/mp20/full-k{k}.json")["uFlowCSP"]
                else:
                    group = [(rms[:,j], strict[:,j], volume[:,j]) for s in SEEDS
                             for j in [arms.index(f"{method}-seed{s}")]]
                    summary = published["methods"][method]
                values = []
                for r, fit, logv in group:
                    r = r[:,:k]
                    volume_hit = np.isfinite(r) & (np.abs(logv[:,:k]) <= np.log(1.10))
                    values.append([*endpoint(r), fit[:,:k].any(1).mean(), volume_hit.any(1).mean()])
                keys = ("rms_recovery","mean_best_rms_among_recovered","strict_recovery","rms_and_volume_1.10")
                average = np.mean(values, axis=0)
                np.testing.assert_allclose(average, [summary[key]["mean"] for key in keys])
                outputs[method] = dict(zip(keys, average.tolist()))
            data[f"{study}@{k}"] = outputs
    return data


def screens():
    data = {}
    for split, size in (("val",1231),("test",1229)):
        base = REV/"ablation"
        summary = load_json(base/f"{split}-summary.json")["paired_by_case"]
        for arm in ("pair","original","harmonic","reweight"):
            seed_rows = [records(base/f"runs/{arm}/seed{s}-{split}-cases.csv") for s in SEEDS]
            assert all(len(rows)==size for rows in seed_rows)
            assert all([r["id"] for r in rows]==[r["id"] for r in seed_rows[0]] for rows in seed_rows)
            estimates = {}
            for key in ("forecast_energy","match5"):
                by_seed = [[float(row[key]) for row in rows if row[key]] for rows in seed_rows]
                assert all(len(a)==(128 if key=="match5" else size) for a in by_seed)
                estimates[key] = float(np.mean(by_seed))
                np.testing.assert_allclose(estimates[key], summary[key]["estimates"][arm]["mean"])
            data[f"{split}/{arm}"] = estimates
    c = records(REV/"denovo/per-cell.csv")
    g = [r for r in c if r["cohort"]=="generated"]
    assert len(g)==1024 and len(c)==2253
    counts = [sum(r[k]=="True" for r in g) for k in ("struct_valid","comp_valid")]
    assert counts == [1003,930]
    assert sum(r["struct_valid"]==r["comp_valid"]=="True" for r in g)==910
    gen = [r for r in records(ROOT/"de-novo/relaxation/results.csv") if r["role"]=="generated"]
    assert len(gen)==32 and sum(r["converged"]=="True" for r in gen)==23
    assert sum(float(r["final_min_covalent_radius_ratio"])<0.60 for r in gen)==7
    assert sum(float(r["final_max_force_eV_per_A"])<=0.1 for r in gen)==17
    data["relaxation"] = dict(cases=len(gen),converged=23,short_contacts=7,atomic_force_pass=17,
                            median_wall_seconds=float(np.median([float(r["wall_seconds"]) for r in gen])))
    witness = load_json(REV/"ablation/checks.json")["witness"]
    assert witness["original"]["max_absolute_difference"] < 1e-14
    assert witness["scaled"]["dimension"] == 1306 and witness["scaled"]["max_absolute_difference"] < 1e-14
    np.testing.assert_allclose(witness["harmonic"]["max_absolute_difference"], 1/(16*np.sqrt(2)))
    return data


if __name__ == "__main__":
    values = dict(binary=binary(), public_and_wbm=full(), ablation_and_physical=screens())
    path = REV/"verified-claims.json"
    path.write_text(json.dumps(values, indent=2, allow_nan=False)+"\n", encoding="utf8")
    print("Verified all fixed-draw, deadline, throughput, multielement, ablation and relaxation headlines.")
