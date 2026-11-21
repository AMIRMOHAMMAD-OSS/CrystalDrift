from pathlib import Path
import csv
import json
import statistics

import numpy as np


root = Path("geometric-study/crystal-drift")
target = Path(__file__).resolve().parent
cases = []
fixed_prefix = {}
tests = None
for name, key, evaluations in [
    ("CrytalDrift", "exact", 1),
    ("Compact flow · Euler 1", "flow1", 1),
    ("Compact flow · Euler 16", "flow16", 16),
    ("Training-set retrieval", "retrieval", 0),
    ("uFlowCSP · released", "uflow", 1),
]:
    seeds = [2026] if key == "uflow" else [11, 23, 37]
    methods = []
    for seed in seeds:
        doc = json.loads((root / "structure-matching" / f"multi-{key}-{seed}.json").read_text())
        ids = doc["material_ids"]
        assert len(ids) == len(set(ids)) == 128
        assert tests is None or tests == ids
        tests = ids
        rms = np.asarray(doc["rms_by_candidate"], dtype=float)
        assert rms.shape == (128, 5)
        methods.append(rms)
    timer_name = "drift" if key == "exact" else key
    times = {}
    seed_times = {}
    for device in ["cpu", "cuda"]:
        with (root / "speed-results" / f"timings-{device}.csv").open() as f:
            raw = [r for r in csv.DictReader(f) if r["cohort"] == "multi" and r["method"] == timer_name]
        for batch in (1, 256):
            selected = [r for r in raw if int(r["batch"]) == batch]
            times[device, batch] = statistics.median(float(r["wall_ms"]) for r in selected) if selected else None
            if selected:
                assert len(selected) == 45
            if device == "cpu" and batch == 1 and selected:
                seed_times = {str(s): statistics.median(float(r["wall_ms"]) for r in selected if int(r["seed"]) == s) for s in seeds}
    for k in [1, 4, 5]:
        byseed = []
        for seed, array in zip(seeds, methods):
            subset = array[:, :k]
            hit = np.isfinite(subset).any(axis=1)
            best = np.nanmin(subset[hit], axis=1)
            byseed.append([int(sum(hit)), float(np.mean(best))])
            if k in (1, 5):
                doc = json.loads((root / "structure-matching" / f"multi-{key}-{seed}.json").read_text())
                assert byseed[-1][0] == doc["summary"][f"hits_at_{k}"]
                assert np.isclose(byseed[-1][1], doc["summary"][f"mean_best_normalized_rms_among_hits_at_{k}"])
        match = float(np.mean([pair[0] / 128 for pair in byseed]))
        rmse = float(np.mean([pair[1] for pair in byseed]))
        if k in [1,5]:
            cases.append(dict(name=name, k=k, nfe=evaluations, mr=match, rmse=rmse,
                            cpu=times["cpu",1], gpu=times["cuda",1], gpu_batch=times["cuda",256],
                            seeds=seeds, counts=byseed))
        if seed_times:
            fixed_prefix[f"{key}-{k}"] = [match, k * statistics.mean(seed_times.values())]
out = dict(cases=cases, prefix=fixed_prefix)
(target / "metrics.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
for r in cases:
    print(r["name"], r["k"], f'{100*r["mr"]:.2f}', f'{r["rmse"]:.4f}', r["cpu"], r["gpu"])
