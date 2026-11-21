"""Screen the fixed 1,024 unrelaxed binary proposals for structural duplication.

Each proposed structure is compared only with structures of matching reduced
formula. Novelty is relative to a named database and a specified matcher.
"""
import csv
from collections import Counter, defaultdict
from functools import lru_cache
from importlib.metadata import version
import json
from math import gcd
from pathlib import Path
import sys

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from standard_match import SETTINGS, make_structure


def formula(row):
    counts = Counter(int(v) for v in row if v)
    divisor = gcd(*counts.values())
    return tuple(sorted((v, n // divisor) for v, n in counts.items()))


def main():
    out = HERE / "novelty"
    out.mkdir(exist_ok=True)
    sources = {"generated": HERE.parent / "de-novo/all-draws.npz",
               "training_binary": HERE.parent / "data/multi-train.npz"}
    for cohort in ("mp20", "wbm"):
        for split in ("train", "val", "test"):
            sources[f"{cohort}_{split}"] = HERE / f"{cohort}-data" / f"{split}.npz"
    arrays, tables = {}, {}
    for name, path in sources.items():
        with np.load(path) as bundle:
            arrays[name] = {k: bundle[k] for k in ("z", "k", "f", "ids")}
        table = defaultdict(list)
        for i, row in enumerate(arrays[name]["z"]):
            if len(np.unique(row[row > 0])) == 2:
                table[formula(row)].append(i)
        tables[name] = table
    matcher = StructureMatcher(**SETTINGS)

    @lru_cache(maxsize=12000)
    def structure(name, i):
        data = arrays[name]
        return make_structure(data["k"][i], data["f"][i], data["z"][i])

    groups = {"training_binary": ["training_binary"],
              "full_mp20": ["mp20_train", "mp20_val", "mp20_test"],
              "wbm_native_20": ["wbm_train", "wbm_val", "wbm_test"]}
    reps, records = defaultdict(list), []
    for i, row in enumerate(arrays["generated"]["z"]):
        key = formula(row)
        candidate = structure("generated", i)
        duplicate = next((j for j in reps[key] if matcher.fit(candidate, structure("generated", j))), None)
        if duplicate is None:
            reps[key].append(i)
        record = {"generated_id": str(arrays["generated"]["ids"][i]),
                  "first_representative": str(arrays["generated"]["ids"][i if duplicate is None else duplicate]),
                  "new_representative": duplicate is None}
        for group, names in groups.items():
            hit = next(((name, j) for name in names for j in tables[name].get(key, ())
                        if matcher.fit(candidate, structure(name, j))), None)
            record[group] = "" if hit is None else str(arrays[hit[0]]["ids"][hit[1]])
        records.append(record)
    with (out / "per-proposal.csv").open("w", encoding="utf-8", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    report = {"n_draws": len(records), "n_representatives": sum(row["new_representative"] for row in records),
              "unmatched": {group: sum(not r[group] for r in records) for group in groups},
              "unmatched_both_databases": sum(not r["full_mp20"] and not r["wbm_native_20"] for r in records),
              "reference_rows": {g: sum(len(arrays[n]["z"]) for n in names) for g, names in groups.items()},
              "matcher_settings": SETTINGS, "pymatgen_version": version("pymatgen"),
              "definition": "Unrelaxed, all draws, chemistry exact and volume rescaling enabled. Uniqueness counts first representatives in draw order. Each database result searches all its eligible records independently. No energy or validity prefilter."}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
