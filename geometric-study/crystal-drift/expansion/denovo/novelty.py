"""Score raw unfiltered structural uniqueness and novelty against all MP-20 training structures."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache
import json
from pathlib import Path
import warnings
import numpy as np

from score import HERE, SETTINGS, arrays, make, formula


@lru_cache(maxsize=1200)
def crystal(arm, i):
    return make(arm, i)


def chunk(job):
    name, block = job
    from pymatgen.analysis.structure_matcher import StructureMatcher

    warnings.filterwarnings("ignore")
    matcher = StructureMatcher(**SETTINGS)
    out = []
    for formula_key, indices, training in block:
        reps = []
        for i in indices:
            exc = ""
            if not formula_key:
                out.append((i, -1, -1, False, "empty candidate"))
                continue
            try:
                candidate = crystal(name, i)
                duplicate = next(
                    (j for j in reps if matcher.fit(candidate, crystal(name, j))), None
                )
                if duplicate is None:
                    reps.append(i)
                hit = next(
                    (j for j in training if matcher.fit(candidate, crystal("training", j))), -1
                )
                out.append((i, i if duplicate is None else duplicate, hit, bool(training), exc))
            except (
                ValueError,
                RuntimeError,
                FloatingPointError,
                OverflowError,
                np.linalg.LinAlgError,
            ) as e:
                out.append((i, -1, -1, bool(training), type(e).__name__))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    dest = HERE / "novelty"
    dest.mkdir(exist_ok=True)
    src = arrays("training")
    train = defaultdict(list)
    for i, row in enumerate(src["z"]):
        train[formula(row)].append(i)
    for name in args.arms:
        destfile = dest / f"{name}.csv"
        if destfile.exists():
            continue
        pool = arrays(name)
        group = defaultdict(list)
        for i, row in enumerate(pool["z"]):
            group[formula(row) if row.any() else ()].append(i)
        grouped = sorted(
            [(k, v, train.get(k, [])) for k, v in group.items()],
            key=lambda t: len(t[1]) * (len(t[1]) + len(t[2])),
            reverse=True,
        )
        jobs = [(name, [g]) for g in grouped[: max(12, args.workers * 2)]]
        others = grouped[len(jobs) :]
        jobs.extend((name, others[j::64]) for j in range(64))
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = sorted(item for chunk_rows in executor.map(chunk, jobs) for item in chunk_rows)
        assert [r[0] for r in rows] == list(range(len(pool["ids"])))
        with destfile.open("w", newline="", encoding="utf8") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    "index",
                    "id",
                    "representative_index",
                    "matched_train_id",
                    "seen_reduced_formula",
                    "error",
                ]
            )
            for i, rep, hit, seen, error in rows:
                writer.writerow(
                    [i, pool["ids"][i], rep, src["ids"][hit] if hit >= 0 else "", int(seen), error]
                )
        report = dict(
            draws=len(rows),
            new_representative=sum(i == r for i, r, _, _, e in rows if not e),
            unmatched_training=sum(h < 0 for _, _, h, _, e in rows if not e),
            unseen_reduced_formula=sum(not s for _, _, _, s, e in rows if not e),
            exceptions=sum(bool(e) for *_, e in rows),
            matcher=SETTINGS,
        )
        (dest / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, report, flush=True)


if __name__ == "__main__":
    main()
