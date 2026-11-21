"""Recompute the recovery and displacement columns in the two published score tables."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


STUDY = Path(__file__).resolve().parent / "geometric-study" / "crystal-drift"
TABLES = STUDY / "lattice_symmetry" / "v3" / "release"


def compare(value, row, column, table):
    if not np.isclose(value, float(row[column]), rtol=1e-9, atol=1e-12):
        raise ValueError(
            f"{table.name}: {row['method']}, k={row['k']}, {column}: "
            f"recomputed {value}, recorded {row[column]}"
        )


def verify_table(table, main_comparison):
    with table.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_file = defaultdict(list)
    for row in rows:
        by_file[row["score_file"]].append(row)
    for filename, group in by_file.items():
        with np.load(STUDY / filename, allow_pickle=False) as scores:
            rms = scores["rms"]
            strict = scores["strict_fit"]
            volume = scores["log_volume_ratio"]
        for row in group:
            k = int(row["k"])
            if not 0 < k <= rms.shape[1]:
                raise ValueError(f"{table.name}: invalid draw count {k} for {filename}")
            first_k = rms[:, :k]
            finite = np.isfinite(first_k)
            hit = finite.any(axis=1)
            best = np.min(np.where(finite, first_k, np.inf), axis=1)
            native = (finite & (np.abs(volume[:, :k]) <= np.log(1.10))).any(axis=1)
            strict_hit = strict[:, :k].any(axis=1)
            compare(len(hit), row, "targets", table)
            compare(hit.sum(), row, "hits", table)
            if main_comparison:
                compare(hit.mean(), row, "MR", table)
                compare(best[hit].mean(), row, "mean_best_normalized_RMS", table)
                compare(native.mean(), row, "native_volume_MR", table)
                compare(strict_hit.mean(), row, "max_site_MR", table)
            else:
                compare(best[hit].mean(), row, "rms", table)
                compare(native.sum(), row, "native", table)
                compare(strict_hit.mean(), row, "strict_mr", table)
    print(f"{table.name}: verified recovery and displacement in {len(rows)} rows.")


if __name__ == "__main__":
    verify_table(TABLES / "MAIN-COMPARISON.csv", main_comparison=True)
    verify_table(TABLES / "SEED-RESULTS.csv", main_comparison=False)
