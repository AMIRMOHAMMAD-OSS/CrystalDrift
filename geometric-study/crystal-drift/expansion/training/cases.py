"""Write a portable per-target/seed/budget companion to the raw candidate scores."""
import csv
import gzip
from pathlib import Path

import numpy as np
from analysis import BUDGETS, HERE, ROOT, SEEDS, array_metrics


def write_method(writer, ids, label, seed, data):
    for k in BUDGETS:
        values = array_metrics(data, k)
        for mid, (accepted, rms, strict, volume) in zip(ids, values):
            writer.writerow((mid, label, seed, k, int(accepted), repr(float(rms)) if accepted else "",
                             int(strict), int(volume)))


def main():
    target = HERE / "test-case-metrics.csv.gz"
    building = target.with_suffix(".writing")
    keys = ("rms", "strict_fit", "log_volume_ratio")
    with np.load(ROOT / "revision/full/mp20/cohort.npz") as ref:
        ids = ref["ids"]
    with gzip.open(building, "wt", newline="", encoding="utf-8", compresslevel=5) as stream:
        writer = csv.writer(stream)
        writer.writerow(("material_id", "method", "seed", "k", "matched", "best_accepted_normalized_rms",
                         "maximum_site_match", "native_volume_match"))
        for method in ("drift", "iid", "meanflow", "flow1", "flow16"):
            for seed in SEEDS:
                with np.load(HERE / f"test/{method}-seed{seed}/outcomes.npz") as data:
                    np.testing.assert_array_equal(ids, data["ids"])
                    write_method(writer, ids, method, seed, {key: data[key] for key in keys})
        with np.load(ROOT / "revision/full/mp20/complete-crosswalk-k20.npz") as data:
            np.testing.assert_array_equal(ids, data["ids"])
            arms = list(data["arms"])
            stored = {key: data[key] for key in keys}
            for seed in SEEDS:
                arm = arms.index(f"retrieval-seed{seed}")
                write_method(writer, ids, "retrieval", seed, {k: arr[:, arm] for k, arr in stored.items()})
        with np.load(ROOT / "revision/full/external/mp20/complete-crosswalk-k20.npz") as data:
            np.testing.assert_array_equal(ids, data["ids"])
            write_method(writer, ids, "uflow", 2026, {key: data[key] for key in keys})
        ext = HERE.parent / "external/csp-s5/score-crosswalk.npz"
        if ext.exists():
            with np.load(ext) as data:
                np.testing.assert_array_equal(ids, data["ids"])
                if data["scored"].all():
                    write_method(writer, ids, "crystalflow5", "release", {key: data[key] for key in keys})
    building.replace(target)
    print(target.name, target.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
