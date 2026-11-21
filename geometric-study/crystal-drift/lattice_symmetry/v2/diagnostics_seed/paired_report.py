"""Build paired test-set comparisons from the frozen prototype ranking experiment."""
import csv
import json
import os
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent / "test384"


def write_csv(filename, rows):
    with (HERE/filename).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    with np.load(HERE/"scores.npz") as scores, np.load(HERE/"proposals.npz") as proposals:
        assert np.array_equal(scores["ids"], proposals["ids"])
        ids, arms = scores["ids"], scores["arms"].tolist()
        scaled = np.isfinite(scores["rms"]) & ~scores["errors"]
        native = scaled & (np.abs(scores["log_volume_ratio"]) <= np.log(1.1))
        large = np.count_nonzero(proposals["z"],axis=1) >= 9
        known = scores["known"]
    count_rows, pair_rows = [], []
    for stratum, mask in (("all", np.ones(len(ids), bool)),("nine_plus_atoms", large),
                          ("exact_train_composition", known),("unseen_train_composition", ~known)):
        n = int(mask.sum())
        weights = np.random.default_rng(51039).multinomial(n, np.full(n,1/n), size=10000)
        for k in (1,5,20):
            for kind, values in (("scaled",scaled),("native",native)):
                by_id = values[mask,:,:k].any(2).astype(int)
                for a, name in enumerate(arms):
                    count_rows.append(dict(stratum=stratum,targets=n,k=k,metric=kind,method=name,
                                           recovered=int(by_id[:,a].sum())))
                h = arms.index("volume_anchor")
                for name in ("uflow","archived_angular","angular","hybrid_prior","hybrid_angular"):
                    delta = by_id[:,h] - by_id[:,arms.index(name)]
                    ci = np.quantile((weights @ delta)/n*100, [.025,.975])
                    pair_rows.append(dict(stratum=stratum,targets=n,k=k,metric=kind,contrast="volume_anchor - "+name,
                                          gained=int((delta>0).sum()),lost=int((delta<0).sum()),
                                          difference_pp=100*delta.mean(),low_pp=ci[0],high_pp=ci[1]))
    write_csv("counts.csv", count_rows)
    write_csv("paired.csv", pair_rows)
    print(json.dumps([r for r in count_rows if r["stratum"]=="all" and r["k"]==20]))


if __name__ == "__main__":
    main()
