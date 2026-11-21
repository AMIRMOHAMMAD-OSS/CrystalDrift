"""As-generated symmetry distributions against held-out and chemistry-source structures."""
import argparse
import csv
from itertools import combinations
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[name]="1"
import numpy as np
import spglib
from scipy.spatial.distance import jensenshannon
from .cif_catalog import HERE,ROOT,sha,unpack_gram


def evaluate_reference(ids, input_file, dest):
    if dest.is_file():
        with np.load(dest) as cached:
            assert np.array_equal(cached["ids"],ids) and str(cached["raw_sha256"])==sha(input_file)
            return cached["detected"]
    with np.load(input_file) as d:
        lookup={mid:i for i,mid in enumerate(d["ids"].astype(str))}
        z,f,k=(d[name] for name in ("z","f","k"))
    memo={}
    for mid in dict.fromkeys(ids.tolist()):
        i=lookup[mid];m=z[i]>0
        sg=spglib.get_symmetry_dataset((np.linalg.cholesky(unpack_gram(k[i])),f[i,m].astype(float),z[i,m]),symprec=.01)
        memo[mid]=sg.number if sg is not None else 0
    values=np.array([memo[mid] for mid in ids],np.int16)
    np.savez_compressed(dest,ids=ids,detected=values,raw_sha256=sha(input_file))
    return values


def main(arms):
    folder=HERE/"catalog"
    with np.load(HERE/"pools/prior.npz") as p:
        source_ids=p["chemistry_source"];draw_ids=p["ids"]
    with np.load(ROOT/"scaled/mp20-data/test.npz") as p:test_ids=p["ids"]
    test=evaluate_reference(test_ids,ROOT/"scaled/mp20-data/test.npz",folder/"reference-test.npz")
    source=evaluate_reference(source_ids,ROOT/"scaled/mp20-data/train.npz",folder/"reference-chemistry-source.npz")
    protected=set(json.loads((HERE.parent/"protocol/splits.json").read_text())["train_ids"])
    assert set(source_ids)<=protected
    def hist(x):return np.bincount(x,minlength=231)
    distributions={"held_out_reference":hist(test),"source_structures":hist(source)}
    summary={}
    detected_by_arm={}
    for name in arms:
        with np.load(folder/f"{name}-spacegroups.npz") as p:
            assert np.array_equal(p["ids"],draw_ids)
            g=p["detected"]
        with np.load(HERE/f"pools/{name}.npz") as p:
            assert np.array_equal(p["chemistry_source"],source_ids)
        counts=hist(g);distributions[name]=counts
        detected_by_arm[name]=g
        summary[name]=dict(attempts=len(g),detected=int((g>0).sum()),p1=int((g==1).sum()),
            distinct_positive_types=int(np.count_nonzero(counts[1:])),
            js_distance_base2_to_test=float(jensenshannon(counts,hist(test),base=2)),
            js_distance_base2_to_composition_sources=float(jensenshannon(counts,hist(source),base=2)),
            equal_source_type=int(((g>0)&(g==source)).sum()))
    with (folder/"spacegroup-counts.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["international_number_0_is_failure",*distributions])
        for i in range(231):w.writerow([i,*[q[i] for q in distributions.values()]])
    paired={f"{a}/{b}":int(np.count_nonzero(detected_by_arm[a]!=detected_by_arm[b]))
            for a,b in combinations(arms,2)}
    (folder/"spacegroup-summary.json").write_text(json.dumps(dict(reference_test_n=len(test),
        composition_source_n=len(source),spglib=spglib.__version__,symprec_A=.01,arms=summary,
        paired_draws_with_different_detection=paired,
        distance="Square root of Jensen-Shannon divergence in bits, probability vectors over group 0..230 including detection failures."),
        indent=2)+"\n",encoding="utf-8")
    print(summary,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("arms",nargs="+")
    main(parser.parse_args().arms)
