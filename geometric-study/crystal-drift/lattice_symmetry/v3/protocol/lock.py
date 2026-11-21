"""Register MP-20 cohorts from identifiers and train-only symmetry metadata."""
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V2 = HERE.parents[1] / "v2"
SALT = "CrytalDrift-v3-20261121-01/"

def digest(tag, value):
    return hashlib.sha256((SALT + tag + "/" + str(value)).encode()).hexdigest()

def chemistry(z):
    e, n = np.unique(z[z > 0], return_counts=True)
    return tuple(zip(e.tolist(), (n // np.gcd.reduce(n)).tolist()))

def families():
    with np.load(V2/"wyckoff/charts.npz") as data:
        gs, pos, mat = (data[k] for k in ("groups","positions","matrices"))
        attrs = {}
        for g in range(230):
            for i in range(gs[g], gs[g+1]):
                letter = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"[i-gs[g]]
                attrs[g+1,letter] = (int(pos[i+1]-pos[i]), int(np.linalg.matrix_rank(mat[pos[i]])))
    out = {}
    for file in sorted((V2/"wyckoff/labels").glob("train-*.gz")):
        for raw in gzip.open(file, "rt"):
            r = json.loads(raw)
            if "error" in r:
                continue
            groups = collections.defaultdict(list)
            for z,l in r["template"]:
                groups[z].append(attrs[r["group"],l])
            family = [r["group"], sorted(sorted(vals) for vals in groups.values())]
            out[r["id"]] = json.dumps(family, separators=(",",":"))
    return out

def main():
    splits = {}
    hashes = {}
    for s in ("train","val","test"):
        file = ROOT/f"scaled/mp20-data/{s}.npz"
        hashes[s] = hashlib.sha256(file.read_bytes()).hexdigest()
        with np.load(file) as data:
            ids = data["ids"].astype(str).tolist()
            assert len(set(ids)) == len(ids)
            splits[s] = ids
            if s == "train":
                chem = dict(zip(ids, map(chemistry, data["z"])))
    assert not any(set(splits[a]) & set(splits[b]) for a,b in (("train","val"),("val","test"),("train","test")))
    fam = families()
    counts = collections.Counter(fam.values())
    # Exclude large families from challenge selection to avoid a single dominant family.
    eligible = sorted((k for k,n in counts.items() if 3<=n<=30),key=lambda k:digest("family",k))
    held = set(eligible[:24])
    complete = sorted((x for x in fam if fam[x] in held),key=lambda x:digest("family-id",x))
    # Within represented families, remove entire chemical systems at fixed reduced formula.
    seed_ids = sorted((x for x in fam if 12<=counts[fam[x]] and fam[x] not in held),key=lambda x:digest("shared",x))
    shared = []
    chosen = set()
    for x in seed_ids:
        if fam[x] not in chosen:
            shared.append(x)
            chosen.add(fam[x])
        if len(shared)==96: break
    blocked_chem = {chem[x] for x in complete+shared}
    train = [x for x in splits["train"] if x in fam and fam[x] not in held and chem[x] not in blocked_chem]
    retained = {fam[x] for x in train}
    assert set(fam[x] for x in shared)<=retained and not retained & held
    val = sorted(splits["val"],key=lambda x:digest("validation",x))
    existing = set()
    for source in [*list((V2/"csp/fits/angular-11").glob("val-*.npz")),V2/"diagnostics_site_gain/validation.npz"]:
        if source.is_file():
            with np.load(source) as v:
                if "ids" in v: existing.update(v["ids"].astype(str))
    for file in (V2/"wyckoff/val-cohort.json",):
        if file.is_file(): existing.update(x["id"] for x in json.loads(file.read_text()))
    for file in [V2/"diagnostics_template_prediction/draws.csv"]:
        existing.update(x["id"] for x in csv.DictReader(file.open()))
    val=[x for x in val if x not in existing]
    payload=dict(train_ids=train,dev_ids=val[:96],confirm_ids=val[96:192],family_ids=complete,
                 shared_ids=shared,test_ids=splits["test"],sha256=hashes,held_families=sorted(held),
                 family_by_challenge_id={x:fam[x] for x in complete+shared},
                 counts=dict(raw_train=len(splits["train"]),chart_failed=len(splits["train"])-len(fam),
                   removed_family=len(complete),additional_removed_chemistry=len(fam)-len(complete)-len(train),
                   retained=len(train),original_families=len(counts),retained_families=len(retained),shared=len(shared),
                   heldout=len(held),dev=96,confirm=96,test=len(splits["test"]),previous_validation_exclusions=len(existing)))
    out=HERE/"splits.json"
    serialized=json.dumps(payload,separators=(",",":"))+"\n"
    if out.is_file() and out.read_text()!=serialized:
        raise ValueError("split differs from registered file")
    out.write_text(serialized)
    print(payload["counts"],hashlib.sha256(out.read_bytes()).hexdigest())

if __name__=="__main__":
    main()
