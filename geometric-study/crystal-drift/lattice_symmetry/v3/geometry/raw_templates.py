"""Verify retrospective shortlist flags against the actual ordered formula-only proposal records."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .reachability import OUT, V3, keys, records
from ..evaluation.storage import pool


def audit(panel, methods):
    split = "train" if panel in ("family", "shared") else "test" if panel == "test" else "val"
    labels = {r["id"]: r for r in records(split)}
    train_ids = set(json.loads((V3 / "models/cache/inventory.json").read_text())["eligible"])
    with np.load(OUT / f"reachable-{panel}.npz") as saved:
        target_ids = saved["ids"].tolist()
        flags = {k: saved[k] for k in ("canonical", "equivalent", "family", "group", "fixed")}
    raw_ids = []
    seen_donors, seen_cells = set(), set()
    source = hashlib.sha256()
    settings_by_tag={}
    incomplete = []
    shards = sorted(pool(panel,"prior").glob("*.npz"))
    assert shards
    for path in shards:
        with np.load(path) as raw:
            ids, group = raw["ids"].tolist(),raw["group"]
            assert not np.any(raw["errors"] != ""), ("generation failure prevents coverage verification",path)
            fields = {k: raw[k] for k in ("ids","group","patterns","donor","cell_donor","errors")}
            for k in ("group","patterns","donor","cell_donor"):
                source.update(fields[k].tobytes())
            seen_donors.update(fields["donor"].ravel().tolist())
            seen_cells.update(fields["cell_donor"].ravel().tolist())
            for row,(mid,text) in enumerate(zip(ids, fields["patterns"])):
                ix = len(raw_ids)
                assert mid == target_ids[ix]
                raw_ids.append(mid)
                truth = labels[mid]
                known = "error" not in truth
                if known:
                    tg = truth["group"]
                    tp = tuple(sorted(tuple(x) for x in truth["template"]))
                    tf,te,_ = keys(tg,tp)
                patterns = json.loads(text)
                assert len(patterns) == group.shape[1] == 20
                for j,raw_pattern in enumerate(patterns):
                    g,p = int(group[row,j]),tuple(sorted(tuple(x) for x in raw_pattern))
                    f,e,n = keys(g,p)
                    expected = dict(fixed=n == 0, group=known and g == tg,
                        canonical=known and (g,p) == (tg,tp), equivalent=known and e == te,
                        family=known and f == tf)
                    assert all(bool(flags[k][ix,j]) == v for k,v in expected.items()), (mid,j,g,p,expected)
        for tag in methods:
            twin = pool(panel,tag)/path.name
            if not twin.is_file():
                incomplete.append((tag,path.name))
                continue
            with np.load(twin) as raw:
                setting = json.loads(str(raw["signature"]))
                assert setting["family"] == "symbolic" and setting["selector"] == "lookup",setting
                if tag in settings_by_tag:assert settings_by_tag[tag]==setting
                settings_by_tag[tag]=setting
                assert all(np.array_equal(raw[k],v) for k,v in fields.items()), (tag,path.name)
    assert raw_ids == target_ids, (panel, len(raw_ids),len(target_ids))
    assert not incomplete, incomplete
    assert seen_donors - {""} <= train_ids and seen_cells - {""} <= train_ids
    destination=OUT/f"raw-templates-{panel}.json"
    earlier=json.loads(destination.read_text()) if destination.is_file() else {}
    if earlier:
        assert earlier["metadata_sha256"]==source.hexdigest() and earlier["targets"]==len(raw_ids)
        methods=list(dict.fromkeys(earlier["matching_metadata_methods"]+methods))
        settings_by_tag={**earlier.get("signatures",{}),**settings_by_tag}
    result = dict(targets=len(raw_ids), ordered_draws=20*len(raw_ids), metadata_sha256=source.hexdigest(),
        protected_donors=len(seen_donors - {""}), protected_cell_donors=len(seen_cells - {""}),
        matching_metadata_methods=methods, signatures=settings_by_tag,checked_flags=list(flags),complete=True)
    destination.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(panel, result, flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("panel")
    p.add_argument("tags",nargs="*")
    args=p.parse_args()
    audit(args.panel,args.tags)
