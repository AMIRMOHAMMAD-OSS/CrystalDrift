"""Reconstruct candidate settings receipts for subsequent comparison to backup chunk hashes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

V3 = Path(__file__).resolve().parents[1]


def reconstruct(root, out):
    split = {"confirm": "val", "family": "train", "shared": "train", "test": "test"}
    stored = Counter()
    for path in (V3 / "evaluation/raw").glob("*/*/settings.json"):
        record = json.loads(path.read_text())
        if all(key in record for key in ("device", "batch", "threads")):
            stored[(record["device"],record["batch"],record["threads"])] += 1
    configurations = list(dict.fromkeys([("cpu",32,1),("cpu",64,1),*stored]))
    proposals = []
    for pointer in sorted((V3 / "evaluation/raw-locations").rglob("*.json")):
        panel, arm = pointer.parent.name, pointer.stem
        sources = sorted((root/"raw"/panel/arm).glob("?????-?????.npz"))
        assert sources, (panel, arm)
        signature = None
        source_tag = None
        count = 0
        for path in sources:
            with np.load(path) as data:
                current = str(data["signature"])
                tag = str(data["tag"])
                first, last = map(int,path.stem.split("-"))
                assert first == count and len(data["ids"]) == last-first
                count = last
            if signature is None: signature, source_tag = current, tag
            assert current == signature and tag == source_tag
        alias = V3/"evaluation/aliases"/panel/(arm+".json")
        candidates = {}
        if alias.is_file():
            exact = alias.read_bytes()
            archived = json.loads(exact)
            assert archived["count"] == count and archived["signature"] == signature
            candidates["alias"] = exact
        else:
            assert source_tag == arm, (panel, arm, source_tag)
            choices = configurations if arm != "reference-scale-11" else [("cuda",32,1),*configurations]
            for device,batch,threads in choices:
                value = dict(split=split[panel],panel=panel,count=count,
                             draws=json.loads(signature)["draws"],signature=signature,
                             device=device,batch=batch,threads=threads)
                candidates[f"{device}-{batch}-{threads}"] = (json.dumps(value)+"\n").encode()
        expected = "research-scratch/tmp/crytaldrift-v3-20261121/raw/" + panel + "/" + arm + "/settings.json"
        row = dict(original=expected, source_tag=source_tag, shards=len(sources), targets=count, candidates={})
        for config, data in candidates.items():
            path = out/panel/arm/(config+".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() != data: raise FileExistsError(path)
            path.write_bytes(data)
            row["candidates"][config] = {"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data),"file":str(path.relative_to(out))}
        proposals.append(row)
    (out/"settings-candidates.json").write_text(json.dumps(proposals,indent=1)+"\n")
    print(len(proposals),"settings receipts reconstructed;",sum(len(r["candidates"]) for r in proposals),"candidate digests",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("recovered_root",type=Path)
    parser.add_argument("destination",type=Path)
    args=parser.parse_args()
    reconstruct(args.recovered_root,args.destination)
