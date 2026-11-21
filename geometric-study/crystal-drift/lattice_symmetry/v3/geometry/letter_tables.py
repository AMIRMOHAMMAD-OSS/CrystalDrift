"""Extract transformations from the same upstream PyXtal wheel as the v2 chart archive."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p=argparse.ArgumentParser();p.add_argument("wheel",type=Path);arg=p.parse_args()
    here=Path(__file__).resolve().parent
    meta=json.loads((here.parents[1]/"v2/wyckoff/chart-source.json").read_text())
    assert hashlib.sha256(arg.wheel.read_bytes()).hexdigest()==meta["sha256"]
    with zipfile.ZipFile(arg.wheel) as archive:
        content=archive.read("pyxtal/database/wyckoff_sets.json")
    out={}
    for g,entry in json.loads(content).items():
        letters=list(dict.fromkeys("".join(v.split()) for v in entry["Transformed WP"]))
        assert all(sorted(q)==sorted(letters[0]) for q in letters)
        out[g]=letters
    (here/"letter-tables.json").write_text(json.dumps(dict(source=meta["url"],
        wheel_sha256=meta["sha256"],data_sha256=hashlib.sha256(content).hexdigest(),groups=out),
        separators=(",",":"))+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
