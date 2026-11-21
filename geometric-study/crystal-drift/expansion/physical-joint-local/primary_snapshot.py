"""Digest the primary five-arm inputs and results without changing them."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PRIMARY=HERE.parent/"physical"


def snapshot():
    files=[p for name in ("inputs","records","cifs") for p in (PRIMARY/name).iterdir() if p.is_file()]
    files += [PRIMARY/name for name in ("by-arm.csv","per-candidate.csv","reference-library.json","figure-selection.json")]
    h=hashlib.sha256()
    for p in sorted(files):
        h.update(p.relative_to(PRIMARY).as_posix().encode())
        h.update(b"\x00")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return dict(number_of_files=len(files),sha256=h.hexdigest())


if __name__=="__main__":
    path=HERE/"primary-before.json"
    current=snapshot()
    if path.exists():
        assert json.loads(path.read_text())==current
    else:
        path.write_text(json.dumps(current,indent=2)+"\n")
    print(current)
