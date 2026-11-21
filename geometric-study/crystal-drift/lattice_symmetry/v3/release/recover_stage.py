"""Rehydrate lost ephemeral research paths from their verified portable archive members."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from zipfile import ZipFile

from build_evidence import BASE, ROOT, V3, sha


def recover(stage, destination, pilot):
    stage = stage.resolve(strict=True)
    destination = destination.resolve()
    archive = stage / "payloads.zip"
    registration = json.loads((stage / "stage.json").read_text())
    if sha(archive) != registration["sha256"]:
        raise ValueError("Scientific stage SHA-256 mismatch")
    records = registration["files"]
    queue = {}
    temporary_roots = set()

    def add(original, key, expected=None):
        name = str(original).replace("\\", "/")
        if name.startswith("//wsl.localhost/Ubuntu/"):
            name = "/" + name.split("/", 4)[4]
        path = Path(name)
        if not path.is_relative_to("/tmp") or not path.parts[2].startswith("crytaldrift-"):
            return
        if original is None or ".." in path.parts or key not in records:
            if expected:
                raise ValueError((name, key))
            return
        if expected and expected != records[key]["sha256"]:
            candidates = [k for k in records if k.startswith(str(Path(key).parent) + "/deployment-history/") and records[k]["sha256"] == expected]
            if len(candidates) != 1:
                raise ValueError(("stage and pointer disagree", key, expected))
            key = candidates[0]
        temporary_roots.add(path.parts[2])
        out = destination / path.relative_to("/tmp")
        if out in queue and queue[out] != key:
            raise ValueError(("conflicting recovery sources", out, key))
        queue[out] = key

    for category in ("models", "uflow"):
        fits = V3 / category / "fits"
        for receipt in sorted(fits.glob("*/*.pointer.json")):
            reg = json.loads(receipt.read_text())
            add(reg["path"], receipt.relative_to(ROOT).as_posix().removesuffix(".pointer.json"), reg.get("sha256"))
        for receipt in sorted(fits.glob("*/resume-location.json")):
            reg = json.loads(receipt.read_text())
            original = reg.get("scratch_file") or reg.get("path")
            add(original, receipt.with_name("resume.pt").relative_to(ROOT).as_posix(), reg.get("sha256"))
    for receipt in sorted((V3 / "evaluation/raw-locations").rglob("*.json")):
        reg = json.loads(receipt.read_text())
        prefix = BASE + reg["relative"] + "/"
        for key in records:
            if key.startswith(prefix):
                add(str(Path(reg["actual"]) / key[len(prefix):]), key)

    destination.mkdir(parents=True, exist_ok=True)
    unique = {}
    def restore(z, member, out, expected):
        if not out.resolve().is_relative_to(destination):
            raise ValueError(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            if out.stat().st_size != expected["bytes"] or sha(out) != expected["sha256"]:
                raise FileExistsError(out)
        elif expected["sha256"] in unique:
            os.link(unique[expected["sha256"]], out)
        else:
            temp = out.with_name(out.name + ".recovering")
            digest = hashlib.sha256()
            with z.open(member) as source, temp.open("wb") as target:
                for block in iter(lambda: source.read(4 << 20), b""):
                    if shutil.disk_usage("/mnt/c").free < (320 << 20):
                        raise OSError("Insufficient host space for WSL recovery")
                    target.write(block)
                    digest.update(block)
            if digest.hexdigest() != expected["sha256"] or temp.stat().st_size != expected["bytes"]:
                raise ValueError(out)
            temp.replace(out)
        unique[expected["sha256"]] = out
    print("Restoring",len(queue),"registered raw or checkpoint paths",flush=True)
    with ZipFile(archive) as z:
        for out, key in sorted(queue.items()):
            restore(z, key, out, records[key])
    pilot_hash = sha(pilot)
    pilot_receipt = json.loads(pilot.with_suffix(".receipt.json").read_text())
    if pilot_hash != pilot_receipt["sha256"]:
        raise ValueError("Prior source pilot SHA-256 mismatch")
    with ZipFile(pilot) as z:
        source = json.loads(z.read(BASE + "release/EVIDENCE-MANIFEST.json"))["files"]
        for key, spec in source.items():
            prefix = BASE + "uflow/vendor/"
            if key.startswith(prefix):
                restore(z, key, destination / "vendor" / key[len(prefix):], spec)
    for name in sorted(temporary_roots):
        alias = Path("/tmp") / name
        target = destination / name
        if not alias.exists():
            alias.symlink_to(target, target_is_directory=True)
        elif alias.resolve() != target:
            raise FileExistsError(alias)
    result = dict(stage_sha256=registration["sha256"], source_pilot_sha256=pilot_hash,
                  paths=len(queue), unique_sha256=len(unique), temporary_roots=sorted(temporary_roots))
    (destination / "stage-recovery.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("source_pilot", type=Path)
    args = parser.parse_args()
    recover(args.stage, args.destination, args.source_pilot)
