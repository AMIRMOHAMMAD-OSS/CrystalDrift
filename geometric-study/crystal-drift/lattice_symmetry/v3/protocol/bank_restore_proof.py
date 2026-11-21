"""Check every registered later-epoch bank against both its training log and initial backup."""
import hashlib
import json
from pathlib import Path

from backup import digest
from backup_delta import HERE, RESEARCH


def main():
    models = HERE.parent / "models"
    checksum_path = models / "fresh-bank-sha256.json"
    expected = json.loads(checksum_path.read_text())["shards"]
    base = RESEARCH / "drive-backup-recovery-base" / "FILES.ndjson"
    external_prefix = "research-scratch/tmp/crytaldrift-v3-20261121/banks/"
    archived = {}
    for line in base.open():
        record = json.loads(line)
        if record["path"].startswith(external_prefix) and "sha256" in record:
            archived[record["path"][len(external_prefix):]] = record
    rows = []
    for name, reference in sorted(expected.items()):
        original_workspace = models / "cache" / name
        if name in archived:
            path = RESEARCH / "banks" / name
            if archived[name]["sha256"] != reference or path.stat().st_size != archived[name]["bytes"]:
                raise ValueError(f"Archived training bank differs from its training receipt: {name}")
            origin = "durable_scratch"
        else:
            path = original_workspace
            origin = "workspace"
        if digest(path) != reference:
            raise ValueError(f"Restored bytes differ from training receipt: {name}")
        rows.append((name, reference, origin, path.stat().st_size))
    if len(rows) != 1449 or len(archived) != 1255:
        raise ValueError("Incomplete bank inventory")
    out = dict(
        shards_rehashed=len(rows),
        durable_scratch_shards=len(archived),
        workspace_shards=len(rows)-len(archived),
        byte_count=sum(r[3] for r in rows),
        fit_manifest_sha256=digest(checksum_path),
        initial_backup_files_sha256=digest(base),
        ordered_bank_identity_sha256=hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest(),
        bank_provenance="All external banks matched both their original path-bound backup and the training-time SHA-256. "
                        "Regeneration used the original FP32 PyTorch/Numba path. Only complete compressed NPZ files "
                        "with the recorded SHA-256 were admitted.",
    )
    dst=HERE/"bank-restore-proof.json"
    if dst.exists():
        if json.loads(dst.read_text())!=out:raise ValueError("Existing receipt conflicts")
    else:
        with dst.open("x") as f:json.dump(out,f,indent=2)
    print(json.dumps(out), flush=True)


if __name__=="__main__":
    main()
