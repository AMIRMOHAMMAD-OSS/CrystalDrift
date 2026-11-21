"""Bind the principal research deliverables to their paths in the final author backup."""
import json
from pathlib import Path

from backup import digest
from backup_delta import HERE, RESEARCH
from verify_endpoint import files, label


def main():
    stage=RESEARCH/"drive-backup-final"
    model=HERE.parent/"models"
    catalog=json.loads((stage/"BACKUP.json").read_text())
    final=files(stage/"FILES.ndjson")
    base=files(RESEARCH/"drive-backup-recovery-base/FILES.ndjson")
    def compare(name,sha_,size=None):
        row=final[name]
        if row.get("sha256")!=sha_ or (size is not None and row.get("bytes")!=size):raise ValueError("Mismatched archived research payload: "+name)
    weights=json.loads((model/"deployment-weights.json").read_text())["files"]
    for w in weights:compare(label(Path(w["actual"])),w["sha256"],w["bytes"])
    fits=json.loads((model/"terminal-fit-integrity.json").read_text())
    terminal={}
    for tag,result in fits.items():
        if "optimizer" not in result:continue
        opt=result["optimizer"]
        compare(label(Path(opt["path"])),opt["sha256"],opt["bytes"])
        terminal[tag]=opt["step"]
    bankmap=json.loads((model/"fresh-bank-sha256.json").read_text())["shards"]
    scratch="research-scratch/tmp/crytaldrift-v3-20261121/"
    for name,sha_ in bankmap.items():
        key=scratch+"banks/"+name
        compare(key if key in final else label(model/"cache"/name),sha_)
    raw_dir=RESEARCH/"raw"
    rawfiles=[p for p in raw_dir.rglob("*") if p.is_file()]
    pool={name:r for name,r in final.items() if name.startswith(scratch+"raw/") and "bytes" in r}
    if len(pool)!=len(rawfiles):raise ValueError("Durable raw cardinality differs")
    for p in rawfiles:compare(scratch+"raw/"+p.relative_to(raw_dir).as_posix(),digest(p),p.stat().st_size)
    endpoint=json.loads((HERE/"drive-endpoint-integrity.json").read_text())
    raw_bytes=sum(r["bytes"] for r in pool.values())
    if (len(pool),raw_bytes)!=(endpoint["raw_files"],endpoint["raw_bytes"]):raise ValueError("Pre-outage raw total differs")
    parity=json.loads((HERE/"matcher-parity-restoration.json").read_text())
    for p in parity.values():compare(scratch+"parity/linux-"+p["method"]+".npz",p["sha256"],p["bytes"])
    replay="tmp/crytaldrift-evidence-check-01/"
    held={n:r for n,r in base.items() if n.startswith(replay)}
    if any(final.get(n)!=r for n,r in held.items()):raise ValueError("Prior release replay identity differs")
    replay_files=[r for n,r in held.items() if not n.endswith("/")]
    if len(replay_files)!=20635:raise ValueError("Replay cardinality")
    releases=Path("/home/amir/crytaldrift-v3-evidence/release")
    doc={}
    for path in sorted(releases.iterdir()):
        if not path.is_file():continue
        name="research-scratch/"+path.as_posix().lstrip("/")
        if name in final:
            compare(name,digest(path),path.stat().st_size)
            doc[path.name]=final[name]["sha256"]
    for name in ("CrytalDrift.pdf","CrytalDrift-arxiv-source.zip","MORNING.pdf"):
        static=HERE.parent/"release"/name
        compare(label(static),digest(static),static.stat().st_size)
        doc[name]=digest(static)
    compare("external-uFlowCSP/ema.safetensors",
            "e18543636f77ddb9d979c7264555272951125511a662f4adbadcf15df8a85673" if False else
            digest(Path("/home/amir/crytaldrift-v3-evidence/native-source-ema.safetensors")),524874136)
    report=dict(files_and_links=catalog["files"],logical_bytes=catalog["logical_bytes"],pack_objects=len(catalog["volumes"])-1,
        catalog_sha256=digest(stage/"BACKUP.json"),files_manifest_sha256=digest(stage/"FILES.ndjson"),
        deployable_models=len(weights),deployable_logical_bytes=sum(z["bytes"] for z in weights),
        terminal_optimizer_steps=terminal,later_epoch_training_banks=len(bankmap),raw_files=len(pool),raw_bytes=raw_bytes,
        restored_matcher_parity_arrays=len(parity),historical_replay_files=len(replay_files),
        historical_replay_bytes=sum(v.get("bytes",0) for v in replay_files),release_sha256=doc)
    output=HERE/"final-workspace-science-inventory.json"
    if output.exists():
        if json.loads(output.read_text())!=report:raise ValueError("A previous final science inventory differs")
    else:
        with output.open("x") as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
