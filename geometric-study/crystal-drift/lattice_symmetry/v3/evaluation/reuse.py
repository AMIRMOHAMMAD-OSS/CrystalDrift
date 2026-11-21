"""Reuse byte-identical frozen predictions when a later validation look selects the same weights."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from storage import HERE,pool

def transfer(src,dst):
    if dst.is_file():
        if hashlib.sha256(dst.read_bytes()).digest()!=hashlib.sha256(src.read_bytes()).digest():
            raise ValueError(f"Previously generated destination differs: {dst}")
        return
    try:os.link(src,dst)
    except OSError:shutil.copyfile(src,dst)

def equivalent(panel,specs):
    for target in specs:
        if not target.get("sha"):continue
        b=target["tag"]
        if (pool(panel,b)/"settings.json").exists() or any(pool(panel,b).glob("?????-?????.npz")):continue
        roots=[HERE/"raw"/panel]
        if os.getenv("CRYTALDRIFT_RAW"):roots.append(Path(os.environ["CRYTALDRIFT_RAW"])/panel)
        for root in roots:
            found=False
            for source in sorted(root.glob("*/settings.json")):
                a=source.parent.name
                if a==b:continue
                old=json.loads(source.read_text());s=json.loads(old["signature"])
                if not (s["weight"]==target["sha"] and s["family"]==target.get("family","symbolic") and
                        s["selector"]==target.get("selector","lookup") and s["draws"]==20 and
                        s["spec_sha"]==hashlib.sha256((HERE.parent/"protocol/splits.json").read_bytes()).hexdigest() and
                        s["seed_stream"]=="draw_seed/20261121" and old["device"]=="cpu" and old["threads"]==1 and
                        old["batch"]==32):continue
                dst=root/b;dst.mkdir(exist_ok=True)
                files=list(source.parent.glob("?????-?????.npz"))
                if not files or sum(int(p.stem[-5:])-int(p.stem[:5]) for p in files)!=old["count"]:continue
                for f in files:transfer(f,dst/f.name)
                receipt=dict(old,identical_to=a,tag=b,weight_sha256=target["sha"],
                    derivation="Same weights, formula selector, seed bank, ordered noise, CPU precision, batch and output boundary. Original raw shard metadata and bytes retained.")
                (dst/"settings.json").write_text(json.dumps(receipt,indent=2))
                source_scores=HERE/"scores"/panel/a;dest_scores=HERE/"scores"/panel/b
                if (source_scores/"outcomes.npz").exists():
                    dest_scores.mkdir(parents=True,exist_ok=True)
                    for f in source_scores.glob("*.npz"):transfer(f,dest_scores/f.name)
                log=HERE/"aliases"/panel;log.mkdir(parents=True,exist_ok=True)
                (log/(b+".json")).write_text(json.dumps(receipt,indent=2))
                print("Identical checkpoint",panel,b,"=",a,flush=True)
                found=True;break
            if found:break
