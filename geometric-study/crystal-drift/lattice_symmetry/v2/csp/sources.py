"""Bind checkpoint source digests to the exact stored module bytes from the pilot and extended controls."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent


def sha(body):return hashlib.sha256(body).hexdigest()


def main():
    need={}
    for f in (HERE/"fits").glob("*/update-*.json"):
        state=json.loads(f.read_text())
        if "implementation" not in state:continue
        for name,h in state["implementation"].items():need.setdefault(name,set()).add(h)
    model=(HERE/"model.py").read_text()
    no_prior=model.replace('    if arm=="prior":return HERE/"prior.npz"\n','').replace('    if arm=="prior":return Prior.load(path).draw\n','')
    no_power=no_prior.replace('("radial","angular","power0","fixed-angular","flow16")','("radial","angular","fixed-angular","flow16")').replace('("angular","power0","radial")','("angular","radial")')
    train=(HERE/"train.py").read_text()
    old_train=train.replace('    slices=blocks(angular=angular)\n    dim=slices["power0"].stop if arm=="power0" else max(v.stop for v in slices.values())','    dim=max(v.stop for v in blocks(angular=angular).values())').replace('**DESCRIPTOR)[:,:dim]-mu','**DESCRIPTOR)-mu')
    paths={"model.py":(model,no_prior,no_power),"train.py":(train,old_train),
           "family.py":((HERE.parents[2]/"scaled/family.py").read_text(),),
           "physical.py":((HERE.parent/"physical.py").read_text(),),
           "neighbors.py":((HERE.parent/"neighbors.py").read_text(),)}
    audit={}
    for name,requested in need.items():
        bodies={}
        for text in paths[name]:
            for body in (text.encode(),text.replace("\n","\r\n").encode()):bodies[sha(body)]=body
        for h in sorted(requested):
            if h not in bodies:raise ValueError(f"Unknown checkpoint source {name}: {h}")
            out=HERE/"source-versions"/h/name
            out.parent.mkdir(exist_ok=True,parents=True)
            out.write_bytes(bodies[h])
            assert sha(out.read_bytes())==h
            audit[h]=out.relative_to(HERE).as_posix()
    (HERE/"source-versions/index.json").write_text(json.dumps(audit,indent=2)+"\n")
    print("Recovered all",len(audit),"individually hashed fit-source files.")

if __name__=="__main__":main()
