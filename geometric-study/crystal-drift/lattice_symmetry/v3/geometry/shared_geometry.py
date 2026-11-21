"""Exhaustive same-family anonymous reference search until the first geometric witness."""
import hashlib
import json
import os
from pathlib import Path
from importlib.metadata import version
from functools import lru_cache
for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[name]="1"
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from data import unpack_gram
from standard_match import SETTINGS
from ..protocol.lock import families

HERE=Path(__file__).resolve().parent;V3=HERE.parent;ROOT=V3.parents[1]

def h(mid,donor):
    return hashlib.sha256(("v3-family-analogue/"+mid+"/"+donor).encode()).digest()


def main():
    assert version("pymatgen")=="2024.4.13"
    splits=json.loads((V3/"protocol/splits.json").read_text());fam=families()
    train=json.loads((V3/"models/cache/inventory.json").read_text())["eligible"]
    ids=splits["shared_ids"]
    with np.load(ROOT/"scaled/mp20-data/train.npz") as d:
        I={str(x):i for i,x in enumerate(d["ids"])};Z=d["z"];F=d["f"]
        L=np.linalg.cholesky(unpack_gram(d["k"]))
    @lru_cache(maxsize=2048)
    def structure(mid):
        ix=I[mid];m=Z[ix]>0
        return Structure(L[ix],Z[ix,m],F[ix,m])
    out=HERE/"audits";out.mkdir(exist_ok=True)
    state=out/"shared-geometry-progress.json"
    signature=hashlib.sha256(("\n".join(train+ids)+json.dumps(SETTINGS,sort_keys=True)).encode()).hexdigest()
    matcher=StructureMatcher(**SETTINGS)
    records=[]
    if state.exists():
        partial=json.loads(state.read_text())
        assert partial["signature"]==signature
        records=partial["records"]
        assert ids[:len(records)]==[r["id"] for r in records]
    for mid in ids[len(records):]:
        eligible=sorted((v for v in train if fam[v]==fam[mid]),key=lambda v:h(mid,v))
        assert mid not in train and eligible
        results=[]
        for v in eligible:
            try:
                rms,mapping=matcher.get_rms_anonymous(structure(v),structure(mid))
                results.append(dict(donor=v,rms=rms,
                    map=({str(a):str(b) for a,b in mapping.items()} if mapping is not None else None)))
                if rms is not None:break
            except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError) as e:
                results.append(dict(donor=v,rms=None,error=str(e)))
        records.append(dict(id=mid,family=fam[mid],eligible=len(eligible),attempts=results))
        temp=state.with_suffix(".building")
        temp.write_text(json.dumps(dict(signature=signature,records=records),separators=(",",":"))+"\n",encoding="utf-8")
        temp.replace(state)
    result=dict(pymatgen=version("pymatgen"),matcher=SETTINGS,split="shared protected cases",
        definition="A donor in the same operational family matches under some bijection of element identities. In deterministic hash order stop at first public get_rms_anonymous match, exhaust eligible within-family donors for misses.",
        ids=len(records), matched=sum(any(r["rms"] is not None for r in x["attempts"]) for x in records),
        tried=sum(len(x["attempts"]) for x in records),errors=sum("error" in r for x in records for r in x["attempts"]),
        candidates=sum(x["eligible"] for x in records),records=records)
    (out/"shared-geometry.json").write_text(json.dumps(result,separators=(",",":"))+"\n",encoding="utf-8")
    print("Finished the registered shared-family analogue search",len(records))

if __name__=="__main__":main()
