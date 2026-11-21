"""Outcome-blind and resumable validation search for unseen geometry at a represented pattern."""
import argparse,csv,gzip,hashlib,json,os
from collections import Counter,defaultdict
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
import numpy as np
from data import unpack_gram
from standard_match import SETTINGS

HERE=Path(__file__).resolve().parent;V3=HERE.parent;V2=V3.parent/"v2";ROOT=V3.parents[1];O=HERE/"audits"
ALT=json.loads((HERE/"letter-tables.json").read_text())["groups"]
with np.load(V2/"wyckoff/charts.npz") as d:
    group,pos,mat=[d[k] for k in ("groups","positions","matrices")]
LETTERS="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
DOF={(g,l):int(np.linalg.matrix_rank(mat[pos[group[g-1]+j]])) for g in range(1,231)
     for j,l in enumerate(LETTERS[:group[g]-group[g-1]])}

def sha(x):return hashlib.sha256(x).hexdigest()
def write(file,data):
    tmp=file.with_suffix(".building");tmp.write_text(json.dumps(data,separators=(",",":"))+"\n",encoding="utf-8")
    tmp.replace(file)

def split_rows(split):
    for f in sorted((V2/"wyckoff/labels").glob(split+"-*.gz")):
        with gzip.open(f,"rt") as stream:
            for row in stream:yield json.loads(row)

def anonymous(r):
    g=r["group"];pat=r["template"];maps=ALT[str(g)]
    s=defaultdict(list)
    for z,l in pat:s[z].append(maps[0].index(l))
    key=min(tuple(sorted(tuple(sorted(perm[i] for i in v)) for v in s.values())) for perm in maps)
    return json.dumps([g,key],separators=(",",":"))

def exclusions(split,valid):
    allids=set(split["dev_ids"]+split["confirm_ids"]);log={"v3 dev/confirm":len(allids)}
    def add(path,ids):
        found=set(map(str,ids))&valid;log[str(path.relative_to(ROOT))]=len(found);allids.update(found)
    for name in ("val-ab-per-case.csv","val-multi-per-case.csv"):
        path=ROOT/"comparison"/name
        with path.open(newline="") as f:add(path,[r["id"] for r in csv.DictReader(f)])
    path=V2/"diagnostics_template_prediction/draws.csv"
    with path.open(newline="") as f:add(path,[r["id"] for r in csv.DictReader(f)])
    path=V2/"wyckoff/val-cohort.json"
    add(path,[r["id"] for r in json.loads(path.read_text())])
    # Raw subsets can have informed earlier validation even if only a subset was scored.
    paths=list((ROOT/"data").glob("*val.npz"))
    for d in (ROOT/"scaled/runs/mp20",ROOT/"revision/ablation",ROOT/"revision/binary",ROOT/"revision/full/mp20",
              ROOT/"expansion",V2/"wide",V2/"csp/fits",V2/"diagnostics_onecall",V2/"diagnostics_site_gain",
              V2/"diagnostics_seed",V2/"diagnostics_template_prediction"):
        for p in d.rglob("*.npz"):
            n=p.name.lower()
            if "val" in n.replace("retrieval","") or "diagnostics_" in str(p):
                paths.append(p)
    for p in sorted(set(paths)):
        with np.load(p) as d:
            for k in ("ids","material_ids","case_ids","target_ids"):
                if k in d and d[k].dtype.kind in "US":
                    add(p,d[k].flatten());break
    return allids,log

def register(frozen_exclusions=False):
    split=json.loads((V3/"protocol/splits.json").read_text())
    train=set(json.loads((V3/"models/cache/inventory.json").read_text())["eligible"])
    banks=defaultdict(list)
    for r in split_rows("train"):
        if r["id"] in train:banks[anonymous(r)].append(r["id"])
    val=list(split_rows("val"))
    path=O/"arrangement-registration.json"
    if frozen_exclusions:
        saved=json.loads(path.read_text())
        fixed=json.loads((O/"arrangement-frozen.json").read_text())
        assert fixed["registration_sha256"]==sha(path.read_bytes())
        excluded=set(saved["excluded_ids"]);log=saved["exclusions"]
        assert set(split["dev_ids"]+split["confirm_ids"])<=excluded<={r["id"] for r in val}
    else:
        excluded,log=exclusions(split,{r["id"] for r in val})
    counts=Counter();eligible=[]
    for r in val:
        mid=r["id"]
        if mid in excluded:counts["prior_panel"]+=1;continue
        if "error" in r:counts["chart_failure"]+=1;continue
        if r["native_n"]!=r["primitive_n"]:counts["nonprimitive"]+=1;continue
        if not sum(DOF[r["group"],l] for _,l in r["template"]):counts["fixed_chart"]+=1;continue
        key=anonymous(r);bank=banks[key]
        if not bank:counts["unrepresented"]+=1;continue
        if len(bank)>64:counts["catalog_over_64"]+=1;continue
        eligible.append(dict(id=mid,key=key,donors=sorted(bank,key=lambda x:sha(
            ("v3-arrangement-donor/"+mid+"/"+x).encode()))))
    counts["eligible"]=len(eligible);assert sum(counts.values())==len(val)
    eligible.sort(key=lambda x:sha(("v3-arrangement-challenge-20261121/"+x["id"]).encode()))
    data=dict(protocol_sha256=sha((HERE/"ARRANGEMENT-PROTOCOL.md").read_bytes()),
              exclusions=log,excluded_ids=sorted(excluded),counts=counts,registered=eligible[:512])
    if path.exists():assert json.loads(path.read_text())==data
    else:write(path,data)
    print("registered",len(data["registered"]),"of",len(val),flush=True)

def classify():
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    assert version("pymatgen")=="2024.4.13"
    path=O/"arrangement-registration.json";registered=json.loads(path.read_text());signature=sha(path.read_bytes())
    state=O/"arrangement-progress.json";rows=[]
    if state.exists():
        old=json.loads(state.read_text());assert old["registration_sha"]==signature;rows=old["rows"]
    native={}
    for split in ("train","val"):
        with np.load(ROOT/f"scaled/mp20-data/{split}.npz") as d:
            L=np.linalg.cholesky(unpack_gram(d["k"]));F,Z=d["f"],d["z"]
            native[split]=({str(s):i for i,s in enumerate(d["ids"])},L,F,Z)
    @lru_cache(maxsize=512)
    def structure(split,mid):
        ids,L,F,Z=native[split];i=ids[mid];mask=Z[i]>0
        return Structure(L[i],Z[i,mask],F[i,mask])
    matcher=StructureMatcher(**SETTINGS)
    for a in registered["registered"][len(rows):]:
        if sum(x["status"]=="uncovered" for x in rows)>=96:break
        mid=a["id"];attempts=[];outcome="uncovered"
        for v in a["donors"]:
            try:
                rms,mapping=matcher.get_rms_anonymous(structure("train",v),structure("val",mid))
                attempts.append(dict(id=v,rms=rms,element_map={str(x):str(y) for x,y in mapping.items()} if mapping else None))
                if rms is not None:outcome="covered";break
            except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as error:
                attempts.append(dict(id=v,rms=None,error=str(error)));outcome="unknown"
        assert outcome!="uncovered" or len(attempts)==len(a["donors"])
        rows.append(dict(id=mid,key=a["key"],status=outcome,attempts=attempts))
        write(state,dict(registration_sha=signature,rows=rows))
    ids={k:[r["id"] for r in rows if r["status"]==k] for k in ("uncovered","covered","unknown")}
    result=dict(registration_sha256=signature,classification_sha256=sha(state.read_bytes()),panel_split="val",
                uncovered_ids=ids["uncovered"],covered_ids=ids["covered"][:96],unknown_ids=ids["unknown"],
                key_by_id={r["id"]:r["key"] for r in rows},counts={s:len(v) for s,v in ids.items()},
                inspected=len(rows),registered=len(registered["registered"]),n_pairs=sum(len(r["attempts"]) for r in rows),
                minimum_rms_rule="a witness at any accepted RMS; require exhaustion for uncovered")
    write(O/"arrangement-frozen.json",result);print(result["counts"],result["n_pairs"],flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("task",choices=["register","replay-register","classify"]);a=ap.parse_args()
    O.mkdir(exist_ok=True)
    if a.task=="classify":classify()
    else:register(frozen_exclusions=a.task=="replay-register")
