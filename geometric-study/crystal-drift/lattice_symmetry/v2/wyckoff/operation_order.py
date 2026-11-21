"""Independently re-detect all groups and audit finite operation equality in the supplied cell."""
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[v]="1"
import numpy as np
import spglib
from charts import HERE,operations
from geometry import cohort,subgroup,TOLS
from labels import ROOT
from data import unpack_gram
from report import LABELS

def detect(L,F,Z,g,origin):
    n=len(operations(g)[0]);res=dict(requested_order=n)
    for tol in TOLS:
        d=spglib.get_symmetry_dataset((L,F,Z),symprec=tol);t=str(tol)
        res["contains"+t]=subgroup(d,L,g,origin,tol)
        res["number"+t]=int(d.number) if d is not None else 0
        res["exact"+t]=res["number"+t]==g
        res["detected_order"+t]=len(d.rotations) if d is not None else 0
        res["exactops"+t]=res["contains"+t] and n==res["detected_order"+t]
    return res

def execute(arm):
    pool=HERE/f"pools/{arm}.npz";target=cohort("test")
    with np.load(pool) as b:ls,fs,zs=b["L"],b["F"],b["Z"]
    raw=HERE/f"scores/{arm}.jsonl.gz"
    with gzip.open(raw,"rt") as fh:rows=list(map(json.loads,fh))
    post=HERE/f"relaxation/{arm}.json";relaxed=json.loads(post.read_text())
    audit={}
    for stage,seq in [("raw",rows),("relaxed",relaxed)]:
        counts={str(t):[0,0,0,0,0] for t in TOLS}
        examples={str(t):[] for t in TOLS}
        for v in seq:
            if stage=="raw" and "error" in v:continue
            i=v["index"];j=v.get("draw",1)-1;g=target[i]["group"];z=zs[i,zs[i]>0]
            origin=np.zeros(3)
            if stage=="raw":L,F=ls[i,j],fs[i,j,:len(z)]
            else:
                f=HERE/f"relaxation/records/{arm}-{i:03}.json"
                if not f.exists():continue
                record=json.loads(f.read_text())
                if "final_f" not in record:continue
                L=np.linalg.cholesky(unpack_gram(record["final_k"]));F=np.array(record["final_f"])
                origin=((F-fs[i,j,:len(z)]+.5)%1-.5).mean(0)
            q=detect(L,F,z,g,origin)
            for t in counts:
                assert v["number"+t]==q["number"+t] and v["exact"+t]==q["exact"+t]
                a=v["contains"+t]!=q["contains"+t];b=q["exact"+t]!=q["exactops"+t]
                counts[t][0]+=int(a);counts[t][1]+=int(b);counts[t][2]+=q["contains"+t]
                counts[t][3]+=q["exact"+t];counts[t][4]+=q["exactops"+t]
                if b:examples[t].append(dict(id=v["id"],draw=j+1,requested_group=g,number=q["number"+t],
                    required=q["requested_order"],detected=q["detected_order"+t],contains=q["contains"+t]))
            for tol in TOLS:
                t=str(tol)
                v['occupancy'+t]=bool(v['occupancy'+t] and v['dmin']>tol)
            v.update(q)
        audit[stage]=dict(columns=["containment_changes","type_vs_operation_set_disagreements","contained","exact_type","exact_operations"],
                          counts=counts,disagreements=examples,total=len(seq))
    tmp=raw.with_suffix(".updated")
    with gzip.open(tmp,"wt",encoding="utf8") as f:
        for r in rows:f.write(json.dumps(r,separators=(",",":"))+"\n")
    tmp.replace(raw);post.write_text(json.dumps(relaxed,indent=1)+"\n")
    print(arm,{s:{t:c[:2] for t,c in v["counts"].items()} for s,v in audit.items()},flush=True)
    return arm,audit

if __name__=="__main__":
    with ProcessPoolExecutor(3) as pool:res=dict(pool.map(execute,LABELS))
    (HERE/"operation-order-audit.json").write_text(json.dumps(res,indent=1)+"\n")
