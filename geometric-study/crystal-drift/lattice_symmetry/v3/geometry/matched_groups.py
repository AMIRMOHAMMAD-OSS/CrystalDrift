"""What group does a geometry accepted by the scale-enabled matcher actually have?"""
import argparse
import json
import os
from pathlib import Path
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
import numpy as np
from .subgroups import group_check
from ..evaluation.storage import pool

HERE=Path(__file__).resolve().parent;OUT=HERE/"audits";E=HERE.parent/"evaluation"
def main():
    p=argparse.ArgumentParser();p.add_argument("panel");p.add_argument("tags",nargs="+");a=p.parse_args()
    with np.load(OUT/f"reachable-{a.panel}.npz") as v:
        ids=v["ids"];truth=dict(zip(ids.astype(str),v["truth_group"]))
    out={}
    for tag in a.tags:
        sd=E/f"scores/{a.panel}/{tag}";rd=pool(a.panel,tag);rows=[];n=0
        with np.load(sd/"outcomes.npz") as score:
            assert np.array_equal(score["ids"],ids)
            all_rms=score["rms"][:,:5]
        for raw in sorted(rd.glob("?????-?????.npz")):
            with np.load(raw) as d:
                take=slice(n,n+len(d["ids"]))
                assert np.array_equal(d["ids"],ids[take]);rms=all_rms[take]
                L,F,Z,G=(d[k] for k in ("L","F","Z","group")); n+=len(rms)
                for i,mid in enumerate(d["ids"].astype(str)):
                    if not np.isfinite(rms[i]).any():continue
                    j=int(np.nanargmin(rms[i]));z=Z[i,j];m=z>0;g=int(G[i,j])
                    obs=group_check(L[i,j],F[i,j,m].astype(float),z[m],g,np.zeros(3),.01)
                    rows.append(dict(id=str(mid),best_draw_zero_based=j,rms=float(rms[i,j]),requested=g,
                        reference=int(truth[mid]),**obs))
        assert n==len(ids)
        out[tag]=dict(N=n,matched=len(rows),reference_known_among_matches=sum(r["reference"]>0 for r in rows),
            recovered_with_correct_detected_group=sum(r["reference"]>0 and r["detected"]==r["reference"] for r in rows),
            recovered_with_correct_requested_group=sum(r["requested"]==r["reference"] for r in rows),
            contains_requested=sum(r["contains"] for r in rows),records=rows)
    file=OUT/f"matched-group-{a.panel}.json";prior=json.loads(file.read_text()) if file.exists() else {};prior.update(out)
    file.write_text(json.dumps(prior,separators=(",",":"))+"\n",encoding="utf-8");print(a.panel,a.tags)
if __name__=="__main__":main()
