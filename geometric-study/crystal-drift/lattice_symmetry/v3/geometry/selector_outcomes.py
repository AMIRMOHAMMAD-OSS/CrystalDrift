"""Join each selector's candidate-specific Boolean matches to its own ordered occupation list."""
import csv
from pathlib import Path
import numpy as np
from ..evaluation.analysis import ci

HERE=Path(__file__).resolve().parent;OUT=HERE/"audits";E=HERE.parent/"evaluation"
selectors={"prior":"lookup","empirical-rank":"frequency","learned-rank":"learned","learned-rank-angular":"learned"}

def main():
    rows=[]
    for panel in ("dev","test"):
        with np.load(OUT/f"ranking-coverage-{panel}.npz") as d:
            ids=d["ids"];rank=dict(zip(d["methods"],d["flags"][:,d["fields"].tolist().index("equivalent")]))
            known=d["truth_available"]
        truth={}
        for tag,method in selectors.items():
            file=E/f"scores/{panel}/{tag}/outcomes.npz"
            if not file.is_file():continue
            with np.load(file) as d:
                assert np.array_equal(ids,d["ids"]);h=np.isfinite(d["rms"])
                truth[tag]=h
            for k in (1,5,20):
                if h.shape[1]<k or truth["prior"].shape[1]<k:continue
                claims=rank[method][:,:k];candidate=h[:,:k]
                r=claims.any(1);s=candidate.any(1)
                baseline=truth["prior"][:,:k].any(1)
                delta=s.astype(float)-baseline.astype(float)
                lo,hi=ci(delta)
                rows.append(dict(panel=panel,tag=tag,order=method,k=k,n=len(ids),matched=int(s.sum()),
                    reachable=int(r.sum()),matched_among_reachable=int((r&s).sum()),
                    matched_on_same_draw=int((claims&candidate).any(1).sum()),known_unreachable=int((known&~r).sum()),
                    matched_among_known_unreachable=int((known&~r&s).sum()),matched_unknown=int((~known&s).sum()),
                    change_vs_lookup_prior_pp=100*delta.mean(),low=100*lo,high=100*hi))
    with (OUT/"selector-conditional.csv").open("w",newline="") as fp:
        out=csv.DictWriter(fp,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)
    print("Rows of end-to-end order and matcher diagnostics:",len(rows))

if __name__=="__main__":main()
