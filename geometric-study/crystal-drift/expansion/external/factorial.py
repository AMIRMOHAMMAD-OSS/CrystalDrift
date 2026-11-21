"""Separate Euler resolution and coordinate annealing on paired MP-20 pilot draws."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ARMS={"5/0":"csp-s5","5/5":"csp-s5-coord5","100/0":"csp-s100-coord0","100/5":"csp-s100"}


def main():
    records={}
    outcomes={}
    identifiers=None
    for name,folder in ARMS.items():
        settings=json.loads((HERE/folder/"settings.json").read_text())
        steps,slope=map(int,name.split("/"))
        assert settings["steps"]==steps and settings.get("coord_slope",0)==slope and settings["tf32"]
        with np.load(HERE/folder/"score-crosswalk.npz") as data:
            assert data["scored"][:128,0].all()
            if identifiers is None:identifiers=data["ids"][:128]
            np.testing.assert_equal(identifiers,data["ids"][:128])
            rms=data["rms"][:128,0]
        hit=np.isfinite(rms)
        records[name]=dict(hits=int(hit.sum()),targets=len(hit),mean_matched_rms=float(rms[hit].mean()))
        outcomes[name]=rms
    rng=np.random.default_rng(9091);idx=rng.integers(0,128,(10000,128))
    contrasts={}
    for left,right in (("5/0","100/0"),("5/5","100/5"),("5/0","5/5"),("100/0","100/5")):
        x,y=outcomes[left],outcomes[right]
        a,b=np.isfinite(x),np.isfinite(y)
        delta=b.astype(float)-a
        common=a&b
        contrasts[f"{right} minus {left}"]=dict(mr_gap=float(delta.mean()),ci95=np.quantile(delta[idx].mean(1),[.025,.975]).tolist(),
            improved_hits=int((b&~a).sum()),lost_hits=int((a&~b).sum()),common_hits=int(common.sum()),
            common_hit_rms=[float(x[common].mean()),float(y[common].mean())])
    d=np.isfinite(outcomes["100/5"]).astype(float)-np.isfinite(outcomes["5/5"])-np.isfinite(outcomes["100/0"])+np.isfinite(outcomes["5/0"])
    out=dict(cells=records,contrasts=contrasts,additive_recovery_interaction=float(d.mean()),
             interaction_95ci=np.quantile(d[idx].mean(1),[.025,.975]).tolist())
    (HERE/"step-annealing-factorial.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))


if __name__=="__main__":main()
