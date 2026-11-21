"""Join measured recovery to the actual paired shortlist, including off-template matches."""
import argparse
import json
from pathlib import Path
import numpy as np
from ..evaluation.storage import pool

HERE=Path(__file__).resolve().parent;OUT=HERE/"audits";E=HERE.parent/"evaluation"

def main():
    p=argparse.ArgumentParser();p.add_argument("panel");p.add_argument("tags",nargs="+")
    a=p.parse_args()
    with np.load(OUT/f"reachable-{a.panel}.npz") as d:
        ids=d["ids"];known=d["truth_available"];eq=d["equivalent"];family=d["family"];fixed=d["fixed"]
    comp={}
    for tag in a.tags:
        settings=json.loads((pool(a.panel,tag)/"settings.json").read_text())
        stage=json.loads(settings["signature"])
        if stage["family"]!="symbolic" or stage["selector"]!="lookup":
            raise ValueError(f"{tag} uses a different first-stage proposal stream")
        f=E/f"scores/{a.panel}/{tag}/outcomes.npz"
        with np.load(f) as d:
            assert np.array_equal(ids,d["ids"]);hit=np.isfinite(d["rms"])
        comp[tag]={}
        for k in (1,5,20):
            if k>hit.shape[1]:continue
            h=hit[:,:k];rr=eq[:,:k]
            coverage=rr.any(1);s=h.any(1)
            v={"targets":len(h),"recovered":int(s.sum()),"unknown_label":int((~known).sum()),
               "compatible_reachable":int(coverage.sum()),
               "recovered_among_reachable_targets":int((coverage&s).sum()),
               "target_with_match_on_compatible_draw":int((h&rr).any(1).sum()),
               "recovered_without_compatible_prior_draw":int((~coverage&known&s).sum()),
               "known_lacking_compatible_prior_draw":int((~coverage&known).sum()),
               "targets_with_any_coarse_family":int(family[:,:k].any(1).sum()),
               "successful_fixed_chart_draws":int((h&fixed[:,:k]).sum()),
               "fixed_chart_draws":int(fixed[:,:k].sum())}
            assert v["recovered_among_reachable_targets"]+v["recovered_without_compatible_prior_draw"]+int((~known&s).sum())==s.sum()
            comp[tag][k]=v
    dest=OUT/f"conditional-{a.panel}.json"
    allarms=json.loads(dest.read_text()) if dest.exists() else {}
    allarms.update(comp);dest.write_text(json.dumps(allarms,indent=2)+"\n",encoding="utf-8")
    print(a.panel,a.tags)

if __name__=="__main__":main()
