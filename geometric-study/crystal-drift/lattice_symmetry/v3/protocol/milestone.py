"""Write a date-stamped first-window report using only complete ordered target records."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
V3=HERE.parent;ROOT=V3.parents[1]
sys.path.insert(0,str(V3/"evaluation"))
from analysis import ci

def endpoint(path,ids,k):
    with np.load(path) as d:
        names=d["ids"].astype(str);lut={name:i for i,name in enumerate(names)}
        take=[lut[n] for n in ids];r=d["rms"][take,:k];v=d["log_volume_ratio"][take,:k]
    return np.isfinite(r).any(1),(np.isfinite(r)&(np.abs(v)<=np.log(1.1))).any(1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--save-as",default="FIRST-WINDOW.md");a=ap.parse_args()
    splits=json.loads((HERE/"splits.json").read_text());ids=splits["test_ids"]
    base=V3/"evaluation/scores/test";ph,pn=endpoint(base/"prior/outcomes.npz",ids,5)
    files={p.parent.name:p for p in base.glob("*/outcomes.npz")}
    files["released-uflow"]=ROOT/"revision/full/external/mp20/complete-crosswalk-k20.npz"
    rr=["# Overnight research milestone","",datetime.now().astimezone().isoformat(timespec="minutes"),"",
        "Each table entry covers all 9,046 official targets and preserves the order of its 20 proposals. Only complete target-level crosswalks enter this snapshot.","",
        "| Pipeline | MR@5, count | MR@20, count | Native@5, count | Five-draw gain over symbolic prior, percentage points (95% interval) |",
        "|---|---:|---:|---:|---:|"]
    for name,path in sorted(files.items()):
        h,n=endpoint(path,ids,5);h20,_=endpoint(path,ids,20)
        x=h.astype(float)-ph;l,r=100*np.array(ci(x))
        rr.append(f"| {name} | {h.sum():,} | {h20.sum():,} | {n.sum():,} | {100*x.mean():+.2f} ({l:+.2f}, {r:+.2f}) |")
    rr+=["","The primary descriptor screen trained 218,497-parameter angular, inter-orbit contact and reciprocal updates over the same 26,419 anchored primitive structures. They each recovered 63/96 development targets and 50/96 on independent confirmation at their two-look selections. The identical prior recovered 59 and 48. Both 2,098,297-parameter drift and same-backbone MeanFlow have completed a first traversal. The downsized native uFlowCSP models finished eight traversals of 26,614 structures, and separate one-traversal fits used the exact common 26,419 anchor IDs.",
        "","With 24 prototype families withheld, the first-traversal small contact map recovered 49/228 at five draws, against 46 for its identical prior. The paired family-bootstrap interval was 0.00 to 3.23 percentage points. None of the 49 best accepted structures had the target's detected group at 0.01 Å. The same update gained ten targets among 96 held-out chemistries in represented families. The 25 exploratory cases without a continuous anonymous analogue yielded one identical hit for every frozen arm at five draws.",
        "","The small contact model generated 8,883 structurally valid proposals and 4,359 jointly valid, unique and training-novel proposals from 10,000 empirical formulas. The identity prior yielded 8,107 and 4,140. In the paired eight-case CHGNet screen, the contact model yielded zero structures satisfying every restricted-hull gate. Its zero-success upper limit is 31.2% under independent Bernoulli sampling.",
        "","The served seed distribution and physical-volume behavior materially affect interpretation. A single neural evaluation counts work per proposal. The synchronized CPU/GPU clock, remaining replications and extended training budgets are recorded separately when complete. Training and scoring continue after this milestone."]
    dest=V3/"release"/a.save_as
    dest.write_text("\n".join(rr)+"\n",encoding="utf8");print(dest,len(files),flush=True)
if __name__=="__main__":main()
