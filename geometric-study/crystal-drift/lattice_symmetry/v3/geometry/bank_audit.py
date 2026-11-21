"""Train-only activity, bandwidth and anchored-bank counterexample for the registered runs."""
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from .descriptors import bank_direction,feature_blocks
from ..models.network import pack
from ..models.sampler import Sampler

HERE=Path(__file__).resolve().parent;CACHE=HERE.parent/"models/cache"


def main():
    torch.set_num_threads(1)
    q=torch.zeros(1,2,1,dtype=torch.float64)
    examples={}
    for law in ("laplace","gaussian"):
        values=[]
        for anchor in (-1,1):
            field=[]
            for other,n1,n2 in itertools.product((-1,1),repeat=3):
                p=q.new_tensor([[[anchor],[other]]]);n=q.new_tensor([[[n1],[n2]]])
                field.append(float(bank_direction(q,p,n,law)[0,0,0]))
            values.append(float(np.mean(field)))
        assert abs(np.mean(values))<1e-12 and max(values)>.1
        examples[law]=dict(conditional_on_fixed_minus=values[0],conditional_on_fixed_plus=values[1],
                           uniform_random_anchor_mean=float(np.mean(values)))
    s=Sampler()
    with np.load(CACHE/"calibration.npz") as d:
        ids=d["ids"].astype(str)
        scales={k:d[k+"_scale"] for k in ("base","variance","contacts","reciprocal")}
        temps=d["temperatures"];dist=d["distances"]
    anchors=[s.target(s.mid[x]) for x in ids]
    signals={k:[] for k in scales}
    values={k:[] for k in scales}
    for off in range(0,len(ids),16):
        L,F,Z,O,*_=pack(anchors[off:off+16],dtype=torch.float64)
        for k in scales:
            arm=k if k!="base" else "angular"
            x=feature_blocks(L,F,Z,O,arm)[k]; y=feature_blocks(1.05*L,F,Z,O,arm)[k]
            delta=(x-y)/torch.from_numpy(scales[k])
            signals[k].extend(delta.norm(dim=1).tolist());values[k].extend(x.norm(dim=1).tolist())
    out=dict(conditional_bank_example=examples,calibration_ids=ids.tolist(),
             qualified_train_rows=len(s.rows),bank_sizes=dict(sorted((str(n),sum(len(x)==n for x in s.by_formula.values())) for n in set(map(len,s.by_formula.values())))),
             singleton_anchored_rows=sum(len(x)==1 for x in s.by_formula.values()),formulas=len(s.by_formula),
             bandwidth_distances_angular_seed_to_data_quantiles=np.quantile(dist,[0,.25,.5,.75,1]).tolist(),
             shared_temperatures=temps.tolist(),activity={})
    for k in signals:
        v=np.array(values[k]);g=np.array(signals[k])
        out["activity"][k]=dict(nonzero_raw_l2=int(np.sum(v>1e-10)),
            response_to_five_percent_isotropic_dilation=int(np.sum(g>1e-10)),
            standardized_change_l2_quantiles=np.quantile(g,[0,.25,.5,.75,1]).tolist())
    (HERE/"train-bank-audit.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print("Audited",len(ids),"calibration rows")


if __name__=="__main__":main()
