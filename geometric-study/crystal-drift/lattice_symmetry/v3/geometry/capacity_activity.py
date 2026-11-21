"""Count nominal model parameters and analytically identifiable zero-data-gradient entries."""
import json
from pathlib import Path
import numpy as np
import torch
from ..models.network import OrbitNet,TIERS

def main():
    here=Path(__file__).resolve().parent;v3=here.parent;root=v3.parents[1]
    torch.set_num_threads(1);ids=json.loads((v3/"models/cache/inventory.json").read_text())["eligible"]
    with np.load(root/"scaled/mp20-data/train.npz") as data:
        z=data["z"][np.isin(data["ids"],ids)];types=np.unique(z[z>0])
    missing=np.setdiff1d(np.arange(119),types)
    rows={}
    for tier,(width,depth) in TIERS.items():
        with torch.device("meta"):
            net=OrbitNet(width,depth)
        inactive=len(missing)*(net.element.embedding_dim+net.chemistry.embedding_dim)
        rows[tier]=dict(width=width,depth=depth,all_parameters=sum(p.numel() for p in net.parameters()),
            nominal_trainable=sum(p.numel() for p in net.parameters() if p.requires_grad),
            frozen=sum(p.numel() for p in net.parameters() if not p.requires_grad),
            unused_corpus_embedding_entries=inactive,also_inactive_drift_zero_r_column=width,
            distinct_training_only_parameters=0)
    torch.manual_seed(982353);net=OrbitNet(*TIERS["small"])
    with np.load(v3/"models/cache/train-000.npz") as d:
        batch=[torch.as_tensor(d[f"x{i}"][:7],dtype=torch.long if i in (2,3) else torch.float32) for i in range(8)]
    checks={}
    for label,times in (("drift_empty_times",{}),("meanflow_nonzero_times",dict(r=torch.ones(7)*.25,t=torch.ones(7)*.8))):
        net.zero_grad();L,F=net(*batch,**times);(L.square().mean()+F.square().mean()).backward()
        embed=max(float(net.element.weight.grad[missing].abs().max()),float(net.chemistry.weight.grad[missing].abs().max()))
        col=float(net.encoder.weight.grad[:,-2].abs().max());assert embed==0
        assert (col==0)==(label=="drift_empty_times")
        checks[label]=dict(inactive_embedding_max_grad=embed,r_column_max_grad=col)
    out=dict(eligible_rows=len(ids),present_atomic_numbers=types.tolist(),absent_including_padding=missing.tolist(),
             tiers=rows,checks=checks,
             meaning="Entries found analytically to receive no data gradient on the retained corpus. This does not assert that all other parameters are identifiable or have nonzero gradient at each update. AdamW can decay entries in an active tensor.")
    (here/"capacity-activity.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(checks)
if __name__=="__main__":main()
