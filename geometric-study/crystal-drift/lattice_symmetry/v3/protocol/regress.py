"""Re-evaluate archived case outcomes and run the original physical-geometry test."""
import json
import sys
from pathlib import Path
import numpy as np
V3=Path(__file__).resolve().parents[1]; V2=V3.parent/"v2"; ROOT=V3.parents[1]
sys.path.insert(0,str(ROOT))
from lattice_symmetry.v2.acceptance import regress
from lattice_symmetry.v2.csp.model import load_service,new_model
import torch

def main():
    torch.set_num_threads(1)
    d={"physics":regress(),"parameters":{}}
    for arm in ("angular","flow16"):
        model=new_model(arm)
        d["parameters"][arm]=sum(p.numel() for p in model.parameters())
    # A second entry point restores the selected weights and generates a finite native cell.
    service=load_service("angular",11)
    with np.load(ROOT/"scaled/mp20-data/val.npz") as a:
        k,f=service(a["z"][:3],np.random.default_rng(101))
    assert np.isfinite(k).all() and np.isfinite(f).all()
    d["sampler_shape"]=[list(k.shape),list(f.shape)]
    old=V2/"diagnostics_template_prediction/confirmation"
    with np.load(old/"scores.npz") as data:
        d["historic_384"]={m:int(np.isfinite(data[m][:,:5]).any(1).sum()) for m in ("symbolic_prior","symbolic_angular","uflow")}
    assert d["historic_384"]=={"symbolic_prior":210,"symbolic_angular":221,"uflow":254}
    with np.load(old/"predictions.npz") as s:
        np.savez_compressed(V3/"protocol/historic-replay.npz",ids=s["ids"][:12],L=s["L"][:12,:,1],F=s["F"][:12,:,1],Z=s["Z"][:12,:,1])
    (V3/"protocol/regression.json").write_text(json.dumps(d,indent=2)+"\n")
    print(d["parameters"],d["historic_384"],d["physics"]["independent_batch_errors"])

if __name__=="__main__":main()
