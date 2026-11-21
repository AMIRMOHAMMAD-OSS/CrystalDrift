"""Replay the decorated eight-site fixed-basis collision through the old and present code."""
import json
import os
from pathlib import Path
import sys
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[k]="1"
import numpy as np
import torch
from .descriptors import blocks,feature_blocks

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"revision/ablation"))
from run import fingerprint
from scaled.family import features as earlier_multielement


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    k=torch.zeros((2,6));l=torch.eye(3).repeat(2,1,1)
    z=torch.tensor([[8]*4+[14]*4]*2);f=torch.zeros((2,8,3))
    for j,indices in enumerate(((0,1,4,6),(0,1,3,7))):
        f[j,:,0]=torch.tensor(indices).repeat(2)/12.
    f[:,4:,1]=1/3
    o=torch.arange(8).repeat(2,1)
    dif=lambda x:float((x[0]-x[1]).abs().max())
    earlier={}
    for label in ("pair","original","harmonic"):
        feat=fingerprint(k,f,z,label);earlier[label]=dict(dimension=feat.shape[1],difference=dif(feat))
    pad=torch.nn.functional.pad
    feat=earlier_multielement(pad(z,(0,12)),k,pad(f,(0,0,0,12)))
    earlier["multielement"]=dict(dimension=feat.shape[1],difference=dif(feat))
    new={}
    with torch.inference_mode():
        for arm in ("angular","contacts","reciprocal"):
            feat=feature_blocks(l,f,z,o,arm)
            for name,sl in blocks().items():
                x=feat["base"][:,sl];new[name]=float((x[0]-x[1]).norm())
            if arm in feat:new[arm]=dif(feat[arm])
    expected=1/(16*np.sqrt(2))
    assert earlier["original"]["difference"]<1e-14 and earlier["multielement"]["difference"]<1e-14
    assert abs(earlier["harmonic"]["difference"]-expected)<2e-14
    assert new["radial"]<1e-10 and min(new["power0"],new["power1"],new["power2"])>1e-4
    assert new["reciprocal"]<1e-12
    out=dict(dtype="float64",cell="1 angstrom cubic",frac_rows="motif/12 along x; same translated row at y=1/3",
        species=[8]*4+[14]*4,earlier=earlier,current_block_l2_or_added_block_linf=new,
        analytic_separating_difference=float(expected),reciprocal_vectors_in_current_fixture=0)
    (HERE/"decorated-witness.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf8")
    print(out,flush=True)

if __name__=="__main__":main()
