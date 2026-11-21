"""Exercise the pinned author's complete CSPFlow.sample against the portable endpoint."""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from crystalflow import HERE, ROOT, EncoderCache, load_model, sample
from joint_timing import propose


def author_adapter(source, model, embed, polar, joint):
    path=source/"diffcsp/pl_modules/flow.py"
    node,=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=="CSPFlow"]
    node.bases=[]
    methods={"sample","sample_lattice_polar","post_decoder_on_sample","get_anneal_factor"}
    node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in methods]
    assert len(node.body)==len(methods)
    env=dict(torch=torch,F=F,tqdm=lambda x:x,lattice_polar_build_torch=polar)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),env)
    adapter=env["CSPFlow"]()
    for name,val in dict(device=next(model.parameters()).device,guide_threshold=None,lattice_polar=True,from_cubic=False,
                         lattice_polar_sigma=.1,symmetrize_anchor=False,symmetrize_rotavg=False,
                         pred_type=joint,keep_coords=False,keep_lattice=False,decoder=model,
                         time_embedding=embed,type_encoding=model.type_encoding).items():
        setattr(adapter,name,val)
    return adapter


@torch.inference_mode()
def main(a):
    torch.set_num_threads(1)
    model,embed,polar=load_model(a.source,a.weights,"cpu")
    joint=model.pred_type
    adapter=author_adapter(a.source,model,embed,polar,joint)
    z=np.zeros((2,20),np.int16)
    z[0,:2]=[3,8];z[1,:4]=[8,8,14,14]
    n=(z>0).sum(1)
    batch=SimpleNamespace(num_graphs=len(z),num_nodes=int(n.sum()),num_atoms=torch.tensor(n),
                          atom_types=torch.tensor(z[z>0],dtype=torch.long),batch=torch.tensor(np.repeat(np.arange(len(z)),n)))
    tests=((5,0.),(100,0.)) if joint else ((5,0.),(100,5.))
    records=[]
    for steps,slope in tests:
        # Rebuild an untouched decoder for each original trajectory.
        model,embed,polar=load_model(a.source,a.weights,"cpu")
        adapter=author_adapter(a.source,model,embed,polar,joint)
        torch.manual_seed(21122)
        source,_=adapter.sample(batch,N=steps,anneal_coords=(slope!=0),anneal_slope=slope)
        native=source["lattices"].double().numpy()
        coords=source["frac_coords"].numpy()
        zs=source["atom_types"].numpy()
        cache=EncoderCache(model)
        torch.manual_seed(21122)
        if joint:
            out_z,k,f,L,_,valid=propose((model,embed,polar,cache),n,steps)
            assert valid.all()
        else:
            k,f=sample(model,embed,polar,cache,z,steps,"cpu",None,True,slope)
            from timing import decode
            out_z,L,_,_=decode(z,k,f)
        first=0;errors=[]
        for i,count in enumerate(n):
            group=slice(first,first+count);order=np.argsort(zs[group],kind="stable")
            np.testing.assert_equal(zs[group][order],out_z[i,:count])
            d=coords[group][order]-f[i,:count]
            errors.append(float(abs(d-np.rint(d)).max()))
            first+=count
        cell_error=float(abs(native-L).max())
        assert max(errors)<1e-5 and cell_error<5e-4
        records.append(dict(steps=steps,coord_slope=slope,cases=2,native_cell_max_abs_angstrom=cell_error,
                            max_fractional_torus_error=max(errors),species_identical=True))
    label="joint" if joint else "conditional"
    (HERE/f"{label}-source-sampler-parity.json").write_text(json.dumps(records,indent=2)+"\n")
    print(json.dumps(records,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",type=Path,required=True)
    p.add_argument("--weights",type=Path,required=True);main(p.parse_args())
