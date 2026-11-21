"""Replay the published Cartesian uFlowCSP EMA on the two official matched cohorts.

Use --weights for the validated safetensors output of uflowcsp_weights.py.
The actual network blocks, prior and one-step sampler are the pinned authors'
implementations. Optional paths add isolated copies of timm and pymatgen.
"""
import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from data import pack_gram


HERE=Path(__file__).resolve().parent
RESULTS=HERE/"uflowcsp-results"
SOURCE_COMMIT="e87af8483e6d45b0217b4ede430862e992d2b1d0"
MODULES=("src/models/denoisers/atomic_features.py","src/models/denoisers/dit.py",
         "src/models/denoisers/meanflow_sit_raw_formula_atomwise_system.py","src/models/meanflow_transport.py")


def stage_sources(temp):
    tree=json.loads((RESULTS/"cartesian.json").read_text(encoding="utf8"))
    assert tree["sha"]==SOURCE_COMMIT
    expected={item["path"]:item["sha"] for item in tree["tree"]}
    for rel in MODULES:
        local=RESULTS/("source-"+rel.replace("/","_")+".txt")
        data=local.read_bytes()
        assert hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()==expected[rel]
        path=temp/rel
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
    sys.path.insert(0,str(temp))


def main(args):
    for loc in args.site:
        sys.path.append(str(loc))
    from pymatgen.core import Element
    from importlib.metadata import version
    import timm
    assert timm.__version__=="1.0.21"
    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)
    with tempfile.TemporaryDirectory() as temp:
        stage_sources(Path(temp))
        from src.models.denoisers.atomic_features import build_atomic_feature_table
        from src.models.denoisers.meanflow_sit_raw_formula_atomwise_system import MeanFlowSiTRawFormulaAtomwiseSystem,build_sg_to_crystal_system
        from src.models.meanflow_transport import MeanFlowTransport
        with torch.device("meta"):
            model=MeanFlowSiTRawFormulaAtomwiseSystem(input_size=100,atom_type_vocab_size=120,
                         hidden_size=768,depth=12,num_heads=12,predict_atom_types=False,
                         use_spacegroup_conditioning=False,use_formula_embedding=True,
                         use_atomwise_features=True,use_crystal_system_embedding=True)
        with safe_open(args.weights,framework="pt",device="cpu") as weights:
            state={k:weights.get_tensor(k) for k in weights.keys()}
        model.load_state_dict(state,strict=True,assign=True)
        model.atomic_feature_table=build_atomic_feature_table(120)
        model.sg_to_crystal_system=build_sg_to_crystal_system(230)
        assert model.atomic_feature_table.abs().sum()>0
        model.eval()
        transport=MeanFlowTransport(torus_coords=False,lattice_per_structure=False,proj_lattice=False,
                                    proj_coords_anchor=False,proj_coords_rotavg=False)
        train_ids=set(json.loads((RESULTS/"train-ids.json").read_text()))
        test_ids=set(json.loads((RESULTS/"test-ids.json").read_text()))
        prefix="multi-" if args.cohort=="multi" else ""
        with np.load(HERE/"data"/(prefix+"test.npz")) as ref:
            z=ref["z"];ids=ref["ids"]
        if args.cohort=="multi":
            from standard_match import order
            indices=order("multi",ids)
            z,ids=z[indices],ids[indices]
            assert len(ids)==128
        assert set(ids)<=test_ids and set(ids).isdisjoint(train_ids)
        def ordering(q):
            en=Element.from_Z(int(q)).X
            return ((float(en) if en is not None and math.isfinite(en) else 100+q*.001),q)
        atoms=np.zeros_like(z,dtype=np.int64)
        for i,row in enumerate(z):
            active=row[row>0]
            atoms[i,:len(active)]=sorted(active,key=ordering)
        with (RESULTS/(prefix+"crosswalk.csv")).open("w",encoding="utf8",newline="") as f:
            writer=csv.writer(f)
            if args.cohort=="ab":
                writer.writerow(["id","z1","z2","official_first_z","official_second_z","in_train_ids","in_test_ids"])
                for mid,a,b in zip(ids,z,atoms):
                    writer.writerow([mid,*a,*b,mid in train_ids,mid in test_ids])
            else:
                writer.writerow(["id","sorted_active_z","official_active_z","in_train_ids","in_test_ids"])
                for mid,a,b in zip(ids,z,atoms):
                    writer.writerow([mid," ".join(map(str,a[a>0]))," ".join(map(str,b[b>0])),mid in train_ids,mid in test_ids])
        repeated=np.repeat(atoms,args.draws,axis=0)
        all_k,all_f=[],[]
        tic=time.perf_counter()
        with torch.inference_mode():
            for i in range(0,len(repeated),args.batch):
                selected=repeated[i:i+args.batch]
                batch=len(selected)
                types=torch.zeros((batch,100),dtype=torch.long)
                width=selected.shape[1]
                types[:,:width]=torch.from_numpy(selected)
                valid=types!=0
                cond=dict(atom_types=types,mask=valid,spacegroup=torch.zeros(batch,dtype=torch.long),
                          generate_atom_types=False)
                noise=transport.sample_prior(batch,100,12,"cpu")
                output=transport.sample(model,noise,cond,num_steps=1)
                if args.cohort=="multi":
                    active=valid[:,:width,None]
                    lattice_tokens=output[:,:width,3:].masked_fill(~active,0)
                    cell=(lattice_tokens.sum(1)/active.sum(1)).reshape(-1,3,3)
                    frac=torch.remainder(output[:,:width,:3]@torch.linalg.pinv(cell),1)
                    # Restore the data's ascending-Z order, retaining all same-species sites.
                    restore=np.argsort(np.where(selected==0,120,selected),kind="stable",axis=1)
                    output_z=np.take_along_axis(selected,restore,axis=1)
                    expected=z[np.arange(i,i+batch)//args.draws]
                    np.testing.assert_array_equal(output_z,expected)
                    coordinates=np.take_along_axis(frac.masked_fill(~active,0).numpy(),restore[:,:,None],axis=1)
                else:
                    cell=output[:,:2,3:].mean(1).reshape(-1,3,3)
                    frac=torch.remainder(output[:,:2,:3]@torch.linalg.pinv(cell),1)
                    delta=frac[:,1]-frac[:,0]
                    reverse=torch.from_numpy(selected[:,1]<selected[:,0])
                    coordinates=torch.where(reverse[:,None],-delta,delta).remainder(1).numpy()
                C=cell.double().numpy()
                all_k.append(pack_gram(C@C.swapaxes(-1,-2)))
                all_f.append(coordinates)
                if i%(args.batch*10)==0:print(f"Decoded {i+batch}/{len(repeated)}",flush=True)
        elapsed=time.perf_counter()-tic
        k=np.concatenate(all_k).reshape(len(ids),args.draws,6)
        f=np.concatenate(all_f).reshape(len(ids),args.draws,*all_f[0].shape[1:])
        assert np.isfinite(k).all() and np.isfinite(f).all()
        np.savez_compressed(RESULTS/(prefix+"seed2026-euler1.npz"),k=k,f=f,z=z,ids=ids)
        report=dict(repository="https://github.com/usccolumbia/uFlowCSP",commit=SOURCE_COMMIT,
                    checkpoint="https://doi.org/10.6084/m9.figshare.33974422.v1",source_license="MIT",
                    checkpoint_license="CC BY 4.0",checkpoint_figshare_md5="c1ff449289976c793b06bdbfcc9659ad",
                    weights_validation="141 EMA tensor records checked against ZIP CRCs; safe tensor subset only",
                    parameters=sum(x.numel() for x in model.parameters()),torch=torch.__version__,timm=timm.__version__,
                    pymatgen=version("pymatgen"),condition="exact cell composition, no space group; ordered by Pauling electronegativity",
                    target_count=len(ids),draws=args.draws,steps=1,seed=2026,batch=args.batch,threads=args.threads,
                    total_seconds=elapsed,amortized_milliseconds_per_crystal=elapsed*1000/len(repeated),
                    input_representation="Cartesian, 100 padded tokens; reference chart converted after decoding",
                    target_ids_in_train=len(set(ids)&train_ids),target_ids_in_test=len(set(ids)&test_ids),
                    returned_candidates=len(repeated),postprocessing="unrelaxed; all candidates retained before CLI validity filtering")
        if args.cohort=="multi":
            site_counts,frequencies=np.unique((z>0).sum(1),return_counts=True)
            report.update(cohort="First 128 binary 2–8 site test IDs ordered by SHA256 of material ID, matching standard_match.order",
                          active_site_counts=dict(zip(site_counts.tolist(),frequencies.tolist())),
                          coordinates="All active Cartesian sites converted to fractional and restored to ascending-Z order, zero-padded to 8")
        (RESULTS/(prefix+"replay.json")).write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
        print("Finished",len(repeated),"candidates in",round(elapsed,1),"seconds",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights",type=Path,required=True)
    parser.add_argument("--site",type=Path,nargs="*",default=[])
    parser.add_argument("--batch",type=int,default=12)
    parser.add_argument("--threads",type=int,default=3)
    parser.add_argument("--draws",type=int,default=5)
    parser.add_argument("--cohort",choices=("ab","multi"),default="ab")
    main(parser.parse_args())
