"""Replay the pinned 100-token Cartesian uFlowCSP network, with resumable GPU tiles."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
STUDY=HERE.parents[2]
AUTHORS=STUDY/"uflowcsp-results"
COMMIT="e87af8483e6d45b0217b4ede430862e992d2b1d0"
WEIGHT_SHA="e1854363d9e78044d1202879e40a918ab50831cb1e08eb686cb53aeacaa85673"
MODULES=("src/models/denoisers/atomic_features.py","src/models/denoisers/dit.py",
         "src/models/denoisers/meanflow_sit_raw_formula_atomwise_system.py","src/models/meanflow_transport.py")


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def official_sources(dst):
    meta=json.loads((AUTHORS/"cartesian.json").read_text())
    assert meta["sha"]==COMMIT
    shas={x["path"]:x["sha"] for x in meta["tree"]}
    for rel in MODULES:
        raw=(AUTHORS/("source-"+rel.replace("/","_")+".txt")).read_bytes()
        assert hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()==shas[rel]
        dest=dst/rel;dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(raw)
    sys.path.insert(0,str(dst))


def safe_weights(path):
    # Load only trusted F32 storage from verified safetensors; no pickle parsing.
    assert digest(path)==WEIGHT_SHA
    with path.open("rb") as f:
        size=int.from_bytes(f.read(8),"little")
        spec=json.loads(f.read(size))
    mapped=np.memmap(path,dtype=np.uint8,mode="c")
    state={}
    for name,item in spec.items():
        if name=="__metadata__":continue
        assert item["dtype"]=="F32"
        a,b=item["data_offsets"];shape=tuple(item["shape"])
        assert 0<=a<=b<=len(mapped)-8-size and b-a==4*int(np.prod(shape))
        raw=np.frombuffer(mapped,dtype="<f4",count=(b-a)//4,offset=8+size+a).reshape(shape)
        state[name]=torch.from_numpy(raw)
    assert len(state)==141
    return state


def structures(output,types,expected):
    width=expected.shape[1]
    active=(types[:,:width]!=0).unsqueeze(-1)
    cell=(output[:,:width,3:].masked_fill(~active,0).sum(1)/active.sum(1)).reshape(-1,3,3)
    frac=torch.remainder(output[:,:width,:3]@torch.linalg.pinv(cell),1)
    a=types[:,:width].cpu().numpy()
    back=np.argsort(np.where(a>0,a,120),kind="stable",axis=1)
    np.testing.assert_array_equal(np.take_along_axis(a,back,axis=1),expected)
    f=np.take_along_axis(frac.masked_fill(~active,0).cpu().numpy(),back[:,:,None],axis=1)
    L=cell.double().cpu().numpy();G=L@L.swapaxes(-1,-2)
    w,q=np.linalg.eigh(G);assert (w>0).all()
    S=(q*(np.log(w)/2)[:,None,:])@q.swapaxes(-1,-2)
    k=np.column_stack((S[:,0,0],S[:,1,1],S[:,2,2],np.sqrt(2)*S[:,0,1],np.sqrt(2)*S[:,0,2],np.sqrt(2)*S[:,1,2]))
    return k,f


def main(args):
    sys.path.extend(map(str,args.site))
    import timm
    assert timm.__version__=="1.0.21" and torch.cuda.is_available()
    from src.models.denoisers.meanflow_sit_raw_formula_atomwise_system import MeanFlowSiTRawFormulaAtomwiseSystem,build_sg_to_crystal_system
    from src.models.meanflow_transport import MeanFlowTransport
    torch.set_num_threads(2);torch.manual_seed(2026)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    with torch.device("meta"):
        model=MeanFlowSiTRawFormulaAtomwiseSystem(input_size=100,atom_type_vocab_size=120,hidden_size=768,
                depth=12,num_heads=12,predict_atom_types=False,use_spacegroup_conditioning=False,
                use_formula_embedding=True,use_atomwise_features=True,use_crystal_system_embedding=True)
    model.load_state_dict(safe_weights(args.weights),strict=True,assign=True)
    with np.load(HERE/"chemistry.npz") as tab:
        model.atomic_feature_table=torch.from_numpy(tab["table"].copy())
        key=tab["sort_key"]
    model.sg_to_crystal_system=build_sg_to_crystal_system(230);model=model.to("cuda").eval()
    transport=MeanFlowTransport(torus_coords=False,lattice_per_structure=False,proj_lattice=False,
                                proj_coords_anchor=False,proj_coords_rotavg=False)
    ref=STUDY/"scaled/mp20-data/test.npz"
    with np.load(ref) as f:
        raw_ids=f["ids"]
        ix=sorted(range(len(raw_ids)),key=lambda i:hashlib.sha256(str(raw_ids[i]).encode()).digest())
        ids,z=raw_ids[ix],f["z"][ix]
    assert len(ids)==9046 and set(ids)==set(json.loads((AUTHORS/"test-ids.json").read_text()))
    assert not set(ids)&set(json.loads((AUTHORS/"train-ids.json").read_text()))
    with np.load(STUDY/"scaled/external/mp20/uflow-seed2026-test.npz") as p:
        np.testing.assert_array_equal(p["ids"],ids[:128]);np.testing.assert_array_equal(p["z"],z[:128])
        oldk,oldf=p["k"],p["f"]
    tokens=np.zeros((len(z),100),dtype=np.int64)
    for i,row in enumerate(z):
        active=row[row>0];tokens[i,:len(active)]=sorted(active,key=lambda number:(key[number],number))
    directory=HERE/"mp20"
    directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/"targets.npz",ids=ids,z=z)
    meta=dict(seed=2026,torch=torch.__version__,timm=timm.__version__,gpu=torch.cuda.get_device_name(),
              torch_fp32_precision=torch.get_float32_matmul_precision(),checkpoint_sha256=WEIGHT_SHA,
              source_commit=COMMIT,padded_tokens=100,model_steps=1,batch_size=args.batch,
              test_sha256=digest(ref),model_parameters=sum(x.numel() for x in model.parameters()))
    with torch.inference_mode():
        def infer(index,draw,seed_id,probe=False):
            batch=len(index);types=torch.as_tensor(tokens[index],device="cuda")
            torch.manual_seed(2026000000+1000000*draw+seed_id)
            noise=transport.sample_prior(batch,100,12,"cpu").to("cuda")
            def run(x,nt):
                a=types[:,:nt]
                c=dict(atom_types=a,mask=a!=0,spacegroup=torch.zeros(batch,dtype=torch.long,device="cuda"),
                       generate_atom_types=False)
                return transport.sample(model,x[:,:nt],c,num_steps=1)
            output=run(noise,100)
            if probe:
                live=int(np.count_nonzero(z[index],axis=1).max())
                short=run(noise,live)
                diff=(short-output[:,:live]).abs()[types[:,:live]!=0]
                meta["pruned_same_noise_probe"]=dict(cases=len(index),columns=live,max_abs=float(diff.max()),
                                                     mean_abs=float(diff.mean()))
            return structures(output,types,z[index])
        infer(np.arange(8),0,9000,True)
        (directory/"manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
        schedule=[(i,i+1,0,128) for i in range(5)]+[(i,i+1,128,len(ids)) for i in range(5)]
        schedule+=[(i,i+1,0,128) for i in range(5,20)]
        schedule+=[(i,i+1,128,len(ids)) for i in range(5,20)]
        for draw,_,start,end in schedule:
            for lo in range(start,end,256):
                hi=min(end,lo+256)
                name=directory/f"draw{draw:02}-cases{lo:05}-{hi:05}.npz"
                if name.exists():continue
                tic=time.monotonic()
                if draw<5 and hi<=128:
                    k=oldk[lo:hi,draw];f=oldf[lo:hi,draw]
                else:
                    parts=[infer(np.arange(i,min(i+args.batch,hi)),draw,i) for i in range(lo,hi,args.batch)]
                    k=np.concatenate([v[0] for v in parts]);f=np.concatenate([v[1] for v in parts])
                assert np.isfinite(k).all() and np.isfinite(f).all()
                temporary=name.with_suffix(".part.npz")
                np.savez_compressed(temporary,ids=ids[lo:hi],z=z[lo:hi],k=k,f=f,draw=draw,first=lo,
                                    seconds=time.monotonic()-tic)
                temporary.replace(name)
                if lo==start or hi==end:print("draw",draw+1,"rows",lo,hi,round(time.monotonic()-tic,2),"seconds",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights",type=Path,required=True);p.add_argument("--site",type=Path,nargs="*",default=[])
    p.add_argument("--batch",type=int,default=24)
    args=p.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        official_sources(Path(temp))
        main(args)
