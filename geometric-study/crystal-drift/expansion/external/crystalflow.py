"""Replay the released CrystalFlow CSPNet. Run from the crystal-drift project."""
import argparse,ast,hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"scaled"))
from prepare_mp20 import pack_gram

def scatter(source,index,dim=0,reduce="mean",dim_size=None):
    assert dim==0 and reduce=="mean"
    n=dim_size if dim_size is not None else int(index[-1])+1
    out=source.new_zeros((n,source.shape[1])).index_add_(0,index,source)
    count=source.new_zeros(n).index_add_(0,index,source.new_ones(len(source))).clamp_min(1)
    return out/count[:,None]

def definitions(path, names, namespace):
    tree=ast.parse(path.read_text(encoding="utf8"))
    selected=[v for v in tree.body if isinstance(v,(ast.ClassDef,ast.FunctionDef)) and v.name in names]
    assert len(selected)==len(names)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),"exec"),namespace)

def load_model(source,weights,device):
    context=dict(torch=torch,nn=nn,F=F,math=math,MAX_ATOMIC_NUM=100,scatter=scatter,
                 dense_to_sparse=lambda a:(a.nonzero().T,None))
    definitions(source/"diffcsp/common/data_utils.py",{"lattice_polar_build_torch","get_reciprocal_lattice_torch"},context)
    definitions(source/"diffcsp/pl_modules/cspnet.py",{"SinusoidsEmbedding","RecSinusoidsEmbedding","PeriodicNorm","CSPLayer","CSPNet"},context)
    definitions(source/"diffcsp/pl_modules/flow.py",{"SinusoidalTimeEmbeddings"},context)
    hp=json.loads((weights/"model.json").read_text())
    args=dict(hp["decoder"]);args.pop("_target_");args["latent_dim"]=hp["time_dim"]+hp["latent_dim"]
    assert hp["latent_dim"]==0 and hp["lattice_polar"] and hp["lattice_polar_sigma"]==.1
    if hp.get("type_encoding")=="table":
        definitions(source/"diffcsp/pl_modules/type_module.py",{"TypeTableModule"},context)
        args["type_encoding"]=context["TypeTableModule"]()
    model=context["CSPNet"](**args).eval().to(device)
    with np.load(weights/"weights.npz") as z:
        model.load_state_dict({n:torch.from_numpy(z[n]) for n in z.files},strict=True)
    return model,context["SinusoidalTimeEmbeddings"](hp["time_dim"]).to(device),context["lattice_polar_build_torch"]


class EncoderCache:
    # The same embedding modules are shared by all six message-passing layers.
    # Their inputs are bit-identical within each decoder evaluation.
    def __init__(self,net):
        self.cache={}
        for name in ("rec_emb","dis_emb"):
            module=getattr(net,name);original=module.forward
            def forward(*args,key=name,fn=original):
                if key not in self.cache:self.cache[key]=fn(*args)
                return self.cache[key]
            module.forward=forward


def fast_edges(counts,device):
    start=np.r_[0,np.cumsum(counts)[:-1]]
    lists=[np.arange(first,first+n) for first,n in zip(start,counts)]
    return torch.from_numpy(np.stack([np.concatenate([np.repeat(l,len(l)) for l in lists]),
        np.concatenate([np.tile(l,len(l)) for l in lists])])).to(device)

@torch.inference_mode()
def sample(model,time_embed,decode,cache,z,steps,device,seed,check=False,coord_slope=0.):
    if seed is not None:torch.manual_seed(seed)
    counts=(z>0).sum(1); n=len(z); mask=z>0
    num=torch.from_numpy(counts).to(device)
    graph=torch.from_numpy(np.repeat(np.arange(n),counts)).to(device)
    atoms=torch.from_numpy(z[mask].astype(np.int64)).to(device)
    initial=torch.randn(n,6,device=device)*.1; initial[:,-1]+=1.
    coords=torch.rand(int(mask.sum()),3).to(device)
    l=initial
    edges=fast_edges(counts,device)
    original_edges=model.gen_edges
    model.gen_edges=lambda num,coords,lattices,g:(edges,(coords[edges[1]]-coords[edges[0]])%1.)
    for t in range(1,steps+1):
        times=time_embed(torch.full((n,),t/steps,device=device))
        cache.cache.clear()
        cell=decode(l)
        dl,df=model(times,atoms,coords,l,num,graph,lattices_mat=cell)
        if check and t==1:
            ref_edges,ref_delta=original_edges(num,coords,cell,graph)
            assert torch.equal(ref_edges,edges)
            assert torch.equal(ref_delta,(coords[edges[1]]-coords[edges[0]])%1.)
        if coord_slope:
            factor=1.+coord_slope*torch.relu(torch.tensor(t/steps))
            df=df*factor
        l=l+dl/steps;coords=(coords+df/steps)%1
    model.gen_edges=original_edges
    # For a symmetric exp(S) cell, the standard log Gram chart equals S.
    q=l.double().cpu().numpy()
    k=np.column_stack([q[:,3]+q[:,4]+q[:,5],-q[:,3]+q[:,4]+q[:,5],-2*q[:,4]+q[:,5],np.sqrt(2)*q[:,:3]])
    f=np.zeros((n,z.shape[1],3),np.float32); f[mask]=coords.cpu().numpy()
    if check:
        raw=decode(l).double().cpu().numpy()
        np.testing.assert_allclose(k,pack_gram(raw@raw.swapaxes(-1,-2)),atol=2e-4)
    return k,f

def main(opts):
    # The documented 100-step CSP command anneals coordinates with slope five.
    slope=(5. if opts.steps==100 else 0.) if opts.coord_slope is None else opts.coord_slope
    name=f"csp-s{opts.steps}"
    if slope!=(5. if opts.steps==100 else 0.):name+=f"-coord{slope:g}"
    out=HERE/name;out.mkdir(exist_ok=True)
    with np.load(ROOT/"revision/full/mp20/cohort.npz") as a: z=a["z"];ids=a["ids"]
    torch.set_num_threads(1);device=opts.device
    if device=="cuda":
        torch.backends.cuda.matmul.allow_tf32=opts.tf32
        torch.backends.cudnn.allow_tf32=opts.tf32
    model,emb,decode=load_model(opts.source,opts.weights,device)
    cache=EncoderCache(model)
    before=time.perf_counter()
    k,f=sample(model,emb,decode,cache,z[:opts.batch],opts.steps,device,847,True,slope)
    warm=time.perf_counter()-before
    meta=dict(parameters=sum(x.numel() for x in model.parameters()),torch=torch.__version__,
              device=torch.cuda.get_device_name() if device=="cuda" else "cpu",
              source_commit="9c25ff0245d787efd87c9d1d797a8968443608cc",steps=opts.steps,batch=opts.batch,
              tf32=opts.tf32,warm_batch_seconds=warm,
              coord_slope=slope,
              time_stamps="t=j/S for j=1..S; "+("no coordinate/lattice annealing" if slope==0 else f"site velocity multiplied by 1+{slope:g}t; no lattice annealing")+"; no symmetry or property conditioning")
    print(meta,flush=True)
    if opts.pilot:return
    settings=out/"settings.json"
    if settings.exists():
        previous=json.loads(settings.read_text())
        for key in ("source_commit","parameters","torch","device","steps","batch","tf32","time_stamps"):
            if previous[key]!=meta[key]:raise ValueError(f"Existing pool uses different {key}: {previous[key]}")
    else:
        settings.write_text(json.dumps(meta,indent=2)+"\n")
    if not 1<=opts.targets<=len(ids) or not 1<=opts.draws<=20:raise ValueError("Target count or draw count outside cohort")
    groups=[(0,min(128,opts.targets))]+[(i,min(i+256,opts.targets)) for i in range(128,opts.targets,256)]
    jobs=[(j,*groups[0]) for j in range(opts.draws)]
    jobs.extend((j,a,b) for j in range(opts.draws) for a,b in groups[1:])
    for draw,start,end in jobs:
        path=out/f"draw{draw:02}-cases{start:05}-{end:05}.npz"
        if path.exists():continue
        if (HERE/"gpu-hold").is_file():
            torch.cuda.synchronize() if device=="cuda" else None
            marker=HERE/"gpu-idle";marker.touch()
            while (HERE/"gpu-hold").is_file():time.sleep(2)
            marker.unlink(missing_ok=True)
        tic=time.perf_counter();parts=[]
        for i in range(start,end,opts.batch):
            parts.append(sample(model,emb,decode,cache,z[i:min(i+opts.batch,end)],opts.steps,device,2062100+draw*10000+i,coord_slope=slope))
        k,f=[np.concatenate([v[j] for v in parts]) for j in range(2)]
        elapsed=time.perf_counter()-tic;temp=path.with_suffix(".part.npz")
        np.savez_compressed(temp,ids=ids[start:end],z=z[start:end],k=k,f=f,draw=draw,first=start,seconds=elapsed)
        temp.replace(path)
        print(draw,start,end,round(elapsed,3),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True);p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda");p.add_argument("--steps",type=int,default=5)
    p.add_argument("--batch",type=int,default=24);p.add_argument("--draws",type=int,default=20)
    p.add_argument("--coord-slope",type=float,help="Site-velocity multiplier slope (default 5 for S=100, zero otherwise)")
    p.add_argument("--targets",type=int,default=9046);p.add_argument("--pilot",action="store_true")
    p.add_argument("--tf32",action="store_true");main(p.parse_args())
