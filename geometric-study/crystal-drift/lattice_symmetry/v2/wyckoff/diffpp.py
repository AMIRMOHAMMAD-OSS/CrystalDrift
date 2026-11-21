"""Use the pinned DiffCSP++ CSP decoder and original predictor/corrector sampler."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
SRC=HERE/"external"
with np.load(HERE/"charts.npz") as d:
    M,T,ORB,GR=[d[s] for s in ["matrices","t24","positions","groups"]]
LETTERS="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def scatter(source,index,dim=0,reduce="mean",dim_size=None):
    assert dim==0 and reduce=="mean"
    n=dim_size if dim_size is not None else int(index.max())+1
    acc=source.new_zeros((n,source.shape[1])).index_add_(0,index,source)
    cnt=source.new_zeros(n).index_add_(0,index,source.new_ones(len(source))).clamp_min_(1)
    return acc/cnt[:,None]

def run_defs(file,names,env):
    tree=ast.parse((SRC/file).read_text())
    items=[a for a in tree.body if isinstance(a,(ast.ClassDef,ast.FunctionDef)) and a.name in names]
    assert len(items)==len(names)
    exec(compile(ast.Module(body=items,type_ignores=[]),file,"exec"),env)

def construct(device):
    env=dict(torch=torch,nn=nn,math=math,np=np,linalg=torch.linalg,MAX_ATOMIC_NUM=100,
             dense_to_sparse=lambda a:(a.nonzero().T,None),scatter=scatter,tqdm=lambda x:x)
    run_defs("diffcsp/pl_modules/cspnet.py",{"SinusoidsEmbedding","CSPLayer","CSPNet"},env)
    run_defs("diffcsp/pl_modules/lattice/matrix.py",{"logm","expm","sqrtm"},env)
    run_defs("diffcsp/pl_modules/lattice/crystal_family.py",{"CrystalFamily"},env)
    run_defs("diffcsp/pl_modules/diffusion.py",{"SinusoidalTimeEmbeddings"},env)
    src=ast.parse((SRC/"diffcsp/pl_modules/diffusion.py").read_text())
    klass=next(x for x in src.body if isinstance(x,ast.ClassDef) and x.name=="CSPDiffusion")
    body=next(x for x in klass.body if isinstance(x,ast.FunctionDef) and x.name=="sample")
    exec(compile(ast.Module(body=[body],type_ignores=[]),"diffusion.py","exec"),env)
    net=env["CSPNet"](hidden_dim=512,latent_dim=256,num_layers=6,num_freqs=128,ln=True).to(device).eval()
    family=env["CrystalFamily"]().to(device)
    with np.load(SRC/"weights.npz") as wp:
        weights={k:torch.from_numpy(wp[k]).to(device) for k in wp.files}
    def subset(prefix):return {k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)}
    net.load_state_dict(subset("decoder."),strict=True)
    family.load_state_dict(subset("crystal_family."),strict=True)
    return env["sample"],net,family,env["SinusoidalTimeEmbeddings"](256),subset("beta_scheduler."),subset("sigma_scheduler.")

def arrays(rows,device):
    ops=[];zin=[];gr=[];anchors=[];counts=[];start=0
    for row in rows:
        n=0
        for z,letter in row["template"]:
            p=GR[row["group"]-1]+LETTERS.index(letter)
            a,b=ORB[p:p+2]
            op=np.zeros((b-a,4,4));op[:,:3,:3]=M[a:b].transpose(0,2,1);op[:,:3,3]=T[a:b]/24;op[:,3,3]=1
            ops.extend(op);zin.extend([z]*(b-a));anchors.extend([start]*(b-a));start+=b-a;n+=b-a
        counts.append(n);gr.append(row["group"])
    ten=lambda a,dtype:torch.as_tensor(np.asarray(a),dtype=dtype,device=device)
    op=ten(ops,torch.float32)
    return SimpleNamespace(num_graphs=len(rows),num_nodes=start,num_atoms=ten(counts,torch.long),
        ops=op,ops_inv=torch.linalg.pinv(op[:,:3,:3]),anchor_index=ten(anchors,torch.long),
        spacegroup=ten(gr,torch.long),atom_types=ten(zin,torch.long),
        batch=ten(np.repeat(np.arange(len(rows)),counts),torch.long))

def runner(args,steps,device,optimize=True):
    call,net,family,emb,beta,sigma=args
    # Preserve the author's complete endpoint and native original schedule when steps=999.
    n=len(beta["betas"])-1
    assert n==1000
    times=torch.linspace(0,n-1,steps+1,device=device).round().long()
    assert len(torch.unique(times))==steps+1
    ab=beta["alphas_cumprod"][times]
    if steps==n-1:
        vals={k:v[:n] for k,v in beta.items()}
    else:
        a=torch.cat([ab.new_ones(1),ab[1:]/ab[:-1]])
        b=1-a
        sig=torch.cat([ab.new_zeros(1),(b[1:]*(1-ab[:-1])/(1-ab[1:])).sqrt()])
        vals=dict(betas=b,alphas=a,alphas_cumprod=ab,sigmas=sig)
    obj=SimpleNamespace(device=device,decoder=net,crystal_family=family,
             time_embedding=lambda t:emb(times[t]),beta_scheduler=SimpleNamespace(**vals,timesteps=steps+1),
             sigma_scheduler=SimpleNamespace(**{k:v[times] for k,v in sigma.items()},sigma_begin=.005))
    def sample(rows,seed):
        batch=arrays(rows,device)
        # Topology is fixed, and all six source layers embed the exact same relative differences.
        cached_edges=None
        if optimize:
            edge_source=net.gen_edges; embedding=net.dis_emb.forward
            edge,_=edge_source(batch.num_atoms,torch.zeros((batch.num_nodes,3),device=device))
            net.gen_edges=lambda num,frac:(edge,(frac[edge[1]]-frac[edge[0]])%1.)
            pair=[None,None]
            def cache(diff):
                if diff is not pair[0]:pair[:]=[diff,embedding(diff)]
                return pair[1]
            net.dis_emb.forward=cache
        torch.manual_seed(seed)
        with torch.no_grad():
            try:out,traj=call(obj,batch)
            finally:
                if optimize:
                    net.gen_edges=edge_source;net.dis_emb.forward=embedding
        cells=out["lattices"].cpu().numpy()
        fs=out["frac_coords"].cpu().numpy()
        zs=batch.atom_types.cpu().numpy()
        outrows=[];offset=0
        for i,n in enumerate(batch.num_atoms.tolist()):
            outrows.append((cells[i],fs[offset:offset+n],zs[offset:offset+n]))
            offset+=n
        return outrows
    return sample,times.tolist()

def main(a):
    torch.set_num_threads(1);device=a.device
    if device=="cuda":torch.backends.cuda.matmul.allow_tf32=a.tf32
    targets=json.loads((HERE/f"{a.split}-cohort.json").read_text())[:a.count]
    # The sampler receives only the group and occupied/species-labelled Wyckoff template.
    rows=[{k:r[k] for k in ("group","template")} if "error" not in r else None for r in targets]
    arch=construct(device);run,times=runner(arch,a.steps,device)
    if a.parity:
        ref,_=runner(arch,a.steps,device,optimize=False)
        re=ref(rows[:2],700);ours=run(rows[:2],700)
        el=max(np.max(np.abs(x[0]-y[0])) for x,y in zip(re,ours))
        ef=max(np.max(np.abs((x[1]-y[1]+.5)%1-.5)) for x,y in zip(re,ours))
        assert el<1e-5 and ef<1e-5,(el,ef)
        print("parity",el,ef)
        (HERE/"external/parity.json").write_text(json.dumps(dict(steps=a.steps,device=device,cell=float(el),frac=float(ef)))+"\n")
        return
    name=f"DiffCSPpp-S{a.steps}"
    full=a.split=="test" and a.count==128 and a.k==20
    tag=name if full else f"{name}-{a.split}-{a.count}x{a.k}"
    dest=HERE/"pools";dest.mkdir(exist_ok=True)
    archive=dest/(tag+".npz");wip=dest/(tag+"-partial.npz")
    max_n=max(r["std_n"] for r in targets if "std_n" in r)
    shape=(len(rows),a.k)
    if wip.exists():
        with np.load(wip) as b:l,f,z,done,secs=[b[key] for key in ("L","F","Z","done","seconds")]
    else:
        l=np.full((*shape,3,3),np.nan);f=np.full((*shape,max_n,3),np.nan);z=np.zeros((len(rows),max_n),np.int16)
        done=np.zeros(shape,bool);secs=np.zeros(shape)
    # Group by material in small batches to bound the quadratic all-pairs graph.
    allslots=[(i,j) for j in range(a.k) for i,r in enumerate(rows) if r is not None]
    nb=0
    for lo in range(0,len(allslots),a.batch):
        ix=allslots[lo:lo+a.batch]
        if all(done[i,j] for i,j in ix):continue
        if device=="cuda":torch.cuda.synchronize()
        t=perf_counter();result=run([rows[i] for i,j in ix],78109+lo)
        if device=="cuda":torch.cuda.synchronize()
        elapsed=perf_counter()-t
        for (i,j),(ll,ff,zz) in zip(ix,result):
            l[i,j]=ll;f[i,j,:len(zz)]=ff;z[i,:len(zz)]=zz;done[i,j]=True;secs[i,j]=elapsed/len(ix)
        nb+=1
        if nb%4==0 or lo+a.batch>=len(allslots):
            tmp=wip.with_suffix(".tmp.npz");np.savez(tmp,L=l,F=f,Z=z,done=done,seconds=secs);tmp.replace(wip)
        if nb%8==0:print("completed",done.sum(),"of",len(allslots),flush=True)
    np.savez_compressed(archive,id=[r["id"] for r in targets],L=l,F=f,Z=z,elapsed=secs)
    (dest/(tag+".json")).write_text(json.dumps(dict(
        checkpoint_sha256="8d7615c73e4f55e52c310122a0e194057d0ccff252255a6497cbed41ca624142",
        torch=torch.__version__,device=torch.cuda.get_device_name() if device=="cuda" else "cpu",
        tf32=a.tf32,steps=a.steps,nfe=2*a.steps,draws=a.k,targets=len(rows),times=times,batch=a.batch,
        seconds=float(secs.sum()),slots=int(done.sum()),source="pinned native CSPDiffusion.sample, original projections and corrector",
        schedule="noise marginals at equally spaced original time indices, posterior beta respaced; original at 999",
        pool_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),indent=2)+"\n")
    print(tag,"seconds",secs.sum(),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--device",default="cpu");p.add_argument("--steps",type=int,default=20)
    p.add_argument("--split",default="test");p.add_argument("--count",type=int,default=128)
    p.add_argument("--k",type=int,default=20);p.add_argument("--batch",type=int,default=12);p.add_argument("--parity",action="store_true")
    p.add_argument("--tf32",action="store_true");main(p.parse_args())
