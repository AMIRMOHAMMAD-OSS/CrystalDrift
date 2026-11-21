"""Exercise model and matcher entry points with only one chosen extraction on Python's search path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[v]="1"
sys.dont_write_bytecode=True

def main():
    a=argparse.ArgumentParser()
    a.add_argument("root",type=Path);a.add_argument("mode",choices=("model","matcher","symmetry"))
    a.add_argument("output",type=Path)
    a.add_argument("--dependency-site",type=Path,help="Optional external library directory used by the pinned matcher.")
    args=a.parse_args()
    root=args.root.resolve();os.chdir(root);sys.path.insert(0,str(root))
    assert sys.flags.isolated,"Run with python -I to exclude the live checkout and user-site paths."
    if args.dependency_site:
        site=args.dependency_site.resolve()
        assert site.name=="site-packages" and site.is_dir() and not site.is_relative_to(root)
        sys.path.append(str(site))
    import numpy as np
    v3=root/"lattice_symmetry/v3"
    with np.load(root/"scaled/mp20-data/val.npz") as d:
        z=d["z"][:3];f=d["f"][:3];k=d["k"][:3]
    receipt=dict(extraction=str(root),python=sys.executable,version=sys.version,mode=args.mode,case_count=len(z))
    if args.dependency_site:receipt["external_library_directory"]=str(site)
    if args.mode=="model":
        import torch
        from lattice_symmetry.v3.models import prepare,train
        from lattice_symmetry.v3.models.sampler import Sampler
        from lattice_symmetry.v3.models.network import OrbitNet,TIERS,pack
        from lattice_symmetry.v3.geometry.descriptors import feature_blocks,bank_direction
        from lattice_symmetry.v3.geometry import decorated_witness
        from lattice_symmetry.v2.csp.model import load_service
        from lattice_symmetry.v3.uflow import runtime
        torch.manual_seed(921);torch.set_num_threads(1)
        net=runtime.model("small","cpu").eval()
        flow=runtime.transport()
        ll,ff,zz=runtime.propose(net,flow,z)
        assert np.array_equal(zz,z) and np.isfinite(ll).all() and np.isfinite(ff).all()
        receipt["native_uflow_parameters"]=sum(p.numel() for p in net.parameters())
        old=load_service("angular",11)
        kk,oldf=old(z,np.random.default_rng(101))
        assert np.isfinite(kk).all() and np.isfinite(oldf).all()
        receipt["released_angular_deployed"]=True
        service=Sampler();seeds=[service.seeds(row,1,4202+i)[0] for i,row in enumerate(z)]
        args0=pack(seeds,"cpu")
        catalog=v3/"evaluation/first-frozen.json"
        first=None
        if catalog.is_file():
            specs=json.loads(catalog.read_text())["methods"]
            first=next(row for row in specs if row["tag"]=="contacts-small-11")
        if first is not None and (path:=v3/first["checkpoint"]).is_file():
            assert hashlib.sha256(path.read_bytes()).hexdigest()==first["sha"];net=service.load(path)
            receipt["orbit_weights"]=first["sha"]
        else:
            manifest=v3/"release/EVIDENCE-MANIFEST.json"
            if manifest.is_file():assert json.loads(manifest.read_text())["scope"]!="full"
            net=OrbitNet(*TIERS["small"]).eval();receipt["orbit_weights"]="source-only initialization"
        outs=service.apply(net,seeds)
        assert all(np.isfinite(l).all() and np.isfinite(x).all() and np.array_equal(np.sort(a),np.sort(row[row>0]))
                   for (l,x,a),row in zip(outs,z))
        dims={}
        for arm in ("angular","contacts","reciprocal"):
            with torch.inference_mode():
                raw=feature_blocks(*args0[:4],arm)
                features=torch.cat(list(raw.values()),dim=1)
                q=features.unsqueeze(1)
                assert torch.isfinite(features).all() and bank_direction(q,q,q).abs().max().item()==0
            dims[arm]=int(features.shape[1])
        receipt["orbit_case_groups"]=[s.group for s in seeds];receipt["feature_dimensions"]=dims
        torch.set_default_dtype(torch.float64)
        positions=torch.zeros(2,8,3);types=torch.tensor([[8]*4+[14]*4]*2)
        for i,s in enumerate(((0,1,4,6),(0,1,3,7))):positions[i,:,0]=torch.tensor(s).repeat(2)/12
        positions[:,4:,1]=1/3;zero=torch.zeros(2,6)
        spectrum=decorated_witness.fingerprint(zero,positions,types,"harmonic")
        d=float((spectrum[0]-spectrum[1]).abs().max())
        assert abs(d-1/(16*np.sqrt(2)))<1e-14;receipt["old_harmonic_separation"]=d
    elif args.mode=="symmetry":
        from lattice_symmetry.v3.geometry import verify,verify_decoder
        for label,module in (("descriptor",verify),("decoder",verify_decoder)):
            output=args.output.with_name("clean-proof-"+label+".json")
            sys.argv=[label,"--device","cpu","--output",str(output)]
            module.main()
            receipt[label+"_sha256"]=hashlib.sha256(output.read_bytes()).hexdigest()
    else:
        import spglib
        from pymatgen.analysis.structure_matcher import StructureMatcher
        from standard_match import SETTINGS,make_structure
        from lattice_symmetry.v3.geometry.subgroups import crosscheck_table
        crosscheck_table()
        v2=StructureMatcher(**SETTINGS);r=[]
        for i in range(len(z)):
            x=make_structure(k[i],f[i],z[i]);value=v2.get_rms_dist(x,x);assert value is not None
            r.append(value[0])
        assert max(r)<1e-9
        receipt.update(self_rms=r,spglib=spglib.__version__,independently_verified_native_groups=230)
    local={}
    for name,m in list(sys.modules.items()):
        if name.startswith(("lattice_symmetry.","scaled.","revision.","uflowcsp")) or name in {"data","model","train","run","standard_match"}:
            file=getattr(m,"__file__",None)
            if file:
                path=Path(file).resolve()
                assert path.is_relative_to(root),(name,path,root)
                local[name]=str(path.relative_to(root))
    receipt["local_modules"]=local;args.output.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf8")
    print(args.mode,len(local),{k:v for k,v in receipt.items() if k!="local_modules"},flush=True)

if __name__=="__main__":main()
