"""Extract tensor-only state from the verified public checkpoint and pin source provenance."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"expansion/external"))
from weights import Reader

SHA="8d7615c73e4f55e52c310122a0e194057d0ccff252255a6497cbed41ca624142"
PIN="e82e7ffa7cb2a383bde69b067a343ca137d73e47"
SOURCES=("diffcsp/pl_modules/cspnet.py","diffcsp/pl_modules/diffusion.py","diffcsp/pl_modules/lattice/crystal_family.py","diffcsp/pl_modules/lattice/matrix.py","scripts/eval_utils.py","LICENSE")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare(ckpt,repo):
    out=HERE/"external";out.mkdir(exist_ok=True)
    assert sha(ckpt)==SHA
    assert subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()==PIN
    with zipfile.ZipFile(ckpt) as z:
        assert z.testzip() is None
        obj=Reader(io.BytesIO(z.read("archive/data.pkl"))).load()
        tensors={}
        for k,(storage,offset,size,stride) in obj["state_dict"].items():
            dtype,key,n=storage
            x=np.frombuffer(z.read("archive/data/"+key),dtype)
            assert len(x)==n and offset>=0 and all(s>=0 for s in stride)
            assert offset+sum((m-1)*s for m,s in zip(size,stride))<n
            tensors[k]=np.lib.stride_tricks.as_strided(x[offset:],shape=size,
                            strides=tuple(s*x.itemsize for s in stride)).copy()
    np.savez(out/"weights.npz",**tensors)
    for name in SOURCES:
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/name,dest)
    metadata=dict(source_commit=PIN,checkpoint_sha256=SHA,source_url="https://github.com/jiaor17/DiffCSP-PP",
            checkpoint_url="https://drive.google.com/uc?id=1XlV0kntXqkyVoU-wslGQVa28852v2Hpv",
            hparams_sha256=sha(HERE/"diffpp-hparams.yaml"),epoch=obj["epoch"],global_step=obj["global_step"],
            weights_sha256=sha(out/"weights.npz"),source_hashes={s:sha(out/s) for s in SOURCES})
    (out/"source.json").write_text(json.dumps(metadata,indent=2)+"\n")
if __name__=="__main__":prepare(*map(Path,sys.argv[1:]))
