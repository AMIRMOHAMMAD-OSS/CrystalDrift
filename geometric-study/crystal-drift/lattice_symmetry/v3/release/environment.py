"""Record only the scientific software and hardware powering an identified study environment."""
import argparse
import hashlib
import importlib.metadata as packages
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

HERE=Path(__file__).resolve().parent

def main(opts):
    if os.environ.get('UFLOW_TIMM'):sys.path.insert(0,os.environ['UFLOW_TIMM'])
    def version(name):
        try:return packages.version(name)
        except packages.PackageNotFoundError:return None
    names=('numpy','scipy','torch','torchvision','timm','pymatgen','spglib','numba','llvmlite','matminer','smact','chgnet','ase','pyxtal','safetensors','matplotlib','pandas')
    value=dict(role=opts.role,python=sys.version,python_executable=sys.executable,platform=platform.platform(),packages={x:version(x) for x in names})
    if opts.role=='training':
        import torch
        value['pytorch']=dict(version=torch.__version__,cuda_runtime=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
            compiled_arches=torch._C._cuda_getArchFlags(),
            library_path=torch.__file__,threads_for_one_call=1)
        value['nvidia_smi']=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
        from importlib.util import find_spec
        origin=find_spec('timm').origin;value['timm']=dict(path=origin,version=value['packages']['timm'],init_sha256=hashlib.sha256(Path(origin).read_bytes()).hexdigest())
    dst=HERE/(opts.role+'-environment.json');dst.write_text(json.dumps(value,indent=2)+'\n');print(dst)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role',choices=['training','linux-matcher','windows-matcher']);main(p.parse_args())
