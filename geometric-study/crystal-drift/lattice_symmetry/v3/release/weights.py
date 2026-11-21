"""Restore each distinct validation-selected model and generate real compositions from a clean extraction."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()

def main(root,output):
    assert sys.flags.isolated
    root=root.resolve();sys.path.insert(0,str(root));os.chdir(root)
    v3=root/'lattice_symmetry/v3'
    index=json.loads((v3/'release/EVIDENCE-MANIFEST.json').read_text())
    full=index['scope']=='full'
    os.environ.pop('UFLOW_TIMM',None)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    import numpy as np
    import torch
    from lattice_symmetry.v3.models.sampler import Sampler
    from lattice_symmetry.v3.uflow import runtime
    assert Path(runtime.__file__).resolve().is_relative_to(root)
    torch.set_num_threads(1)
    with np.load(root/'scaled/mp20-data/val.npz') as d:z=d['z'][:2]
    service=Sampler('cpu');seeds=[service.seeds(row,1,4517+i)[0] for i,row in enumerate(z)]
    selected={};aliases={}
    for registry in sorted((v3/'models/fits').glob('*/selection.json')):
        value=json.loads(registry.read_text());file=(registry.parent/value['chosen']).resolve()
        if not file.is_file():
            if str(value.get('sha256','')).lower() in ('identity','none',''):continue
            if not full:continue
            raise FileNotFoundError(file)
        if file.suffix!='.pt':continue
        name=file.relative_to(root).as_posix()
        aliases[registry.parent.name]=name
        if name in selected:continue
        sha=digest(file);assert sha==index['files'][name]['sha256']
        net=service.load(file)
        actual=service.apply(net,seeds)
        for (cell,frac,atom),formula in zip(actual,z):
            assert np.isfinite(cell).all() and np.isfinite(frac).all() and abs(np.linalg.det(cell))>0
            np.testing.assert_array_equal(np.sort(atom),np.sort(formula[formula>0]))
        selected[name]={'sha256':sha,'trainable_parameters':sum(p.numel() for p in net.parameters() if p.requires_grad)}
        del net;gc.collect()
    native={};flow=runtime.transport()
    torch.manual_seed(2101);noise=flow.sample_prior(len(z),100,12,'cpu')
    for registry in sorted((v3/'uflow/fits').glob('*/selection.json')):
        value=json.loads(registry.read_text());file=(registry.parent/value['checkpoint']).resolve()
        if not file.is_file():
            if not full:continue
            raise FileNotFoundError(file)
        sha=digest(file);assert sha==value['checkpoint_sha']
        tier=registry.parent.name.split('-')[0]
        net=runtime.model(tier,'cpu').eval()
        net.load_state_dict(torch.load(file,map_location='cpu',weights_only=True),strict=True)
        cell,frac,atom=runtime.propose(net,flow,z,noise)
        assert np.isfinite(cell).all() and np.isfinite(frac).all()
        np.testing.assert_array_equal(atom,z)
        native[registry.parent.name]={'file':file.relative_to(root).as_posix(),'sha256':sha,'parameters':runtime.counts(net)}
        del net;gc.collect()
    ema=v3/'uflow/released/ema.safetensors'
    if full or ema.exists():
        os.environ['UFLOW_WEIGHTS']=str(ema)
        net=runtime.model('released','cpu').eval()
        cell,frac,atom=runtime.propose(net,flow,z,noise)
        assert np.isfinite(cell).all() and np.isfinite(frac).all();np.testing.assert_array_equal(atom,z)
        native['released']={'file':ema.relative_to(root).as_posix(),'sha256':digest(ema),'parameters':runtime.counts(net)}
    if full:assert selected and len(native)>=11
    result=dict(mode=index['scope'],real_formulas=len(z),orbit_selected=selected,orbit_aliases=aliases,native=native,
                python=sys.version,torch=torch.__version__)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print('Restored selected orbit weights, native controls and reference',len(selected),len(native),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    main(a.root,a.output.resolve())
