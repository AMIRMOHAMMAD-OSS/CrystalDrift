"""Replay the ordered first 64 model outputs for settings reconstructed from launch records."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from lattice_symmetry.v3.evaluation.paired import seeds, fit_selector
from lattice_symmetry.v3.evaluation.predict import setup_split,draw_seed
from lattice_symmetry.v3.models.sampler import Sampler

V3 = Path(__file__).resolve().parents[1]


def run(home, output):
    root=home/'recovered-ephemeral/crytaldrift-v3-20261121'
    candidates=json.loads((home/'settings-candidates/settings-candidates.json').read_text())
    matched=json.loads((root/'raw-settings-recovery.json').read_text())['matched']
    residual=[row for row in candidates if row['original'] not in matched]
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    probes={}
    for row in sorted(residual,key=lambda item:item['source_tag']):
        _,panel,tag,_=row['original'].rsplit('/',3)
        device='cuda' if tag=='reference-scale-11' else 'cpu'
        receipt_path=home/'settings-candidates'/row['candidates'][device+'-32-1']['file']
        settings=json.loads(receipt_path.read_text())
        assert (settings['device'],settings['batch'],settings['threads'])==(device,32,1)
        input_signature=json.loads(settings['signature'])
        registry=json.loads((V3/'evaluation'/('weight-'+tag+'.json')).read_text())['methods'][0]
        assert registry['sha']==input_signature['weight']
        service=Sampler(device)
        weight=V3/registry['checkpoint']
        if not weight.is_file():
            pointer=json.loads(weight.with_name(weight.name+'.pointer.json').read_text())
            weight=root/Path(pointer['path']).relative_to('/tmp/crytaldrift-v3-20261121')
        network=service.load(weight)
        cohort,_=setup_split(panel)
        selector=fit_selector(service,input_signature['selector'])
        rank=None if selector is None else lambda z:selector.rank(z,input_signature['selector'])
        draw=[]
        for _,ix,z in cohort[:4]:
            rng=np.random.default_rng(draw_seed(panel,ix))
            draw.extend(seeds(service,z,20,rng,rank,input_signature['family']=='donor'))
        source=next(iter(sorted((root/'raw'/panel/tag).glob('?????-?????.npz'))))
        with np.load(source) as data:
            assert str(data['signature'])==settings['signature'] and str(data['tag'])==tag
            np.testing.assert_array_equal(data['ids'][:4],[key for key,_,_ in cohort[:4]])
            max_cell=max_frac=0.
            exact=0
            for start in (0,32):
                for i,(cell,frac,species) in enumerate(service.apply(network,draw[start:start+32]),start):
                    a,b=divmod(i,20)
                    old_cell=data['L'][a,b];old_frac=data['F'][a,b,:len(frac)]
                    np.testing.assert_array_equal(data['Z'][a,b,:len(frac)],species)
                    assert draw[i].group==data['group'][a,b]
                    max_cell=max(max_cell,float(np.max(abs(cell-old_cell))))
                    max_frac=max(max_frac,float(np.max(abs(frac-old_frac))))
                    exact+=int(np.array_equal(cell,old_cell) and np.array_equal(frac,old_frac))
        assert np.isfinite([max_cell,max_frac]).all()
        probes[row['original']]=dict(device=device,batch=32,threads=1,draws=64,bitwise_geometry=exact,
                                     max_abs_cell=max_cell,max_abs_fractional=max_frac,
                                     settings_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest())
        output.write_text(json.dumps(probes,indent=1)+'\n')
        print(panel,tag,exact,max_cell,max_frac,flush=True)
        del service, network
        if device=='cuda': torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('home',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.home,a.output)
