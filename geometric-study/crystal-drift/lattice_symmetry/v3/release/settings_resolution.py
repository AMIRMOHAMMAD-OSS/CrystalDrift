"""Deposit documented launch settings only after an independent ordered geometry replay."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SCALE={'reference-scale-11','radial4-small','angular-small-local','angular-small-systemaux','angular-large-11'}


def resolve(home):
    root=home/'recovered-ephemeral/crytaldrift-v3-20261121'
    candidates=json.loads((home/'settings-candidates/settings-candidates.json').read_text())
    matched=json.loads((root/'raw-settings-recovery.json').read_text())['matched']
    reproduced=json.loads((HERE/'run-settings-probes.json').read_text())
    rows={}
    for arm in candidates:
        key=arm['original']
        if key in matched:
            continue
        probe=reproduced[key]
        _,panel,tag,_=key.rsplit('/',3)
        device='cuda' if tag=='reference-scale-11' else 'cpu'
        profile=arm['candidates'][device+'-32-1']
        src=home/'settings-candidates'/profile['file']
        payload=src.read_bytes()
        digest=hashlib.sha256(payload).hexdigest()
        assert digest==probe['settings_sha256']==profile['sha256'] and probe['draws']==64
        assert probe['max_abs_cell']<0.0001 and probe['max_abs_fractional']<0.00001,key
        assert probe['device']==device and probe['batch']==32 and probe['threads']==1
        script='evaluation/scale_wave.py' if tag in SCALE else 'evaluation/wave.py'
        corroboration=sorted(source for source in matched if source.split('/')[-2]==tag)
        rows[key]=dict(sha256=digest,binding='reconstructed from launcher and ordered geometry replay; original metadata digest unverified',
                       launcher=script,paired_generator='evaluation/paired.py',verified_sibling_panels=corroboration,
                       device=device,batch=32,threads=1,probe=probe)
        out=root/'raw'/panel/tag/'settings.json'
        if out.is_file() and out.read_bytes()!=payload:raise FileExistsError(out)
        out.write_bytes(payload)
    report=dict(original_settings=len(matched),reconstructed=len(rows),rows=rows,
                independently_replayed=sum(row['probe']['draws'] for row in rows.values()))
    assert len(candidates)==len(rows)+len(matched)==91
    # Publish the completion marker only after all per-arm materials are present.
    output=root/'settings-resolution-final.json';tmp=output.with_suffix('.writing')
    tmp.write_text(json.dumps(report,indent=1)+'\n');tmp.replace(output)
    print(len(matched),'archived settings,',len(rows),'launcher reconstructions;',report['independently_replayed'],'proposals')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('home',type=Path)
    resolve(p.parse_args().home)
