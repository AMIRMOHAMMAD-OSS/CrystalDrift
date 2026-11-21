"""Verify every deployable and optionally materialize a portable evidence tree."""
import argparse,hashlib,json,os,shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):value.update(block)
    return value.hexdigest()

def rows():
    chosen={}
    for selection in (HERE/'fits').glob('*/selection.json'):
        info=json.loads(selection.read_text())
        key=(selection.parent/info['chosen']).resolve()
        chosen.setdefault(key,[]).append(selection.parent.name)
    for finished in sorted((HERE/'fits').glob('*/complete.json')):
        fit=json.loads(finished.read_text())
        for stage,expected in fit['checkpoint_sha256'].items():
            path=(finished.parent/(stage+'.pt')).resolve()
            receipt=path.with_suffix('.pt.pointer.json')
            info=json.loads(receipt.read_text()) if receipt.exists() else {}
            if info and expected!=info['sha256']:raise ValueError(f'Conflicting SHA for {path}')
            actual=path if path.is_file() else Path(info['windows_path' if os.name=='nt' else 'path'])
            size=actual.stat().st_size
            if size!=info.get('bytes',size):raise ValueError(f'Byte count differs for {path}')
            yield dict(path=path.relative_to(ROOT).as_posix(),actual=str(actual),bytes=size,
                       sha256=expected,selected_for=chosen.get(path,[]))
    initial=HERE/'controls/untrained-small-11/initial.pt'
    if initial.is_file():
        yield dict(path=initial.relative_to(ROOT).as_posix(),actual=str(initial),bytes=initial.stat().st_size,
                   sha256=digest(initial),selected_for=['untrained-small-11'])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--destination',type=Path,help='Copy the requested tensors to their ordinary release-relative paths')
    parser.add_argument('--selected',action='store_true',help='Only materialize/verify tensors referenced by frozen selections')
    args=parser.parse_args();manifest=[]
    for row in rows():
        if args.selected and not row['selected_for']:continue
        src=Path(row['actual'])
        if digest(src)!=row['sha256']:raise ValueError(f'Invalid source weight {src}')
        if args.destination:
            dst=args.destination.resolve()/row['path']
            if dst.exists() and digest(dst)!=row['sha256']:raise FileExistsError(f'Different existing file at {dst}')
            if not dst.exists():
                dst.parent.mkdir(parents=True,exist_ok=True);temp=dst.with_suffix('.copying')
                shutil.copyfile(src,temp)
                if digest(temp)!=row['sha256']:raise OSError(f'Copied weight differs at {temp}')
                os.replace(temp,dst)
        manifest.append(row)
    name='selected-weights.json' if args.selected else 'deployment-weights.json'
    doc=dict(files=manifest,total_bytes=sum(r['bytes'] for r in manifest),verification='streaming SHA-256')
    (HERE/name).write_text(json.dumps(doc,indent=2))
    print('Verified',len(manifest),'distinct deployables,',doc['total_bytes'],'bytes.')

if __name__=='__main__':main()
