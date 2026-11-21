"""Select among the four registered exposure milestones using development only."""
import argparse,json,os,time
from pathlib import Path
from .select_weights import score,write,digest
from .sampler import HERE,Sampler
from .train import dev

def cycle(s):
    for target in (HERE/'fits').glob('*-8pass'):
        if (target/'confirmation.json').exists():continue
        if (target/'epoch4-dev.npz').exists():score(target/'epoch4-dev.npz')
        if not (target/'epoch8-dev.npz').exists():continue
        if os.name!='nt' and (os.getenv('CRYTALDRIFT_WINDOWS_CONFIRM')=='1' or
                (os.getenv('WSL_DISTRO_NAME') and (HERE/'windows-confirm-active.json').exists())):
            # The registered local launch delegates frozen confirmation to the original
            # Windows CPU sampler/matcher. Cluster or independent runs score in place.
            continue
        base=target.with_name(target.name.removesuffix('-8pass'))
        vals={}
        paths=[]
        for i,(folder,mile) in enumerate([(base,'half'),(base,'full'),(target,'epoch4'),(target,'epoch8')]):
            vals[mile]=score(folder/(mile+'-dev.npz'))
            paths.append(folder/(mile+'.pt'))
        if not (target/'selection.json').exists():
            key=max(vals,key=lambda k:vals[k]['hits'])
            pick=paths[list(vals).index(key)]
            dest=pick.name if pick.parent==target else '../'+base.name+'/'+pick.name
            write(target/'selection.json',dict(chosen=dest,sha256=digest(pick),dev=vals,
                    rule='highest MR5 at 0.5, 1, 4, 8 traversals; earliest tie'))
        if not (target/'confirmation.json').exists():
            pick=json.loads((target/'selection.json').read_text())['chosen']
            path=target/'frozen-confirm.npz'
            if not path.exists():dev(s,s.load(target/pick),path,'confirm')
            write(target/'confirmation.json',score(path))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--watch',action='store_true',help='Watch all four registered extensions through their frozen confirmations')
    args=parser.parse_args();s=Sampler()
    while True:
        cycle(s)
        if not args.watch or len(list((HERE/'fits').glob('*-8pass/confirmation.json')))>=4:break
        time.sleep(8)

