"""Independent public matcher audit of the native-cell training chart for every represented group."""
import json,subprocess,sys,os
import numpy as np
from .sampler import HERE,V2,V3,Sampler

def main():
    s=Sampler();ix=[v[0] for g,v in sorted(s.by_group.items())]
    allcases=[s.target(i) for i in ix]
    L=np.array([r.L for r in allcases])[:,None,:,:]
    F=np.zeros((len(ix),1,20,3));Z=np.zeros((len(ix),1,20),int)
    for j,r in enumerate(allcases):
        F[j,0,:len(r.Z)]=r.F;Z[j,0,:len(r.Z)]=r.Z
    out=HERE/'native-chart-witnesses.npz';sc=HERE/'native-chart-roundtrip.npz'
    np.savez_compressed(out,ids=np.array(s.ids)[ix],L=L,F=F,Z=Z,group=[r.group for r in allcases])
    exe=os.getenv('MATCHER_PYTHON') or json.loads((V2/'csp/local-runtime.json').read_text())['matcher_executable']
    subprocess.run([exe,str(V3/'evaluation/score.py'),str(out),'--split','train','--output',str(sc),'--public'],check=True)
    with np.load(sc) as q:
        assert np.isfinite(q['rms']).all() and not q['error'].any()
        result=dict(groups=len(ix),max_normalized_rms=float(np.nanmax(q['rms'])),
             max_absolute_logvolume=float(np.max(np.abs(q['log_volume_ratio']))),independent_public_checks=int(q['public_checks']))
    (HERE/'native-chart-roundtrip.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
if __name__=='__main__':main()

