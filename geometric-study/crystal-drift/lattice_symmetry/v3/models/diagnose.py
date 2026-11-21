"""Record update size, native volume, cell anisotropy and validation MR conditional on site count."""
import csv,json
from pathlib import Path
import numpy as np
from .sampler import HERE

def main():
    base=HERE/'fits';prior=np.load(base/'prior/dev.npz')
    prior_score=np.load(base/'prior/dev-scores.npz')
    ids=prior['ids'];L0=prior['L'];F0=prior['F'];z=prior['Z']
    n=(z>0).sum(-1)
    rows=[]
    for fit in sorted(base.iterdir()):
        sel=fit/'selection.json'
        if not sel.exists():continue
        ck=json.loads(sel.read_text())['chosen'];weight=fit/ck;key=weight.stem
        with np.load(weight.with_name(key+'-dev.npz')) as p,np.load(weight.with_name(key+'-dev-scores.npz')) as sc:
            assert np.array_equal(ids,p['ids']) and np.array_equal(z,p['Z'])
            L=p['L'];F=p['F'];ratio=np.linalg.slogdet(L)[1]-np.linalg.slogdet(L0)[1]
            shift=(F-F0+.5)%1-.5
            phys=shift@L0;per_site=np.linalg.norm(phys,axis=-1)
            strain=np.linalg.svd(np.linalg.solve(L0,L),compute_uv=False)
            aniso=np.log(strain.max(-1)/strain.min(-1))
            rms=sc['rms'];logv=sc['log_volume_ratio'];prior_logv=prior_score['log_volume_ratio']
            for lo,hi in [(1,4),(5,8),(9,12),(13,20),(1,20)]:
                mask=(n[:,0]>=lo)&(n[:,0]<=hi)
                active=(z[mask]>0);ok=np.isfinite(rms[mask])
                if not mask.any():continue
                rows.append(dict(model=fit.name,checkpoint=ck,lo=lo,hi=hi,targets=int(mask.sum()),
                       median_site_step_A=float(np.median(per_site[mask][active])),
                       p90_site_step_A=float(np.quantile(per_site[mask][active],.9)),
                       sites_below_1e_5=float(np.mean(per_site[mask][active]<1e-5)),
                       median_logvolume_change=float(np.median(ratio[mask])),
                       signed_native_logvolume_bias=float(np.mean(logv[mask])),
                       median_log_singular_ratio=float(np.median(aniso[mask])),
                       hits=int(ok.any(1).sum()),native=int((ok&(np.abs(logv[mask])<=np.log(1.1))).any(1).sum()),
                       seed_volume_sensitivity_hits=int((ok&(np.abs(prior_logv[mask])<=np.log(1.1))).any(1).sum())))
    with (HERE/'geometry-dev.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':main()

