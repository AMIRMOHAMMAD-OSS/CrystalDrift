"""Separate native log volume, deposited-basis lattice shape, and site-placement errors."""
import csv
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="1"
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"scaled"))
from evaluate import process


def site_displacement(z, fgen, fref):
    """Feasible fixed-basis motif RMS after same-species assignment and origin refinement."""
    a=z[z>0]
    x=fgen[:len(a)].astype(float)
    y=fref[:len(a)].astype(float)
    groups=[np.flatnonzero(a==s) for s in np.unique(a)]
    rare=min(groups,key=len)
    best=np.inf
    for anchor in rare:
        shift=y[rare[0]]-x[anchor]
        for _ in range(4):
            residuals=[]
            for g in groups:
                delta=y[g,None,:]-x[None,g,:]-shift
                delta-=np.floor(delta+.5)
                d2=np.einsum("ijk,ijk->ij",delta,delta)
                i,j=linear_sum_assignment(d2)
                residuals.extend(delta[i,j])
            residuals=np.array(residuals)
            best=min(best,np.sqrt(np.square(residuals).sum()/len(a)))
            increment=residuals.mean(0)
            if np.linalg.norm(increment)<1e-10:break
            shift+=increment
    return best


def main():
    with np.load(ROOT/"revision/full/mp20/cohort.npz") as d:
        ref={key:d[key] for key in ("ids","z","k","f")}
    obs=process(ref["k"][:128],ref["f"][:128],ref["z"][:128])
    rows=[]
    for ready in sorted((HERE/"test").glob("*/ready.json")):
        base=ready.parent
        dest=base/"geometry.npz"
        if not dest.exists():
            k=np.load(base/"k.npy",mmap_mode="r")
            f=np.load(base/"f.npy",mmap_mode="r")
            diff=k-ref["k"][:,None,:]
            trace=diff[:,:,:3].sum(-1)
            diff[:,:,:3]-=trace[:,:,None]/3
            shape=np.linalg.norm(diff,axis=-1)
            # Counterfactual geometry with exact reference lattice, preserving all native fractional sites.
            rz=np.repeat(ref["z"][:128],5,axis=0);rk=np.repeat(ref["k"][:128],5,axis=0)
            site=process(rk,np.ascontiguousarray(f[:128,:5]).reshape(-1,20,3),rz).reshape(128,5,4)
            assert np.isfinite(site).all()
            siteerr=np.abs(site[:,:,1:]-obs[:,None,1:])
            displacement=np.array([[site_displacement(ref["z"][i],f[i,j],ref["f"][i])
                                   for j in range(5)] for i in range(128)])
            np.savez_compressed(dest,ids=ref["ids"],shape_norm=shape,absolute_log_volume=np.abs(trace),
                placement_ids=ref["ids"][:128],site_environment_absolute_error=siteerr,site_assignment_rms=displacement)
        with np.load(dest) as d:
            row=dict(arm=base.name,shape_n=9046,shape_draws=20,site_n=128,site_draws=5,
                     mean_traceless_logstretch_norm=float(d["shape_norm"].mean()),
                     mean_absolute_log_volume_per_atom=float(d["absolute_log_volume"].mean()),
                     mean_fractional_site_assignment_rms=float(d["site_assignment_rms"].mean()))
            for i,p in enumerate(("minimum","median","q90")):
                row["mean_absolute_"+p+"_site_neighbor_A"]=float(d["site_environment_absolute_error"][:,:,i].mean())
            rows.append(row)
    if rows:
        with (HERE/"geometry.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print("Geometry diagnostic arms",len(rows))


if __name__=="__main__":main()
