"""Common raw template metrics, order-preserving CIF geometry I/O and matching."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import spglib
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice,Structure
from scipy.optimize import linear_sum_assignment
from charts import HERE,operations,chart
from labels import SETTINGS,dataset

TOLS=(.001,.01,.1)

def cohort(split):
    return json.loads((HERE/f"{split}-cohort.json").read_text())

def reference(row,split):
    _,L,F,Z=dataset(split);i=row["index"];mask=Z[i]>0
    return Structure(L[i],Z[i,mask],F[i,mask])

@lru_cache(None)
def rotation_classes(requested):
    rot,tr=operations(requested)
    return [(r,tr[(rot==r).all((1,2))]) for r in np.unique(rot,axis=0)]

def subgroup(d,cell,requested,origin,tol):
    if d is None:return False
    od=d.rotations.transpose(0,2,1);td=d.translations
    wrapped_is_complete=tol*np.linalg.norm(np.linalg.inv(cell),axis=0).max()<.5
    for r,ts in rotation_classes(requested):
        ix=(od==r).all(axis=(1,2))
        if ix.sum()<len(ts):return False
        trans=ts+origin-origin@r
        delta=(td[ix][None,:]-trans[:,None]+.5)%1-.5
        distance=np.linalg.norm(delta@cell,axis=-1) if wrapped_is_complete else Lattice(cell).get_all_distances(trans,td[ix])
        feasible=distance<=tol
        if len(ts)==1:
            if not feasible.any():return False
        else:
            a,b=linear_sum_assignment(~feasible)
            if not feasible[a,b].all():return False
    return True

def quality(L,F,Z,templ,matching=None,start=None):
    g=templ["group"];s=Structure(L,Z,F);dist=s.distance_matrix
    from ase.geometry import minkowski_reduce
    reduced,_=minkowski_reduce(L)
    np.fill_diagonal(dist,np.linalg.norm(reduced,axis=1).min())
    out=dict(valid=bool(dist.min()>=.5 and s.volume>=.1),dmin=float(dist.min()))
    if matching is not None:
        value=StructureMatcher(**SETTINGS).get_rms_dist(matching,s)
        out["rms"]=None if value is None else float(value[0])
    n_expected=[len(chart(g,l)[0]) for _,l in templ["template"]]
    atom_expected=np.repeat([int(z) for z,_ in templ["template"]],n_expected)
    # All our geometry arrays keep the coset order. Self-matching a literal group
    # in the initial origin separately checks a given orbit's atom identities.
    exact_species=np.array_equal(Z,atom_expected)
    origin=np.zeros(3) if start is None else np.mean((F-start+.5)%1-.5,axis=0)
    r,t=operations(g)
    q=np.linalg.inv(L)@r@L
    metric_residual=float(np.abs(np.linalg.svd(q,compute_uv=False)-1).max()*np.linalg.norm(L,axis=1).max())
    orbit_residual=0.;st=0
    for n in n_expected:
        images=((F[st]-origin)@r+t+origin)%1
        delta=(images[:,None]-F[None,st:st+n]+.5)%1-.5
        distances=np.linalg.norm(delta@L,axis=-1)
        orbit_residual=max(orbit_residual,float(distances.min(0).max()),float(distances.min(1).max()))
        st+=n
    out.update(metric_residual=metric_residual,orbit_residual=orbit_residual)
    for tol in TOLS:
        d=spglib.get_symmetry_dataset((L,F,Z),symprec=tol)
        code=str(tol)
        contains=subgroup(d,L,g,origin,tol)
        out["contains"+code]=contains
        out["number"+code]=0 if d is None else int(d.number)
        out["exact"+code]=bool(d is not None and d.number==g)
        out["requested_order"]=len(r)
        out["detected_order"+code]=len(d.rotations) if d is not None else 0
        out["exactops"+code]=bool(contains and out["detected_order"+code]==len(r))
        # At an enhanced group equivalent_atoms may merge supplied orbits, so
        # verify occupancy under the requested subgroup on labelled seed slots.
        out["occupancy"+code]=bool(exact_species and orbit_residual<=tol and out["dmin"]>tol)
        out["direct"+code]=bool(exact_species and max(metric_residual,orbit_residual)<=tol)
    return out
