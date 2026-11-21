"""Score frozen native arrays against their own MP-20 targets, preserving every draw."""
import argparse
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[name]="1"

import hashlib
from contextlib import contextmanager
from importlib.metadata import version
from functools import lru_cache
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"revision/full"))
sys.path.insert(0,str(ROOT))
from match_full import candidate, target_context, MATCHER
from prescreen import frame_possible
from data import pack_gram
from pymatgen.core import Lattice, Structure

def origin_images(original,kwargs,minimum_condition=300.):
    """Enumerate a severely skewed origin lattice in an equivalent integer basis.

    Restore the pinned C routine's original images, distances, and visitation
    order, including its 1-Angstrom outer search for small requested radii.
    All other point/center queries retain the stock code path.
    """
    if not all(k in kwargs for k in ("r","all_coords","center_coords","lattice","pbc")):return None
    r=kwargs['r'];L=kwargs['lattice']
    if r<=0 or kwargs['all_coords'].shape!=(1,3) or kwargs['center_coords'].shape!=(1,3):return None
    if np.any(kwargs['all_coords']) or np.any(kwargs['center_coords']) or not np.all(kwargs['pbc']):return None
    condition=np.linalg.cond(L)
    if condition<minimum_condition or condition>1e7:return None
    mapping=Lattice(L).lll_mapping
    U=np.rint(mapping).astype(np.int64)
    if not np.array_equal(mapping,U):return None
    inv=np.rint(np.linalg.inv(U)).astype(np.int64)
    if not np.array_equal(U@inv,np.eye(3,dtype=np.int64)):return None
    B=np.ascontiguousarray(U@L)
    if np.linalg.cond(B)>min(100.,condition):return None
    tol=kwargs.get('tol',1e-8)
    # Expand the preliminary radius to absorb roundoff in the two integer-basis
    # products before rechecking every candidate in the original arithmetic.
    imax=1+np.linalg.norm(np.linalg.inv(B),ord=1)*(1+r)
    margin=1e-8+128*np.finfo(float).eps*imax*np.abs(U).sum()*max(1.,np.max(np.abs(L)))
    probe=dict(kwargs,lattice=B,r=np.sqrt(r*r+tol)+margin,min_r=0.)
    first,second,translated,_=original(**probe)
    images=np.rint(translated).astype(np.int64)@U
    cart=images[:,0,None]*L[0]+images[:,1,None]*L[1]+images[:,2,None]*L[2]
    d2=cart[:,0]**2+cart[:,1]**2+cart[:,2]**2
    distance=np.sqrt(d2)
    outer=max(r,1.)
    keep=(distance<=r if r<1. else d2<r*r+tol)&(np.abs(cart)<outer+tol).all(1)
    m=images[keep];d=distance[keep]
    # Neighbor cubes are visited lexicographically. Each cube's linked list
    # reverses the original lexicographic image insertion order.
    bins=np.floor((cart[keep]+outer+tol+1e-8)/max(.1,outer)).astype(np.int64)
    order=np.lexsort((-m[:,2],-m[:,1],-m[:,0],bins[:,2],bins[:,1],bins[:,0]))
    return first[keep][order],second[keep][order],m[order].astype(float),d[order]

@contextmanager
def tiny_sphere_search():
    """Use the native compiled sphere search at the requested radius; preserve its final distance mask."""
    from pymatgen.optimization import neighbors
    original=neighbors.find_points_in_spheres

    def requested_radius(*args,**kwargs):
        # Lattice.get_points_in_sphere calls exclusively by keyword in the pinned version.
        r=kwargs.get("r")
        if not args and 'min_r' not in kwargs:
            try:
                exact=origin_images(original,kwargs)
            except (np.linalg.LinAlgError,ValueError,FloatingPointError,OverflowError):
                exact=None
            if exact is not None:return exact
        if r is None or not 0<r<1 or "min_r" in kwargs:
            return original(*args,**kwargs)
        result=original(*args,**kwargs,min_r=0.)
        mask=result[-1]<=r
        return tuple(values[mask] for values in result)
    neighbors.find_points_in_spheres=requested_radius
    try:
        yield
    finally:
        neighbors.find_points_in_spheres=original

def tiny_cell(ctx,L,f,z,verify=False):
    """Use the same necessary scale-free frame test before the public tiny-cell path."""
    _,ref,ref_f,ref_z,anchor,ref_g=ctx
    possible=True
    if len(ref)==len(z):
        # Guard floating conditioning and integer enumeration bounds. Plates and needles
        # outside this numerical regime go straight to the public matcher.
        s=np.linalg.svd(L,compute_uv=False)
        if 0<s[-1] and s[0]/s[-1]<1e7:
            metric_basis=L
            fractional=f
            if s[0]/s[-1]>6:
                # Use an independently checked integer change of frame only in the
                # necessary test. Stock pymatgen still sees the original cell when forwarded.
                try:
                    mapping=Lattice(L).lll_mapping
                    U=np.rint(mapping).astype(np.int64)
                    inverse=np.rint(np.linalg.inv(U)).astype(np.int64)
                    if np.array_equal(mapping,U) and np.array_equal(U@inverse,np.eye(3,dtype=np.int64)):
                        metric_basis=U@L
                        fractional=(f@inverse)%1.
                except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError):
                    pass
            if np.linalg.cond(metric_basis)<1000:
                g=metric_basis@metric_basis.T
                possible=frame_possible(g,np.ascontiguousarray(fractional),z,ref_g,ref_f,ref_z,anchor)
    if possible or verify:
        pred=Structure(L,z,f)
        with tiny_sphere_search():
            hit=MATCHER.get_rms_dist(pred,ctx[0])
            strict=MATCHER.fit(pred,ctx[0])
        if not possible and (hit is not None or strict):
            raise RuntimeError("Tiny-cell lattice certificate disagrees with pymatgen")
        return np.nan if hit is None else hit[0],strict,True
    return np.nan,False,False

@lru_cache(maxsize=3)
def truth(split):
    with np.load(ROOT/f"scaled/mp20-data/{split}.npz") as raw:
        data={k:raw[k] for k in raw}
    return {str(v):i for i,v in enumerate(data["ids"])},data

def score(path,split,out,public=False,row_span=None):
    if version("pymatgen")!="2024.4.13":
        raise RuntimeError("Use the pinned 2024.4.13 matcher")
    path,out=Path(path),Path(out)
    fingerprint=hashlib.sha256(path.read_bytes()).hexdigest()
    if out.exists():
        with np.load(out) as done:
            cached_span=tuple(done["row_span"].tolist()) if "row_span" in done else ()
            if cached_span==tuple(row_span or ()) and str(done["source_sha"])==fingerprint and str(done["split"])==split and (not public or bool(done["public_all"])):
                return
    with np.load(path,allow_pickle=False) as pred:
        scope=slice(*row_span) if row_span is not None else slice(None)
        ids=pred["ids"][scope].astype(str)
        cells,fs,zs=(pred[k][scope] for k in ("L","F","Z"))
    n,k=cells.shape[:2]
    assert len(ids)==n==len(set(ids)) and cells.shape==(n,k,3,3) and fs.shape[:2]==(n,k)
    if zs.ndim==2: zs=np.broadcast_to(zs[:,None,:],fs.shape[:-1])
    assert zs.shape==fs.shape[:-1]
    rms=np.full((n,k),np.nan)
    vol=rms.copy()
    strict=np.zeros((n,k),bool)
    error=np.zeros((n,k),np.int8)
    formal=checks=0
    lookup,data=truth(split)
    start=time.perf_counter()
    for i,mid in enumerate(ids):
        ix=lookup[mid]; rz=data["z"][ix]
        ctx=target_context(data["k"][ix],data["f"][ix],rz)
        for j in range(k):
            z=zs[i,j]; mask=z>0; z=np.ascontiguousarray(z[mask],dtype=np.int64)
            f=fs[i,j][mask].astype(float)
            L=cells[i,j].astype(float)
            if not np.isfinite(L).all() or not np.isfinite(f).all() or not np.array_equal(np.sort(z),np.sort(rz[rz>0])):
                error[i,j]=1;continue
            try:
                v=abs(np.linalg.det(L))
                if not np.isfinite(v) or v<=0: raise ValueError("degenerate cell")
                vol[i,j]=np.log(v)-np.sum(data["k"][ix,:3],dtype=float)
                verify=public or ((ix*104729+j*1543)%3001==0)
                if v<=0.100001:
                    rms[i,j],strict[i,j],sent=tiny_cell(ctx,L,f,z,verify)
                    formal+=int(sent);checks+=int(sent)
                    continue
                G=L@L.T; chart=pack_gram(G)
                rms[i,j],strict[i,j],sent=candidate(ctx,chart,f,z,G,verify)
                checks+=int(verify);formal+=int(sent)
            except MemoryError:
                error[i,j]=3
            except (FloatingPointError,OverflowError,ValueError,np.linalg.LinAlgError):
                error[i,j]=2
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix(f".{os.getpid()}.building")
    with tmp.open("wb") as fp:
        np.savez_compressed(fp,ids=ids,rms=rms,strict_fit=strict,log_volume_ratio=vol,error=error,
            source_sha=fingerprint,split=split,row_span=np.array(row_span or (),dtype=int),public_checks=checks,public_all=public,forwarded=formal,
            seconds=time.perf_counter()-start,pymatgen=version("pymatgen"))
    tmp.replace(out)
    print(f"{out.name} {np.isfinite(rms).any(1).sum()}/{n} at {k} draws, {error.astype(bool).sum()} failed draws",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("input",type=Path)
    ap.add_argument("--split",choices=["train","val","test"],required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--public",action="store_true",help="Cross-check every outcome through public API.")
    args=ap.parse_args()
    score(args.input,args.split,args.output,args.public)
