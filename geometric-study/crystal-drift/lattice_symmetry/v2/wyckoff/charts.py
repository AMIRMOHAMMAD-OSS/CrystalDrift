"""Conventional affine charts, canonical Hall operations and deterministic round-trip checks."""
import itertools
import string
from functools import lru_cache
from pathlib import Path

import numpy as np
import spglib
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
with np.load(HERE/"charts.npz") as db:
    MATRICES = db["matrices"].astype(float)
    TRANSLATIONS = db["t24"].astype(float)/24
    ORBITS = db["positions"]; GROUPS = db["groups"]; HALLS = db["halls"]
LETTERS = string.ascii_lowercase+string.ascii_uppercase
SHIFTS = np.array(list(itertools.product([0,-1,1],repeat=3)))

@lru_cache(None)
def chart(group,letter):
    p = GROUPS[group-1]+LETTERS.index(letter)
    if p >= GROUPS[group]: raise ValueError((group,letter))
    sl = slice(ORBITS[p],ORBITS[p+1])
    m,t = MATRICES[sl],TRANSLATIONS[sl]
    free = np.any(abs(m[0])>0, axis=1)
    assert free.sum()==np.linalg.matrix_rank(m[0])
    return m,t,free

@lru_cache(None)
def operations(group):
    d = spglib.get_symmetry_from_database(int(HALLS[group-1]))
    return np.transpose(d["rotations"],(0,2,1)), d["translations"]

def xyz(group,letter,u):
    m,t,free = chart(group,letter)
    return (np.einsum("d,edj->ej",u,m)+t)%1.

def fit_orbit(group,letter,sites,cell):
    m,t,_ = chart(group,letter)
    if len(m)!=len(sites): raise ValueError("multiplicity")
    # Try a representative on each torus branch, compare the entire unordered orbit.
    inv = np.linalg.pinv(m[0])
    offset = sites[:,None,:]-t[0]+SHIFTS
    candidates = (offset @ inv).reshape(-1,3)
    misfit = np.abs((candidates@m[0]+t[0]-np.repeat(sites,len(SHIFTS),axis=0)+.5)%1-.5).max(axis=1)
    best = float("inf")
    for i in np.argsort(misfit,kind="stable")[:min(30,len(candidates))]:
        u = candidates[i]%1
        expanded = xyz(group,letter,u)
        delta = (expanded[:,None]-sites[None,:]+.5)%1-.5
        distances = np.linalg.norm(delta@cell,axis=-1)
        a,b=linear_sum_assignment(distances)
        err=distances[a,b].max()
        if err<best: best,bestu = err,u
        if best<1e-7: break
    return bestu,best

def label(cell,frac,z,tolerance=.01):
    native = (cell,frac,z)
    native_data = spglib.get_symmetry_dataset(native,symprec=tolerance)
    if native_data is None: raise ValueError("spglib-native")
    number=int(native_data.number)
    hall=int(HALLS[number-1])
    if native_data.hall_number != hall:
        native_data=spglib.get_symmetry_dataset(native,symprec=tolerance,hall_number=hall)
    if native_data is None: raise ValueError("spglib-forced-Hall")
    d=native_data
    scell=d.std_lattice
    spoints=d.std_positions
    sz=d.std_types
    # Both maps index the same primitive sites. Use the crystallographic orbit,
    # which also represents symmetries absent from a nonprimitive input setting.
    indices = np.array([np.flatnonzero(d.mapping_to_primitive==p)[0] for p in d.std_mapping_to_primitive])
    if not np.array_equal(sz,z[indices]):raise ValueError("primitive-species-map")
    owners=d.crystallographic_orbits[indices]
    specs=[];us=[];errs=[]
    for i in np.unique(owners):
        mask=owners==i; letter=d.wyckoffs[i]; atom=int(z[i])
        if not np.all(sz[mask]==atom): raise ValueError("mixed orbit")
        u,err=fit_orbit(number,letter,spoints[mask],scell)
        specs.append((atom,letter));us.append(u.tolist());errs.append(float(err))
    if max(errs)>.025:raise ValueError("chart-reconstruction")
    rot,_=operations(number);G=scell@scell.T
    gramerr=np.abs(rot@G@rot.transpose(0,2,1)-G).max()/np.abs(G).max()
    if gramerr>1e-7: raise ValueError("cell-metric")
    scale=len(sz)/len(z)
    if not np.allclose(np.bincount(sz,minlength=119),scale*np.bincount(z,minlength=119)):raise ValueError("stoichiometry")
    if abs(abs(np.linalg.det(scell))/abs(np.linalg.det(cell))/scale-1)>.02: raise ValueError("volume ratio")
    ix=sorted(range(len(specs)),key=lambda j:specs[j])
    return dict(group=number,hall=hall,template=[specs[j] for j in ix],
                u=[us[j] for j in ix],cell=scell.tolist(),std_n=len(sz),native_n=len(z),primitive_n=len(set(d.mapping_to_primitive)),
                native_to_primitive=d.mapping_to_primitive.tolist(),std_to_primitive=d.std_mapping_to_primitive.tolist(),
                err=max(errs),gramerr=float(gramerr),transformation=d.transformation_matrix.tolist(),origin=d.origin_shift.tolist())

def expand(group,template,u):
    arrays=[xyz(group,letter,point) for (_,letter),point in zip(template,u)]
    return np.concatenate(arrays),np.repeat([z for z,letter in template],[len(x) for x in arrays])

def generic_draw(group,template,rng):
    prev=[]
    for atom,letter in template:
        for trial in range(32):
            u=rng.random(3); orbit=xyz(group,letter,u)
            compare=np.concatenate([orbit]+[q for _,q in prev])
            delta=(orbit[:,None]-compare[None,:]+.5)%1-.5
            if (abs(delta).max(-1)<1e-7).sum()==len(orbit):break
        else:raise ValueError("collision or specialization after 32 draws")
        prev.append((u,orbit))
    return np.array([u for u,_ in prev])

def compatible_metric(group,G):
    rot,_=operations(group)
    g=(rot@G@rot.transpose(0,2,1)).mean(0)
    return .5*(g+g.T)
