"""Conservative lattice and site-subset exclusions for the pinned StructureMatcher.

Used only when the reduced target has as many sites as the native prediction.
A successful prediction must then be primitive at the same size. All native
integer unimodular frames are enumerated within the reciprocal-norm bounds.
The fractional-coordinate subset check is a necessary test that pymatgen
applies before any Cartesian matching. Passing always defers to pymatgen.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def nearest_possible(m, det, gen_f, gen_z, ref_f, ref_z, anchor, tolerances):
    n=len(gen_z)
    # The three cofactor cross products are the columns of the inverse.
    inv=np.empty((3,3),np.float64)
    for j in range(3):
        v=m[(j+1)%3];w=m[(j+2)%3]
        for k in range(3):
            inv[k,j]=(v[(k+1)%3]*w[(k+2)%3]-v[(k+2)%3]*w[(k+1)%3])/det
    aligned=gen_f@inv
    for a in range(n):
        if gen_z[a]!=ref_z[anchor]:continue
        shift=aligned[a]-ref_f[anchor]
        all_possible=True
        for b in range(n):
            possible=False
            for c in range(n):
                if gen_z[c]!=ref_z[b]:continue
                within=True
                for k in range(3):
                    v=ref_f[b,k]+shift[k]-aligned[c,k]
                    if abs(v-round(v))>tolerances[k]:
                        within=False
                        break
                if within:
                    possible=True
                    break
            if not possible:
                all_possible=False
                break
        if all_possible:return True
    return False


@njit(cache=True)
def dot3(a,g,b):
    s=0.
    for i in range(3):
        for j in range(3):s+=a[i]*g[i,j]*b[j]
    return s


@njit(cache=True)
def frame_possible(g, gen_f,gen_z, target_g,ref_f,ref_z,anchor,ltol=.3,angle_tol=10.):
    # Both volumes equal one; pymatgen's symmetric rescaling differs only by a
    # common scalar, which cancels in normalized distances and lattice tests.
    g=g/(np.linalg.det(g)**(1./3)); rg=target_g/(np.linalg.det(target_g)**(1./3))
    ref_len=np.sqrt(np.diag(rg)); upper=ref_len*(1+ltol)+1e-7;lower=ref_len/(1+ltol)-1e-7
    bounds=np.ceil(upper.max()*np.sqrt(np.diag(np.linalg.inv(g)))+1e-7).astype(np.int64)
    n_box=(2*bounds[0]+1)*(2*bounds[1]+1)*(2*bounds[2]+1)
    if n_box>30000:return True
    vv=np.empty((n_box,3),np.int64); length=np.empty(n_box,np.float64);n=0
    for i in range(-bounds[0],bounds[0]+1):
        for j in range(-bounds[1],bounds[1]+1):
            for k in range(-bounds[2],bounds[2]+1):
                v=vv[n];v[0]=i;v[1]=j;v[2]=k;val=math.sqrt(max(dot3(v,g,v),0.))
                if (lower[0]<=val<=upper[0]) or (lower[1]<=val<=upper[1]) or (lower[2]<=val<=upper[2]):
                    length[n]=val;n+=1
    if n<6:return False
    if n>360:return True
    rad=math.pi/180; tolerance=angle_tol*rad+1e-7
    ref_angles=np.empty(3)
    for a in range(3):
        b=(a+1)%3;c=(a+2)%3
        ref_angles[a]=math.acos(min(1.,max(-1.,rg[b,c]/(ref_len[b]*ref_len[c]))))
    # Inversion-related triples can yield different site maps, so keep both.
    m=np.empty((3,3),np.int64);angles=np.zeros(3);lens=np.zeros(3);tols=np.zeros(3)
    for a in range(n):
        if not lower[0]<=length[a]<=upper[0]:continue
        m[0]=vv[a]
        for b in range(n):
            if not lower[1]<=length[b]<=upper[1]:continue
            cg=math.acos(min(1.,max(-1.,dot3(vv[a],g,vv[b])/(length[a]*length[b]))))
            if abs(cg-ref_angles[2])>tolerance:continue
            m[1]=vv[b]
            for c in range(n):
                if not lower[2]<=length[c]<=upper[2]:continue
                m[2]=vv[c]
                det=(m[0,0]*(m[1,1]*m[2,2]-m[1,2]*m[2,1])
                    +m[0,1]*(m[1,2]*m[2,0]-m[1,0]*m[2,2])
                    +m[0,2]*(m[1,0]*m[2,1]-m[1,1]*m[2,0]))
                if abs(det)!=1:continue
                cb=math.acos(min(1.,max(-1.,dot3(vv[a],g,vv[c])/(length[a]*length[c]))))
                if abs(cb-ref_angles[1])>tolerance:continue
                ca=math.acos(min(1.,max(-1.,dot3(vv[b],g,vv[c])/(length[b]*length[c]))))
                if abs(ca-ref_angles[0])>tolerance:continue
                angles[0]=.5*(ca+ref_angles[0]);angles[1]=.5*(cb+ref_angles[1]);angles[2]=.5*(cg+ref_angles[2])
                lens[0]=.5*(length[a]+ref_len[0]);lens[1]=.5*(length[b]+ref_len[1]);lens[2]=.5*(length[c]+ref_len[2])
                p0,p1,p2=math.cos(angles[0]),math.cos(angles[1]),math.cos(angles[2])
                q=1.+2*p0*p1*p2-p0*p0-p1*p1-p2*p2
                if q<=1e-12:return True
                volume=lens[0]*lens[1]*lens[2]*math.sqrt(q)
                for d in range(3):tols[d]=math.sin(angles[d])/(lens[d]*math.sqrt(q))*(volume/len(ref_z))**(1./3)+2e-7
                if nearest_possible(m,det,gen_f,gen_z,ref_f,ref_z,anchor,tols):return True
    return False
