"""Detect the requested native-basis subgroup for the prospectively selected relaxation panel."""
import argparse
import hashlib
from pathlib import Path
from importlib.metadata import version
import json
import numpy as np
import spglib
from pymatgen.core import Lattice
from scipy.optimize import linear_sum_assignment
from data import unpack_gram

HERE=Path(__file__).resolve().parent;D=HERE.parent/"denovo";OUT=HERE/"audits"
with np.load(HERE/"native-operations.npz") as db:
    ROWS,OFF,BASE,HALL=(db[k] for k in ("rows","offsets","bases","halls"))


def requested(g):
    rows=ROWS[OFF[g-1]:OFF[g]]
    return rows[:,:9].reshape(-1,3,3),rows[:,9:]/24


def crosscheck_table():
    for g in range(1,231):
        A=BASE[g-1];inv=np.linalg.inv(A)
        s=spglib.get_symmetry_from_database(int(HALL[g-1]))
        r=A @ s["rotations"].transpose(0,2,1) @ inv
        t=s["translations"]@inv
        table=np.unique(np.column_stack([np.rint(r).reshape(-1,9),np.rint(t*24)%24]),axis=0)
        assert np.array_equal(table,ROWS[OFF[g-1]:OFF[g]]),(g,table,ROWS[OFF[g-1]:OFF[g]])


def group_check(L,F,Z,g,origin,tol):
    rot,tr=requested(g);d=spglib.get_symmetry_dataset((L,F,Z),symprec=tol)
    if d is None:return dict(detected=0,contains=False,exact_operations=False,exact_type=False)
    observed=d.rotations.transpose(0,2,1);trans=d.translations;contains=True
    for r in np.unique(rot,axis=0):
        expect=tr[(rot==r).all(axis=(1,2))]+origin-origin@r
        avail=trans[(observed==r).all(axis=(1,2))]
        if len(avail)<len(expect):contains=False;break
        dist=Lattice(L).get_all_distances(expect,avail)
        ix,jy=linear_sum_assignment(dist>tol)
        if np.any(dist[ix,jy]>tol):contains=False;break
    return dict(detected=int(d.number),contains=contains,
        exact_operations=bool(contains and len(observed)==len(rot)),exact_type=int(d.number)==g,
        detected_order=len(observed),requested_order=len(rot))


def main():
    p=argparse.ArgumentParser();p.add_argument("arms",nargs="+");arg=p.parse_args()
    assert version("spglib")=="2.5.0";crosscheck_table();OUT.mkdir(exist_ok=True)
    for arm in arg.arms:
        records=[];panel=json.loads((D/"screen/panel.json").read_text())
        with np.load(D/f"pools/{arm}.npz") as data:
            kk,ff,zz,g=(data[k] for k in ("k","f","z","group"))
        # Include a larger prespecified hash sample before consulting any symmetry outcomes.
        extra_ix=sorted(range(len(g)),key=lambda i:hashlib.sha256(("v3-group/"+
                    str(i)).encode()).digest())[:256]
        extra=[]
        for i in extra_ix:
            mask=zz[i]>0
            if g[i]>0 and mask.any() and np.isfinite(kk[i]).all():
                answer=group_check(np.linalg.cholesky(unpack_gram(kk[i])),ff[i,mask].astype(float),
                                   zz[i,mask],int(g[i]),np.zeros(3),.01)
            else:
                answer=dict(detected=0,contains=False,exact_operations=False,exact_type=False)
            extra.append(dict(index=i,requested=int(g[i]),**answer))
        for i in panel["indices_zero_based"]:
            assert g[i]>0
            mask=zz[i]>0;z=zz[i,mask];f=ff[i,mask].astype(float);l=np.linalg.cholesky(unpack_gram(kk[i]))
            name=f"{arm}-{i+1:05}";r=json.loads((D/f"screen/records/{name}.json").read_text())
            row=dict(index=i,id=name,group=int(g[i]),relaxation_status=r["status"])
            for tol in (.001,.01,.1):
                raw=group_check(l,f,z,int(g[i]),np.zeros(3),tol)
                if "final_f" in r:
                    final=np.asarray(r["final_f"]);L=np.linalg.cholesky(unpack_gram(r["final_k"]))
                    t=((final-f+.5)%1-.5).mean(0)
                    relaxed=group_check(L,final,z,int(g[i]),t,tol)
                else:relaxed=dict(detected=0,contains=False,exact_operations=False,exact_type=False)
                row[f"{tol:g}"]=dict(raw=raw,relaxed=relaxed)
            records.append(row)
        summary={str(tol):{phase:{k:sum(r[f"{tol:g}"][phase][k] for r in records)
            for k in ("contains","exact_operations","exact_type")} for phase in ("raw","relaxed")}
            for tol in (.001,.01,.1)}
        (OUT/f"subgroup-{arm}.json").write_text(json.dumps(dict(arm=arm,N=len(records),
            native_tables_verified_against_spglib_hall=230,spglib=version("spglib"),summary=summary,
            unrelaxed_hash256_summary={k:sum(r[k] for r in extra) for k in ("contains","exact_operations","exact_type")},
            unrelaxed_hash256=extra,
            records=records),indent=2)+"\n",encoding="utf-8")
        print(arm,summary["0.01"])

if __name__=="__main__":main()
