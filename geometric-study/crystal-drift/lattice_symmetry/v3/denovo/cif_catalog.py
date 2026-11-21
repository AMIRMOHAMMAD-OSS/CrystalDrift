"""Export every ordered attempt, retaining metric and site provenance, and detect raw space groups."""
import argparse
from collections import Counter
import csv
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import sys
import warnings
from zipfile import ZipFile, ZIP_DEFLATED, ZipInfo

for thread_variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[thread_variable]="1"
import numpy as np
import spglib
from pymatgen.io.cif import CifParser, CifFile
from scipy.optimize import linear_sum_assignment

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from data import ELEMENTS, Z, unpack_gram


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda:fp.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def cif(arm,label,g,f,z):
    lengths=np.sqrt(g.diagonal())
    cos=g/np.outer(lengths,lengths)
    angles=np.degrees(np.arccos(np.clip(cos[[1,0,0],[2,2,1]],-1,1)))
    lines=[f"data_{arm}_{label}","_symmetry_space_group_name_H-M 'P 1'",
           "_symmetry_Int_Tables_number 1"]
    for key,val in zip(("length_a","length_b","length_c","angle_alpha","angle_beta","angle_gamma"),[*lengths,*angles]):
        lines.append(f"_cell_{key} {val:.15g}")
    lines+=["loop_","_symmetry_equiv_pos_as_xyz","'x,y,z'","loop_",
            "_atom_site_type_symbol","_atom_site_label","_atom_site_fract_x","_atom_site_fract_y",
            "_atom_site_fract_z","_atom_site_occupancy"]
    for i,(e,row) in enumerate(zip(z,f)):
        symbol=ELEMENTS[e]
        coord=" ".join(f"{x:.15g}" for x in row%1)
        lines.append(f"{symbol} {symbol}{i+1} {coord} 1")
    return "\n".join(lines)+"\n"


def put(archive,name,body):
    record=ZipInfo(name,date_time=(1980,1,1,0,0,0))
    record.compress_type=ZIP_DEFLATED
    archive.writestr(record,body,compresslevel=5)


def read_back(body,g,f,z):
    # Check the CIF itself, then exercise a separate structure parser and same-species assignment.
    block=next(iter(CifFile.from_str(body).data.values())).data
    assert int(block["_symmetry_Int_Tables_number"])==1 and block["_symmetry_equiv_pos_as_xyz"]==["x,y,z"]
    rawz=[Z[e] for e in block["_atom_site_type_symbol"]]
    assert np.array_equal(rawz,z)
    x=np.array([block["_atom_site_fract_"+v] for v in "xyz"],dtype=float).T
    delta=(x-f+.5)%1-.5
    assert np.max(abs(delta))<1e-11
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s=CifParser.from_str(body,frac_tolerance=0,site_tolerance=1e-10).parse_structures(primitive=False)[0]
    P=s.lattice.metric_tensor
    metric=float(abs(P-g).max()/max(1.,abs(g).max()))
    ez=np.array(s.atomic_numbers)
    assert np.array_equal(np.sort(z),np.sort(ez))
    pos=0.
    for e in np.unique(z):
        df=(s.frac_coords[ez==e,None,:]-f[None,z==e,:]+.5)%1-.5
        cost=abs(df).max(axis=-1);i,j=linear_sum_assignment(cost)
        pos=max(pos,float(cost[i,j].max()))
    assert metric<2e-12 and pos<2e-8,(metric,pos)
    return metric,pos


def export(arm,output):
    source=HERE/f"pools/{arm}.npz"
    checksum=sha(source); original=json.loads(source.with_suffix(".json").read_text())
    assert original["sha"]==checksum
    with np.load(source) as d:
        ids,k,f,z,valid=(d[key] for key in ("ids","k","f","z","valid_output"))
        def field(key,fallback):
            return d[key] if key in d else np.full(len(ids),fallback)
        req=field("group",0);don=field("donor","");cell=field("cell_donor","");chem=field("chemistry_source","")
    n=len(ids);assert n==10000 and len(set(ids))==n
    out=HERE/"catalog";out.mkdir(exist_ok=True)
    dest=output/f"{arm}-ordered-cifs.zip";tmp=dest.with_suffix(".part")
    meta=out/f"{arm}.json"
    if meta.exists() and dest.exists():
        before=json.loads(meta.read_text())
        if before.get("source_sha256")==checksum and before["archive_sha256"]==sha(dest):
            print("Already verified",arm,flush=True);return
    rows=[]; groups=np.zeros(n,np.int16);halls=np.zeros(n,np.int16)
    export_count=0;matched_types=0;err=Counter()
    output.mkdir(parents=True,exist_ok=True)
    with ZipFile(tmp,"w",allowZip64=True) as arc:
        for i,mid in enumerate(ids.astype(str)):
            mask=z[i]>0;zs=z[i,mask];fr=f[i,mask].astype(float);requested=int(req[i])
            exception="";detection="";entry=""
            if not valid[i]:
                exception="raw generator marked invalid"
            elif not mask.any() or not np.isfinite(k[i]).all() or not np.isfinite(fr).all():
                exception="empty or nonfinite raw arrays"
            else:
                try:
                    G=unpack_gram(k[i]);L=np.linalg.cholesky(G)
                    body=cif(arm,mid,G,fr,zs)
                except (FloatingPointError, ValueError, np.linalg.LinAlgError) as ex:
                    exception=f"{type(ex).__name__} {ex}"
                else:
                    entry=f"cifs/{mid}.cif"
                    put(arc,entry,body);export_count+=1
                    try:
                        sg=spglib.get_symmetry_dataset((L,fr,zs),symprec=.01,angle_tolerance=-1.)
                        if sg is None:detection=str(spglib.get_error_message())
                        else:
                            groups[i]=sg.number;halls[i]=sg.hall_number
                            matched_types+=int(requested>0 and requested==sg.number)
                    except (ValueError,RuntimeError) as ex:
                        detection=f"{type(ex).__name__} {ex}"
            if exception:err["CIF: "+exception]+=1
            if detection:err["symmetry: "+detection]+=1
            rows.append((i+1,mid,entry,exception,detection,requested,int(groups[i]),int(halls[i]),chem[i],don[i],cell[i]))
        s=io.StringIO(newline="")
        writer=csv.writer(s,lineterminator="\n")
        writer.writerow(["order","id","cif","export_error","detection_error","requested_group_if_any",
            "detected_group_at_0p01_A","detected_hall","composition_source","pattern_donor","cell_donor"])
        writer.writerows(rows)
        put(arc,"attempts.csv",s.getvalue())
        put(arc,"README.txt","One row per original attempt, in original order. Each emitted CIF uses explicit P1 sites, "
            "retains the input cell metric and wraps fractional rows into [0,1). Detected groups are independent, "
            "as-generated spglib assignments at 0.01 angstrom using automatic angle tolerance. Zero encodes unavailable. "
            "Geometric validity or a successful symmetry assignment is not a stability assessment.\n"
            f"Raw input SHA-256 {checksum}\n")
    checks=[]
    chosen=sorted((i for i,r in enumerate(rows) if r[2]),key=lambda i:hashlib.sha256(("v3-cif-read/"+str(ids[i])).encode()).digest())[:32]
    with ZipFile(tmp) as arc:
        assert len(arc.namelist())==export_count+2 and arc.testzip() is None
        for i in chosen:
            mask=z[i]>0;mid=ids[i]
            metric,position=read_back(arc.read(rows[i][2]).decode(),unpack_gram(k[i]),f[i,mask].astype(float),z[i,mask])
            checks.append(dict(id=str(mid),metric_relative=metric,fractional_max=position))
    tmp.replace(dest)
    def counts(x):
        return {str(int(key)):int(num) for key,num in zip(*np.unique(x,return_counts=True))}
    result=dict(arm=arm,source_sha256=checksum,archive_sha256=sha(dest),archive=str(dest),
        bytes=dest.stat().st_size,attempts=n,cifs=export_count,failed_cif=n-export_count,detected=int(np.count_nonzero(groups)),
        groups=counts(groups),requested=counts(req),matching_requested_types=matched_types,
        spglib=version("spglib"),pymatgen=version("pymatgen"),symprec_A=.01,angle_tolerance=-1.,
        errors=err,read_back=checks,crc_verified=True)
    np.savez_compressed(out/f"{arm}-spacegroups.npz",ids=ids,requested=req,detected=groups,hall=halls)
    meta.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(arm,export_count,"CIFs;",result["detected"],"group assignments;",len(checks),"read-backs",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms",nargs="+")
    parser.add_argument("--output",type=Path,default=HERE/"catalog")
    a=parser.parse_args()
    assert version("spglib")=="2.5.0" and version("pymatgen")=="2024.4.13"
    for arm in a.arms:export(arm,a.output)
