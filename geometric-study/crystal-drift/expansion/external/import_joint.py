"""Import the 10,000 unrelaxed Table 3 CrystalFlow samples in deposited order."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"scaled"))
from prepare_mp20 import SYMBOLS, pack_gram


def main(path):
    payload = path.read_bytes()
    assert hashlib.md5(payload).hexdigest()=="6ed29bcd3389d43a3a36d645d0e65400"
    data = payload.decode().splitlines()
    atomic_numbers = {s:i for i,s in enumerate(SYMBOLS if isinstance(SYMBOLS,list) else SYMBOLS.split())}
    z=np.zeros((10000,20),dtype=np.int64); frac=np.zeros((10000,20,3),dtype=np.float64)
    cells=np.zeros((10000,3,3),dtype=np.float64)
    counts={"POSCAR":0,"CONTCAR":0}; missing=[]
    for j,line in enumerate(data):
        m=re.fullmatch(r"-+\s+(POSCAR|CONTCAR)\s+(\d+)\s+-+",line.strip())
        if not m:continue
        kind,idx=m[1],int(m[2])
        assert idx==counts[kind]
        counts[kind]+=1
        if kind=="CONTCAR":continue
        if data[j+1].startswith("----"):
            missing.append(idx)
            continue
        a=np.array([row.split() for row in data[j+3:j+6]],float)
        assert float(data[j+2])==1. and data[j+8].lower().strip()=="direct"
        elements=data[j+6].split()
        species=np.repeat([atomic_numbers[s] for s in elements],list(map(int,data[j+7].split())))
        n=len(species)
        xyz=np.array([row.split()[:3] for row in data[j+9:j+9+n]],float)
        order=np.argsort(species,kind="stable")
        z[idx,:n]=species[order]
        frac[idx,:n]=xyz[order]%1.
        cells[idx]=a
    assert counts=={"POSCAR":10000,"CONTCAR":10000} and np.isfinite(frac).all()
    present=z.any(axis=1)
    k=np.full((10000,6),np.nan)
    k[present]=pack_gram(cells[present]@cells[present].transpose(0,2,1))
    out=HERE/"crystalflow-joint.npz"
    np.savez_compressed(out,ids=[f"crystalflow-{i:05d}" for i in range(10000)],z=z,k=k,f=frac,lattice=cells,present=present)
    (HERE/"crystalflow-joint.json").write_text(json.dumps(dict(source_doi="10.6084/m9.figshare.29978608.v3",
        file_id=57393610,filename="Tab3.txt",sha256=hashlib.sha256(payload).hexdigest(),deposited_counts=counts,missing_raw_indices=missing,
        imported="All raw POSCAR 0..9999 ordered by source, fractional coordinates and species permuted together in ascending Z",
        output_sha256=hashlib.sha256(out.read_bytes()).hexdigest()),indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("path",type=Path)
    main(p.parse_args().path)
