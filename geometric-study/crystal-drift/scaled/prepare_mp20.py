"""Export all released MP-20 structures in their published splits and P1 basis."""
import argparse
import bz2
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = 'https://public.openmaterialsdb.se/wyckoffdiff/data/mp20/raw'
SHA = {'train':'aa9ccf4cf83a58e2aba3492e12f0eaf75bdf2946a7c90fa5bbeea94d731b721a',
       'val':'6b73a81761a74ccf19eb066d4e2cc39a3aae9a0d149d61bd0f966fbf6584a230',
       'test':'716a2dfef434e03e43ac94d195729d4fd8ee9677f3fd11918bb2895e6c8be13a'}
SYMBOLS = 'X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'.split()
Z = {symbol: number for number,symbol in enumerate(SYMBOLS)}


def pack_gram(g):
    eig,q = np.linalg.eigh(g)
    if np.any(eig <= 0):
        raise ValueError('Nonpositive lattice eigenvalue')
    s = (q*(np.log(eig)/2)[...,None,:]) @ q.swapaxes(-1,-2)
    return np.stack([s[...,0,0],s[...,1,1],s[...,2,2],s[...,0,1]*np.sqrt(2),
                     s[...,0,2]*np.sqrt(2),s[...,1,2]*np.sqrt(2)],axis=-1)


def parse_cif(cif):
    head,marker,atoms = cif.partition('_atom_site_occupancy\n')
    operations = head.partition('_symmetry_equiv_pos_as_xyz\n')[2].partition('loop_\n')[0].strip()
    if not marker or '_symmetry_Int_Tables_number   1\n' not in head or operations != "1  'x, y, z'":
        raise ValueError('Expected explicit P1 atom loop')
    sites = [line.split() for line in atoms.strip().splitlines()]
    if not 1 <= len(sites) <= 20 or any(len(s)!=7 or float(s[2])!=1 or float(s[6])!=1 for s in sites):
        raise ValueError('Expected 1–20 full-occupancy, unit-multiplicity sites')
    sites.sort(key=lambda site: Z[site[0]])
    z = np.zeros(20,dtype=np.int16)
    f = np.zeros((20,3),dtype=np.float32)
    z[:len(sites)] = [Z[s[0]] for s in sites]
    f[:len(sites)] = np.array([[float(x) for x in s[3:6]] for s in sites]) % 1
    fields = dict(row.split(maxsplit=1) for row in head.splitlines() if row.startswith('_cell_'))
    lengths = np.array([float(fields['_cell_length_'+a]) for a in 'abc'])
    ca,cb,cg = np.cos(np.deg2rad([float(fields['_cell_angle_'+a]) for a in ('alpha','beta','gamma')]))
    gram = np.outer(lengths,lengths)*np.array([[1,cg,cb],[cg,1,ca],[cb,ca,1]])
    if not np.isclose(np.linalg.det(gram)**0.5,float(fields['_cell_volume']),rtol=2e-6,atol=1e-5):
        raise ValueError('CIF volume mismatch')
    return z,f,gram


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=HERE/'raw')
    p.add_argument('--out',type=Path,default=HERE/'mp20-data')
    p.add_argument('--download',action='store_true')
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    report={}; encountered=set()
    for split,digest in SHA.items():
        name=f'mp20_{split}.csv.bz2'; url=f'{BASE}/{name}'
        choices=[args.raw/name,HERE.parent/'data'/name,HERE.parents[1]/'crystal'/'data'/name]
        path=next((x for x in choices if x.is_file()),None)
        if path is None:
            if not args.download:raise FileNotFoundError(f'Missing {name}; use --download or --raw')
            args.raw.mkdir(parents=True,exist_ok=True);path=choices[0]
            payload=urlopen(url,timeout=120).read()
            if hashlib.sha256(payload).hexdigest()!=digest:raise ValueError('Download SHA256 mismatch')
            path.write_bytes(payload)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError(f'Unexpected release {path}')
        ids=[]; zs=[]; fs=[]; grams=[]
        with bz2.open(path,'rt',encoding='utf8',newline='') as stream:
            for row in csv.DictReader(stream):
                mid=row['material_id']
                if mid in encountered:raise ValueError(f'Duplicate/overlapping ID {mid}')
                encountered.add(mid);ids.append(mid)
                try:z,f,g=parse_cif(row['cif'])
                except ValueError as e:raise ValueError(mid) from e
                zs.append(z);fs.append(f);grams.append(g)
        z=np.stack(zs);k=pack_gram(np.stack(grams)); f=np.stack(fs)
        destination=args.out/(split+'.npz')
        np.savez_compressed(destination,ids=np.array(ids),z=z,f=f,k=k.astype(np.float32))
        counts=Counter(len(set(row[row>0])) for row in z)
        report[split]=dict(rows=len(ids),source=url,source_sha256=digest,
                           processed_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                           by_species_count=dict(sorted(counts.items())),max_sites=int((z>0).sum(1).max()))
        print(split,report[split],flush=True)
    (args.out/'manifest.json').write_text(json.dumps({'splits':report,'id_overlap':0,
        'selection':'All released ordered P1 unit cells in their official splits, with 1–20 atoms inclusive.'},indent=2)+'\n')


if __name__=='__main__':main()
