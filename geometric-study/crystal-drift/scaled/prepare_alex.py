"""Read the pinned MatterGen Alex-MP-20 CIF splits, retaining the deposited cell basis."""
import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

import numpy as np

from prepare_mp20 import Z,pack_gram,parse_cif


HERE=Path(__file__).resolve().parent
COMMIT='92423660a8bd70e83679086e88f88596d484dc16'
URL=f'https://media.githubusercontent.com/media/microsoft/mattergen/{COMMIT}/data-release/alex-mp/alex_mp_20.zip'
SOURCE_SHA='6688e652e45fac4ac8b016dbc4a3280afde9ba687bdacfd8c676775ed19b6228'
BYTES=186506244
COUNTS={'train':607683,'val':67521}
SITE_CAP=20


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as fp:
        for chunk in iter(lambda:fp.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def download(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix('.downloading')
    with urlopen(URL,timeout=120) as response, partial.open('wb') as output:
        for chunk in iter(lambda:response.read(1024*1024),b''):output.write(chunk)
    if partial.stat().st_size!=BYTES or sha(partial)!=SOURCE_SHA:raise ValueError('Downloaded archive checksum mismatch')
    partial.replace(path)


def alias_group(mid):
    if mid.startswith('alex<') and mid.endswith('>'):return 'alex'
    if mid.startswith('mp-'):return 'mp'
    if mid.startswith('mvc-'):return 'mvc'
    return 'other'


def parse_structure(cif):
    explicit='_symmetry_Int_Tables_number   1\n' in cif and "1  'x, y, z'" in cif
    if explicit and len(cif.partition('_atom_site_occupancy\n')[2].strip().splitlines())<=SITE_CAP:
        return (*parse_cif(cif),'explicit_p1',1)
    # This pathway expands symmetry operations in the original CIF conventional cell.
    from pymatgen.io.cif import CifParser
    parser=CifParser.from_str(cif,frac_tolerance=0)
    structures=parser.parse_structures(primitive=False)
    if len(structures)!=1:raise ValueError(f'Expected one structure in CIF; found {len(structures)}')
    s=structures[0]
    if not s.is_ordered or not len(s):raise ValueError('Disordered or empty structure')
    if any(len(site.species)!=1 or not np.isclose(sum(site.species.values()),1) for site in s):
        raise ValueError('Incomplete or mixed occupancy')
    basis='symmetry_expanded';factor=1
    if len(s)>SITE_CAP:
        from pymatgen.analysis.structure_matcher import StructureMatcher
        prim=s.get_primitive_structure(tolerance=.001)
        if len(prim)>SITE_CAP or len(s)%len(prim):raise ValueError('Primitive structure outside supported 1–20 site range')
        transform=s.lattice.matrix@np.linalg.inv(prim.lattice.matrix)
        integer=np.rint(transform).astype(int)
        if not np.allclose(integer,transform,atol=2e-6,rtol=0):raise ValueError('Noninteger supercell transform')
        factor=len(s)//len(prim)
        if abs(round(np.linalg.det(integer)))!=factor:raise ValueError('Supercell determinant mismatch')
        reconstructed=prim.copy();reconstructed.make_supercell(integer)
        strict=StructureMatcher(ltol=1e-5,stol=.001,angle_tol=.001,scale=False,primitive_cell=False)
        if not strict.fit(reconstructed,s):raise ValueError('Reconstructed supercell does not match deposited CIF')
        s=prim;basis='verified_primitive_reduction'
    order=np.argsort(s.atomic_numbers,kind='stable')
    z=np.zeros(SITE_CAP,np.int16);f=np.zeros((SITE_CAP,3),np.float32)
    z[:len(s)]=np.asarray(s.atomic_numbers)[order]
    f[:len(s)]=s.frac_coords[order]%1
    if np.any((z[:len(s)]<1)|(z[:len(s)]>118)):raise ValueError('Unknown atomic number')
    return z,f,s.lattice.matrix@s.lattice.matrix.T,basis,factor


def reduced(z):
    elems,counts=np.unique(z[z>0],return_counts=True)
    return tuple(zip(elems.tolist(),(counts//np.gcd.reduce(counts)).tolist()))


def write_json(path,record):
    tmp=path.with_suffix('.writing')
    tmp.write_text(json.dumps(record,indent=1)+'\n',encoding='utf8');tmp.replace(path)


def prepare(archive,split,out,seen):
    start=perf_counter();rows=COUNTS[split]
    target=out/(split+'.npz');meta=out/(split+'-ingestion.json')
    member=archive.getinfo('alex_mp_20/'+split+'.csv')
    if target.exists() and meta.exists():
        prev=json.loads(meta.read_text())
        if prev['source_sha256']==SOURCE_SHA and prev['npz_sha256']==sha(target):
            with np.load(target,allow_pickle=False) as data:ids=data['ids'].tolist()
            if len(ids)!=rows or set(ids)&seen:raise ValueError('Cached release ID overlap')
            seen.update(ids);return prev
    zsave=np.empty((rows,SITE_CAP),np.int16);fsave=np.empty((rows,SITE_CAP,3),np.float32)
    grams=np.empty((rows,3,3),float);ids=[]
    source_families=Counter();direction=Counter();sites=Counter();differences=Counter();examples=[];reductions=[]
    with io.TextIOWrapper(archive.open(member),encoding='utf8',newline='') as stream:
        reader=csv.DictReader(stream)
        if not {'material_id','num_sites','cif','reduced_formula'}.issubset(reader.fieldnames):raise ValueError('Missing columns')
        for i,r in enumerate(reader):
            mid=r['material_id']
            if i>=rows or not mid or mid in seen:raise ValueError(f'Duplicate or unexpected record: {mid}')
            try:zsave[i],fsave[i],grams[i],basis,factor=parse_structure(r['cif'])
            except (ValueError,KeyError,IndexError,ArithmeticError) as exc:raise ValueError(f'{split}: {mid}') from exc
            count=int(np.count_nonzero(zsave[i]))
            reported=int(r['num_sites'])
            if count*factor!=reported:
                differences[f'{reported} metadata / {count*factor} CIF']+=1
                if len(examples)<5:examples.append({'id':mid,'metadata_sites':reported,'cif_sites':count*factor})
            if factor>1:reductions.append(dict(id=mid,deposited_sites=count*factor,primitive_sites=count,supercell_index=factor))
            formula_sum=re.search(r'(?m)^_chemical_formula_sum\s+([^\r\n]+)',r['cif'])
            if formula_sum is None:raise ValueError(f'{mid}: missing explicit CIF total composition')
            parsed=Counter({Z[s]:float(v or 1) for s,v in re.findall(r'([A-Z][a-z]?)(\d+(?:\.\d+)?)?',formula_sum[1])})
            if parsed!=Counter(np.repeat(zsave[i,:count],factor).tolist()):raise ValueError(f'{mid}: CIF formula sum differs from expanded sites')
            ids.append(mid);seen.add(mid);source_families[alias_group(mid)]+=1;direction[basis]+=1;sites[count]+=1
    if len(ids)!=rows:raise ValueError(f'{split} expected {rows}, found {len(ids)}')
    k=pack_gram(grams).astype(np.float32)
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(fsave)):raise ValueError('Nonfinite geometry')
    # Canonicalize float32-rounding of positions immediately below one.
    np.remainder(fsave,1,out=fsave)
    tmp=target.with_suffix('.writing')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream,ids=np.array(ids),z=zsave,f=fsave,k=k)
    tmp.replace(target)
    info={'name':split,'source_sha256':SOURCE_SHA,'npz_sha256':sha(target),'member':member.filename,
          'member_crc32':f'{member.CRC:08x}','rows':rows,'source_identifier_families':dict(source_families),
          'site_counts':dict(sorted(sites.items())),'parser_counts':dict(direction),'elapsed_seconds':perf_counter()-start,
          'num_sites_metadata_disagreements':dict(differences),'disagreement_examples':examples,'primitive_reductions':reductions}
    write_json(meta,info)
    print(split,rows,direction,'seconds',round(info['elapsed_seconds'],1),flush=True)
    return info


def validate_ref(archive):
    with io.TextIOWrapper(archive.open('ref.csv'),encoding='utf8',newline='') as stream:
        reader=csv.DictReader(stream)
        if 'cif' in reader.fieldnames or 'structure_id' not in reader.fieldnames:raise ValueError('Unexpected reference schema')
        ids=[r['structure_id'] for r in reader]
    return {'rows':len(ids),'unique_ids':len(set(ids)),'columns':reader.fieldnames,
            'contains_structure_geometry':False,'member_crc32':f"{archive.getinfo('ref.csv').CRC:08x}"}


def cohort(data,source,output,n):
    with np.load(data/'val.npz',allow_pickle=False) as a:val={k:a[k] for k in ('ids','z','f','k')}
    with np.load(source/'train.npz',allow_pickle=False) as a:
        train_ids=set(a['ids'].tolist());train_rows=np.unique(a['z'],axis=0)
    known={reduced(z) for z in train_rows};keys={}
    raw_over=np.zeros(len(val['ids']),bool);alias_over=raw_over.copy();known_formula=raw_over.copy();alex=raw_over.copy()
    for i,(mid,z) in enumerate(zip(val['ids'],val['z'])):
        raw_over[i]=mid in train_ids
        aliases=re.findall(r'(?:mp|mvc)-\d+',mid,flags=re.I)
        alias_over[i]=any(v.lower() in train_ids for v in aliases)
        key=z.tobytes()
        if key not in keys:keys[key]=reduced(z) in known
        known_formula[i]=keys[key]
        alex[i]=alias_group(mid)=='alex'
    eligible=alex & ~raw_over & ~alias_over & ~known_formula
    ix=sorted(np.flatnonzero(eligible),key=lambda i:hashlib.sha256(val['ids'][i].encode()).digest())[:n]
    if len(ix)!=n:raise ValueError('Insufficient eligible validation rows')
    output.mkdir(parents=True,exist_ok=True)
    target=output/'val.npz';tmp=target.with_suffix('.writing')
    with tmp.open('wb') as f:np.savez_compressed(f,**{k:v[ix] for k,v in val.items()})
    tmp.replace(target)
    report=dict(release_partition='val',eligible_n=int(eligible.sum()),selected=n,source_train_sha256=sha(source/'train.npz'),
        validation_sha256=sha(data/'val.npz'),selected_sha256=sha(target),raw_validation_rows=len(alex),
        validation_literal_id_overlap=int(raw_over.sum()),validation_embedded_mp_or_mvc_alias_overlap=int(alias_over.sum()),
        reduced_formula_seen_in_source=int(known_formula.sum()),alex_namespace=int(alex.sum()),
        alex_namespace_with_seen_reduced_formula=int((alex&known_formula).sum()),
        selected_id_overlap=int(raw_over[ix].sum()),selected_embedded_alias_overlap=int(alias_over[ix].sum()),
        selected_reduced_formula_overlap=int(known_formula[ix].sum()),ids=val['ids'][ix].tolist(),
        rule='Only alex<...> IDs from released validation. Exclude literal or embedded MP/MVC training-ID matches and any reduced formula present in full MP-20 training. Take the lowest SHA-256 digests of remaining literal IDs.')
    write_json(output/'cohort.json',report)
    print('Eligible',report['eligible_n'],'selected',n,'full validation ID overlap',report['validation_literal_id_overlap'],flush=True)


def training_view(data,output):
    """Give the shared trainer an internal tuning split while preserving the released validation."""
    if data.resolve()==output.resolve():
        raise ValueError('Keep the derived training view separate from the released splits')
    release=json.loads((data/'manifest.json').read_text(encoding='utf8'))
    for split in ('train','val'):
        if sha(data/f'{split}.npz')!=release['release_splits'][split]['npz_sha256']:
            raise ValueError(f'{split} differs from its release conversion')
    output.mkdir(parents=True,exist_ok=True)
    with np.load(data/'train.npz',allow_pickle=False) as archive:
        original={k:archive[k] for k in ('ids','z','f','k')}
    patterns,inverse=np.unique(original['z'],axis=0,return_inverse=True)
    salt='crytaldrift-alex-internal-val-v1/'
    def tune(formula):
        label=','.join(f'{z}:{n}' for z,n in formula)
        digest=hashlib.sha256((salt+label).encode('ascii')).digest()
        return int.from_bytes(digest[:8],'big')%100<2
    train_formulas=set();tune_formulas=set();is_tune=[]
    for pattern in patterns:
        formula=reduced(pattern);choose=tune(formula);is_tune.append(choose)
        (tune_formulas if choose else train_formulas).add(formula)
    if train_formulas & tune_formulas:raise ValueError('Reduced formula overlap with internal tuning')
    use_tune=np.asarray(is_tune)[inverse.ravel()]
    if not use_tune.any() or use_tune.all():raise ValueError('Empty internal split')
    partitions={}
    for name,mask,forms in (('train',~use_tune,train_formulas),('val',use_tune,tune_formulas)):
        tmp=output/f'{name}.writing';target=output/f'{name}.npz'
        with tmp.open('wb') as stream:
            np.savez_compressed(stream,**{key:value[mask] for key,value in original.items()})
        tmp.replace(target)
        partitions[name]={'source':'released train','rows':int(mask.sum()),'reduced_formulas':len(forms),'sha256':sha(target)}
    with np.load(data/'val.npz',allow_pickle=False) as archive:
        released_ids=archive['ids'].tolist()
        if set(original['ids'].tolist()) & set(released_ids):raise ValueError('Released validation ID leakage')
        patterns,freq=np.unique(archive['z'],axis=0,return_counts=True)
    counts={'derived_train':0,'internal_tuning':0}
    for pattern,count in zip(patterns,freq):
        formula=reduced(pattern)
        counts['derived_train']+=int(count)*(formula in train_formulas)
        counts['internal_tuning']+=int(count)*(formula in tune_formulas)
    tmp=output/'test.writing';shutil.copyfile(data/'val.npz',tmp);tmp.replace(output/'test.npz')
    partitions['test']={'source':'entire released validation (study evaluation)','rows':len(released_ids),
                        'sha256':sha(output/'test.npz'),'rows_with_known_reduced_formula':counts}
    if partitions['test']['sha256']!=release['release_splits']['val']['npz_sha256']:
        raise ValueError('Released validation copy hash mismatch')
    write_json(output/'manifest.json',dict(source_archive_sha256=SOURCE_SHA,partitions=partitions,
        tuning_rule='SHA256 of ASCII salt followed by comma-separated ascending Z:reduced_count pairs; first eight bytes big endian modulo 100 less than 2.',
        salt=salt,internal_reduced_formula_overlap=0,pairwise_material_id_overlap=0,
        official_release_partitions=['train','val']))
    print('Study training view', {k:v['rows'] for k,v in partitions.items()},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,default=HERE/'alex-data'/'alex-verified.zip')
    p.add_argument('--out',type=Path,default=HERE/'alex-data');p.add_argument('--download',action='store_true')
    p.add_argument('--split',choices=('both','val','train'),default='both')
    p.add_argument('--mp-source',type=Path,default=HERE/'mp20-data')
    p.add_argument('--cohort',type=Path,default=HERE/'alex-validation-transfer');p.add_argument('--targets',type=int,default=128)
    p.add_argument('--skip-transfer-cohort',action='store_true',help='Convert the Alex release without an MP-20 source dataset')
    p.add_argument('--training-view',type=Path,help='Optional separate train/tuning/released-validation view for the common trainer')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.training_view and args.split!='both':p.error('--training-view requires --split both')
    if args.download and not args.archive.exists():download(args.archive)
    if args.archive.stat().st_size!=BYTES or sha(args.archive)!=SOURCE_SHA:raise ValueError('Not the complete checksum-pinned archive')
    with zipfile.ZipFile(args.archive) as a:
        members=[i.filename for i in a.infolist() if not i.is_dir()]
        if set(members)!={'alex_mp_20/train.csv','alex_mp_20/val.csv','ref.csv'}:raise ValueError('Unexpected release members')
        splits=('val','train') if args.split=='both' else (args.split,)
        seen=set();results={}
        for split in splits:
            results[split]=prepare(a,split,args.out,seen)
            if split=='val' and not args.skip_transfer_cohort:cohort(args.out,args.mp_source,args.cohort,args.targets)
        if args.split=='both':
            write_json(args.out/'manifest.json',dict(source=URL,git_commit=COMMIT,sha256=SOURCE_SHA,source_bytes=BYTES,
                release_splits=results,reference_metadata_only=validate_ref(a),train_val_overlap=0,
                basis='Deposited unit-cell chart with stable ascending atomic numbers; CIFs above 20 sites use a primitive cell only after a strict reconstructed-supercell geometry check.'))
    if args.training_view:training_view(args.out,args.training_view)
    print('Finished',list(results),flush=True)


if __name__=='__main__':main()

