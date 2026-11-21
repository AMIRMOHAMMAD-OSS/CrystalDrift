"""Verify the released CrystalFlow test CSV's ID set against the local ordered pool."""
import csv,hashlib,json,sys
from pathlib import Path
import requests
import numpy as np
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/"scaled"))
from prepare_mp20 import parse_cif,pack_gram
save=Path(sys.argv[1])
url="https://raw.githubusercontent.com/ixsluo/CrystalFlow/9c25ff0245d787efd87c9d1d797a8968443608cc/data/mp_20/test.csv"
if not save.exists():
    with requests.get(url,stream=True,timeout=(30,90)) as r:
        r.raise_for_status()
        temp=save.with_suffix(".partial")
        with temp.open("wb") as f:
            for b in r.iter_content(262144):f.write(b)
        temp.replace(save)
sha=hashlib.sha1(b"blob 26846148\0")
with save.open("rb") as f:
    for b in iter(lambda:f.read(262144),b""):sha.update(b)
assert sha.hexdigest()=="133f5697e1d3fd94f9bf0ab6a3b6c5054d325955"
with np.load(ROOT/"revision/full/mp20/cohort.npz") as d:
    ordered,z,f,k=[d[x] for x in ("ids","z","f","k")]
index={value:i for i,value in enumerate(ordered)}
ids=[];df=dk=0.
with save.open(encoding="utf8",newline="") as stream:
    for row in csv.DictReader(stream):
        identifier=row["material_id"];ids.append(identifier);i=index[identifier]
        zz,xx,gg=parse_cif(row["cif"])
        np.testing.assert_array_equal(zz,z[i])
        delta=xx-f[i]
        df=max(df,float(np.abs(delta-np.rint(delta)).max()))
        dk=max(dk,float(np.abs(pack_gram(gg)-k[i]).max()))
assert len(set(ids))==9046 and set(ids)==set(ordered) and len(ids)==9046
assert df<1e-6 and dk<1e-5
result=dict(commit="9c25ff0245d787efd87c9d1d797a8968443608cc",git_blob_sha1=sha.hexdigest(),rows=len(ids),
            missing=0,extra=0,distinct_cell_species_count_mismatches=0,maximum_target_fractional_discrepancy=df,
            maximum_target_log_lattice_discrepancy=dk)
(ROOT/"expansion/external/official-split.json").write_text(json.dumps(result,indent=2)+"\n")
