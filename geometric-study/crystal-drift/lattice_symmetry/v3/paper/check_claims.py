"""Check quantities in the actual rendered output against source-independent saved reductions."""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

HERE=Path(__file__).resolve().parent
V3=HERE.parent
RELEASE=V3/"release"

def read(p):
    return json.loads(p.read_text(encoding="utf8"))

def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as fp:
        for chunk in iter(lambda:fp.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def main(name):
    receipt=read(RELEASE/(name+"-build.json"))
    assert not receipt["unresolved"] and len(receipt["passes"])==2
    pdf=RELEASE/receipt["pdf"];archive=RELEASE/receipt["source"]
    assert digest(pdf)==receipt["pdf_sha"] and digest(archive)==receipt["source_sha"]
    layout=subprocess.check_output(["pdftotext","-layout","-enc","UTF-8",str(pdf),"-"]).decode("utf8")
    entire=re.sub(r"\s+"," ",layout)
    assert "Mathematical and experimental supplement" in entire and "AmirM.MohammadHosseini" in entire
    assert not re.search(r"\b(?:TODO|TBD|PLACEHOLDER|FIXME)\b|\[\?\]|\(\?\?\)",entire)
    abstract=entire.split("Abstract",1)[1].split("Prediction from the unit-cell composition",1)[0]
    data=read(V3/"evaluation/reports/test.json")
    kinds=("prior","contacts-medium-long","angular-medium-long","meanflow-medium-long","released-uflow","geometry-donor")
    expected={m:f"{100*data['quality']['all/5/'+m]['mr']:.2f}%" for m in kinds}
    for tag,value in expected.items():
        if value not in abstract:raise ValueError((tag,value))
    with (V3/'geometry/audits/seed-volume-pattern.csv').open() as fp:
        v=next(r for r in csv.DictReader(fp) if r['method']=='contacts-medium-long' and r['k']=='5' and r['training_match']==r['availability']=='all')
    for count in (int(v['learned_volume']),int(v['seed_volume']),int(data['quality']['all/5/prior']['native_hits'][0])):
        assert f'{count:,}' in abstract
    clocks=list(csv.DictReader((V3/"evaluation/timing/elapsed.csv").open()))
    for tag in ("contacts-medium-long","released-uflow","geometry-donor"):
        r=next(r for r in clocks if r["method"]==tag and float(r["ms"])==100)
        assert r["hits"]+"/64" in abstract
    with (RELEASE/"MAIN-COMPARISON.csv").open() as fp:
        records=list(csv.DictReader(fp))
    assert len(records)==27
    for r in records:
        q=data["quality"]["all/"+r["k"]+"/"+r["method"]]
        assert int(r["hits"])==q["hits_per_seed"][0] and float(r["MR"])==q["mr"]
        assert int(r["kS"])==int(r["k"])*int(r["S"]) and float(r["mean_best_normalized_RMS"])==q["rms_matched"]
    for p in receipt["layout"]:
        x0,y0,x1,y1=p["bounds"]
        if not(0<x0<x1<p["width"] and 0<y0<y1<p["height"]):raise ValueError(p)
    with zipfile.ZipFile(archive) as z:
        assert sorted(z.namelist())==sorted(receipt["members"])
        assert all(Path(p).suffix in (".tex",".pdf") for p in z.namelist())
    result=dict(report=receipt["pdf"],sha256=digest(pdf),pages=len(receipt["layout"]),
                independent_target_reductions=expected,matched_table_rows=len(records),
                matched_elapsed_arms=3,final_log_unresolved=receipt["unresolved"],
                overfull=receipt["overfull"],clean_source_members=len(receipt["members"]))
    (RELEASE/(name+"-claims.json")).write_text(json.dumps(result,indent=2)+"\n")
    print("Verified",name,len(records),"main table rows, all abstract rates and elapsed counts",flush=True)

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "CrytalDrift")
