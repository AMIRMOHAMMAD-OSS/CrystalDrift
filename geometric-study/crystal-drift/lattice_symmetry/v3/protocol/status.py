"""Summarize only completed training/evaluation receipts, independently of running job logs."""
import json
from datetime import datetime,timezone,timedelta
from pathlib import Path
V3=Path(__file__).resolve().parents[1]
def report():
    runs=[]
    for f in sorted((V3/"models/fits").glob("*/complete.json")):
        v=json.loads(f.read_text());base=f.parent
        s=json.loads((base/"selection.json").read_text()) if (base/"selection.json").exists() else {}
        c=json.loads((base/"confirmation.json").read_text()) if (base/"confirmation.json").exists() else {}
        runs.append(dict(name=base.name,params=v.get("parameters"),updates=v.get("updates"),seconds=v.get("seconds"),
                        selected=s.get("chosen"),confirm5=c.get("hits"),confirm_native5=c.get("native")))
    for f in sorted((V3/"uflow/fits").glob("*/complete.json")):
        v=json.loads(f.read_text());base=f.parent;s=json.loads((base/"selection.json").read_text()) if (base/"selection.json").exists() else {}
        g=json.loads((base/"config.json").read_text());runs.append(dict(name="uflow-"+base.name,params=g["parameters"]["participating"],
            updates=v.get("updates"),seconds=v.get("seconds"),selected=s.get("checkpoint")))
    raw=V3/"evaluation/raw";scores=V3/"evaluation/scores";status={}
    for folder in sorted(raw.glob("*/*")):
        if not folder.is_dir():continue
        pieces=list(folder.glob("?????-?????.npz"));ready=list((scores/folder.relative_to(raw)).glob("?????-?????.npz"))
        def count(files):return sum(int(p.stem[6:])-int(p.stem[:5]) for p in files)
        status[folder.relative_to(raw).as_posix()]=dict(generated=count(pieces),scored=count(ready))
    return dict(timestamp=datetime.now(timezone(timedelta(hours=3,minutes=30))).isoformat(),completed_fits=runs,targets=status)
if __name__=="__main__":
    data=report()
    (V3/"protocol/progress.json").write_text(json.dumps(data,indent=2))
    print(data["timestamp"],len(data["completed_fits"]),"completed fit receipts")

