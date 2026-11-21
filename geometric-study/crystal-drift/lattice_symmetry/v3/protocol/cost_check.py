"""Read actual training exposures and peak allocations from completed fits."""
import csv
import json
from pathlib import Path
V3=Path(__file__).resolve().parents[1]
def table():
    rows=[]
    for fit in sorted((V3/"models/fits").glob("*/complete.json")):
        v=json.loads(fit.read_text());rows.append(dict(method=fit.parent.name,family="orbit",seed=fit.parent.name.split("-")[-1],parameters=v["parameters"],
            seconds=v["seconds"],ids=v["distinct_anchors"],anchors=v["anchor_exposures"],updates=v["updates"],atoms=v["training_atoms"],
            peak_mib=v["peak_allocated_MiB"],dtype=v["float"]))
    for fit in sorted((V3/"uflow/fits").glob("*/complete.json")):
        cfg=fit.parent/"config.json"
        if not cfg.exists():continue
        v=json.loads(fit.read_text());c=json.loads(cfg.read_text())
        rows.append(dict(method="uflow-"+fit.parent.name,family="uFlowCSP",seed=c.get("seed"),parameters=c["parameters"]["participating"],
                     seconds=v.get("elapsed",v.get("elapsed_seconds",v.get("seconds"))),ids=c.get("train_ids"),anchors=v.get("rows",v.get("structures_seen",v.get("seen"))),
                     updates=v.get("updates",c.get("updates")),atoms=v.get("training_atoms",v.get("atoms_seen")),peak_mib=v.get("peak_bytes",0)/2**20,
                     dtype=c.get("forward_dtype")))
    return rows
if __name__=="__main__":
    rows=table()
    with (V3/"protocol/compute.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(len(rows),"training receipts")

