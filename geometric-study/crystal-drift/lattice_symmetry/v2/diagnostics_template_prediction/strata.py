"""Audit where frozen formula-only proposals work; validation labels used solely for scoring."""
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

out=Path(__file__).resolve().parent
root=out.parents[2]
targets=defaultdict(list)
with (out/"draws.csv").open(newline="") as f:
    for r in csv.DictReader(f):
        targets[r["id"]].append(r)
template_truth={}
for path in (out.parent/"wyckoff/labels").glob("val-*.jsonl.gz"):
    with gzip.open(path,"rt") as f:
        for line in f:
            row=json.loads(line)
            if row["id"] in targets:
                template_truth[row["id"]]=row.get("group")
with np.load(root/"scaled/mp20-data/train.npz") as train:
    train_formulas={str(i):tuple(z[z>0]) for i,z in zip(train["ids"],train["z"])}
with np.load(root/"scaled/mp20-data/val.npz") as val:
    val_formulas={str(i):tuple(z[z>0]) for i,z in zip(val["ids"],val["z"]) if str(i) in targets}
all_formulas=set(train_formulas.values())
groups=defaultdict(list)
for i,rows in targets.items():
    f=val_formulas[i]
    groups["all"].append((i,rows))
    groups["formula_seen" if f in all_formulas else "formula_unseen"].append((i,rows))
    groups["9plus_sites" if len(f)>=9 else "under_9_sites"].append((i,rows))
body={}
for name,items in groups.items():
    data={}
    for j in (1,5):
        stats=dict(N=len(items),template_labelled=sum(template_truth[i] is not None for i,_ in items))
        stats["correct_group_in_proposals"]=sum(any(int(r["group"])==template_truth[i] for r in rows[:j]) for i,rows in items)
        for a in ("A","symbolic_prior","symbolic_angular"):
            stats[a]=sum(any(bool(r[a+"_rms"]) for r in rows[:j]) for _,rows in items)
            stats[a+"_reverse"]=sum(any(bool(r[a+"_rms_reverse"]) for r in rows[:j]) for _,rows in items)
        data[j]=stats
    body[name]=data
(out/"strata.json").write_text(json.dumps(body,indent=2)+"\n")
print(json.dumps(body,indent=2))
