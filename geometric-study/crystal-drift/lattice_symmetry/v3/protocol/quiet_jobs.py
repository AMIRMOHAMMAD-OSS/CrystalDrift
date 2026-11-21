"""Reversibly lower CPU priority or quiesce only matching study workers for the latency experiment."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import psutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
WORK=("evaluation/farm.py","evaluation/series.py","evaluation/paired.py","evaluation/follow_scores.py",
      "evaluation/wave.py","evaluation/shards.py","denovo/evaluate.py","denovo/physical.py",
      "uflow/score_queue.py","models/prepare.py","geometry/conditional.py","geometry/volume_strata.py",
      "denovo/sweep.py","denovo/pool.py","denovo/finish.py","protocol/backup.py","models.select_weights",
      "models.long_selection","release/build_evidence.py")
def discover():
    selected={}
    for proc in psutil.process_iter(["pid","name","cmdline"]):
        try:
            if not (proc.info["name"] or "").lower().startswith("python"):continue
            args=" ".join(proc.info["cmdline"] or []).replace("\\","/").lower()
            if not any(x in args for x in WORK):continue
            ancestry=proc.cwd().replace("\\","/").lower()
            if "geometric-study/crystal-drift" not in args and "geometric-study/crystal-drift" not in ancestry:continue
            for item in [proc]+proc.children(recursive=True):
                if item.pid!=os.getpid():selected[item.pid]=item
        except (psutil.AccessDenied,psutil.NoSuchProcess,FileNotFoundError):continue
    return selected

def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("nice","pause","resume","list"))
    a=p.parse_args()
    record=HERE/("timing-paused-"+sys.platform+".json")
    if a.mode=="resume":
        receipt=json.loads(record.read_text())
        for row in reversed(receipt["workers"]):
            try:
                proc=psutil.Process(row["pid"])
                if proc.create_time()==row["birth"]:proc.resume()
            except psutil.NoSuchProcess:pass
        receipt["resumed"]=time.time();record.write_text(json.dumps(receipt,indent=2))
        print("Resumed",len(receipt["workers"]),flush=True)
        return
    found=discover()
    if a.mode=="list":
        print("Eligible PIDs",list(found));return
    rows=[]
    for pid,p in found.items():
        try:
            if a.mode=="nice":p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name=="nt" else 8)
            elif p.status()!=psutil.STATUS_STOPPED:p.suspend()
            else:continue
            rows.append({"pid":pid,"birth":p.create_time()})
        except (psutil.AccessDenied,psutil.NoSuchProcess):pass
    if a.mode=="pause":record.write_text(json.dumps({"paused":time.time(),"workers":rows},indent=2))
    print(a.mode,len(rows),"study processes",flush=True)
if __name__=="__main__":main()
