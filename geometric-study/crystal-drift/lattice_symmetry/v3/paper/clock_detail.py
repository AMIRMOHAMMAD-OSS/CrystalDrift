"""Typeset the warm timing boundary, tails, call counts and same-device throughput."""
import csv
import json
from pathlib import Path
import numpy as np
from write import HERE,V3,table
from compose import figure
def main():
    folder=V3/"evaluation/timing"
    info=json.loads((folder/"settings.json").read_text())
    lat={v["method"]:v for v in csv.DictReader((folder/"latency.csv").open())}
    with np.load(folder/"cpu-first.npz") as c:
        names=c["methods"].astype(str).tolist()
        cpu={n:np.percentile(c["first_ms"][:,i],[50,95]) for i,n in enumerate(names)}
        calls=json.loads(str(c["calls"]))
    body=[]
    names={"prior":"Zero-call prior","geometry-donor":"Training geometry","angular-small-long":"Angular 0.2M","angular-medium-long":"Angular 2M",
           "contacts-medium-long":"Contact 2M","meanflow-medium-long":"Orbit MeanFlow 2M",
           "uflow-small-11-long":"Native uFlow 0.2M","uflow-medium-11-long":"Native uFlow 2M","released-uflow":"Released uFlowCSP"}
    for m in info["method_order"]:
        r=lat[m]
        with np.load(folder/(m+"-10k.npz")) as d:
            assert int(d["calls"])==(0 if m in ("prior","geometry-donor") else 313)
            seconds=float(d["total_seconds"])
            peak=float(d["peak_additional_bytes"])/2**20
        assert int(info["calls"][m])==(0 if m in ("prior","geometry-donor") else 1280)
        assert int(calls[m])==(0 if m in ("prior","geometry-donor") else 64)
        body.append([names[m],*[f"{x:.2f}" for x in cpu[m]],
           f"{float(r['first_median']):.2f}",f"{float(r['first_p95']):.2f}",
           f"{10000/seconds:.1f}",f"{peak:.1f}"])
    rr=[r"\section{What the two clocks include}",
       table("lrrrrrr",["Method","CPU p50","CPU p95","GPU p50","GPU p95","draws/s","MiB"],body,
       "Warm first-candidate latency in milliseconds for 64 common compositions with one host thread. Throughput uses the fixed 10,000 formulas, with batches of 32, and reports additional peak GPU tensor allocation beyond all resident loaded weights. Time includes deterministic template search and decoded CPU arrays. It excludes file writing, matching, scoring and relaxation."),
       f"The common train-only template index took {info['index_seconds']:.2f} seconds to instantiate for the GPU clock. For the serial comparison we reset the shortlist cache at the beginning of each material, start the clock, and record synchronized elapsed time after each completed proposal. The random interleaving order is retained. The 20-draw cap applies equally to all methods and time budgets. No partial proposal is scored. In the repeated 10,000-composition workload, the sampler resets template lookup for each formula, and the final partial batch contains 16 structures. A one-call method invokes the network 313 times when tensor batches cover all 10,000 structures."]
    with np.load(folder/"selector-cost.npz") as d:
        costs=[]
        for i,n in enumerate(d["names"].astype(str)):
            p50,p95=np.percentile(d["ms"][:,i],[50,95]);costs.append([n,f"{p50:.2f}",f"{p95:.2f}"])
        rr.append(table("lrr",["Ranking rule","p50 (ms)","p95 (ms)"],costs,
            f"Independent one-thread CPU timing of building and ranking one shortlist on the same 64 formulas. Each rule examines the same candidates. The ranked lists had a median of {np.median(d['shortlist_lengths']):.0f} entries. The learned ranker made {int(d['calls'])} recorded forward calls in the measured episodes."))
    (HERE/"clock-detail.tex").write_text("\n\n".join(rr)+"\n")
if __name__=="__main__":main()
