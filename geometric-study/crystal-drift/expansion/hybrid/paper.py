"""Write the frozen allocation table and estimate uncertainty paired by material."""
import csv
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]/"paper"
SEEDS=(11,23,37)
METHODS=("drift","flow1","flow16")
NAMES=("CrytalDrift","One-Euler flow","16-step flow")


def report(name):
    stats=json.loads((HERE/f"{name}.json").read_text())["results"]
    with (HERE/f"{name}-per-case.csv").open() as fd:rows=list(csv.DictReader(fd))
    ids=sorted({r["id"] for r in rows})
    pairs={(r["method"],int(r["seed"]),r["id"]):int(float(r["hybrid"])) for r in rows}
    effects={}
    for method in METHODS[1:]:
        d=np.array([np.mean([pairs["drift",s,mid]-pairs[method,s,mid] for s in SEEDS]) for mid in ids])
        rng=np.random.default_rng(129786)
        boot=np.concatenate([d[rng.integers(0,len(ids),(100,len(ids)))].mean(1) for _ in range(100)])
        effects[method]=[float(d.mean()),*np.quantile(boot,[.025,.975]).tolist()]
    return stats,effects


def ci(nums):
    m,lo,hi=100*np.array(nums)
    return "$"+f"{m:+.2f}$ points ($[{lo:.2f},{hi:.2f}]$)"


def main():
    ex,ep=report("exploratory");co,cp=report("confirmation")
    lines=[r"\subsection{What does one retrieval draw add?}",
        r"On previously inspected MP-20 targets, we paired the first training-template proposal with the first nineteen neural proposals, keeping the total at twenty. Before scoring an independent cohort, we fixed this allocation, the one-pass checkpoints, and the hash-selection rule. From Alexandria validation we selected 512 distinct reduced formulas absent from all MP-20 splits and from the earlier 128-target Alexandria cohort. Material IDs and embedded MP aliases were checked. Prototype sharing across formulas remains possible. Each target received the same composition, common retrieval draw and ordered neural noise across arms.","",
        r"\begin{inplace}{table}\centering\small",
        r"\caption{Frozen one-template, nineteen-neural allocation. The first two columns give recovery as a percentage of all targets. The last two count Alexandria targets recovered exclusively by the hybrid or exclusively by twenty retrieval draws, expressed as percentages of all 512 targets averaged across seeds.}\label{t:hybrid}",
        r"\begin{tabular}{lrrrr}\hline Method & MP-20 & Alexandria & Hybrid only & Retrieval only\\\hline"]
    for m,label in zip(METHODS,NAMES):
        lines.append("1 retrieval + 19 "+label+" & "+" & ".join(f"{100*a:.2f}" for a in (ex[m]["hybrid"][0],co[m]["hybrid"][0],co[m]["rescued_all20"][0],co[m]["lost_vs_all20"][0]))+r" \\")
    lines.append(f"20 retrieval & {100*ex['drift']['retrieved_20'][0]:.2f} & {100*co['drift']['retrieved_20'][0]:.2f} & -- & -- "+r"\\\hline\end{tabular}\end{inplace}")
    lines.append("On the independent cohort the drifting hybrid improved on twenty retrieval candidates by "+ci(co["drift"]["difference_vs_retrieval"])+". Its paired differences from the one-Euler and sixteen-step hybrids were "+ci(cp["flow1"])+" and "+ci(cp["flow16"])+", respectively. The bootstrap resamples entire target IDs across the three fit seeds.")
    (PAPER/"exp-hybrid.tex").write_text("\n".join(lines)+"\n",encoding="utf8")
    (HERE/"paired-hybrids.json").write_text(json.dumps(dict(exploratory=ep,confirmation=cp),indent=2)+"\n")

if __name__=="__main__":main()
