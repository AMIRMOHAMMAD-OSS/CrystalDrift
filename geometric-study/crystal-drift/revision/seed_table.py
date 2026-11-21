"""Typeset per-seed recovery from the published case-level summaries."""
import csv
import json
from pathlib import Path

REV = Path(__file__).resolve().parent
full = REV/"full"
with (REV/"binary/headline-per-seed.csv").open(newline="") as stream:
    binary = {(r["method"],int(r["seed"]),int(r["draws"])):float(r["mr"]) for r in csv.DictReader(stream)}
mp20 = {k:json.loads((full/f"mp20/full-k{k}.json").read_text()) for k in (1,5,20)}
wbm = json.loads((full/"wbm/revision-summary.json").read_text())["5"]
rows = [
    r"\begin{inplace}{table}\centering\small",
    r"\caption{Raw recovery by fitted seed and number of draws. Entries are percentages of all targets in the indicated cohort. The binary and full MP-20 columns refer to separately trained networks and separately selected target sets. WBM uses its own formula-disjoint training split. Retrieval seeds index its sampled donors.}\label{t:seed-results}",
    r"\begin{tabular}{llrrrrrrr}\hline",
    r" & & \multicolumn{3}{c}{Binary MP-20, 128} & \multicolumn{3}{c}{Full MP-20, 9,046} & WBM, 128\\",
    r"Method & Seed & $k{=}1$ & 5 & 20 & $k{=}1$ & 5 & 20 & $k{=}5$\\\hline",
]
for method, label in (("drift","CrytalDrift"),("flow1","One-Euler flow"),("flow16","16-step flow"),("retrieval","Retrieval")):
    for seed in (11,23,37):
        binary_name = "exact" if method=="drift" else method
        values = [binary[binary_name,seed,k] for k in (1,5,20)]
        values.extend(mp20[k]["per_seed"][method][str(seed)]["rms_recovery"] for k in (1,5,20))
        values.append(wbm["per_seed"][method][str(seed)]["rms_recovery"])
        cells = [label if seed==11 else "",str(seed),*[f"{100*x:.2f}" for x in values]]
        rows.append(" & ".join(cells)+r" \\")
    rows.append(r"\hline")
rows.append(r"\end{tabular}\end{inplace}")
(REV.parent/"paper/revision-seeds.tex").write_text("\n".join(rows)+"\n",encoding="utf8")
