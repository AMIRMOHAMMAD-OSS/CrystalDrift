"""Write conditional CrystalFlow evidence from frozen case summaries."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]/"paper"
def main():
    text=[r"\paragraph{Released iterative and joint generators.}",
        r"For an iterative comparator we used the released MP-20 CrystalFlow composition-conditioned checkpoint \cite{crystalflow}. Its 20,943,872-parameter decoder receives the supplied elements, fractional coordinates and six coordinates of the current lattice stretch. Dey et al.\ examined a five-step budget on CSPBench \cite{uflow}. Our local five-step replay uses unguided Euler updates and the sampler's default unit site-velocity multiplier. The authors' released evaluation command uses 100 updates and multiplies the site velocity by $1+5t$. The only target input is the exact cell composition. The released CrystalFlow test file contains exactly the same 9,046 material IDs as our MP-20 split."]
    parity=HERE/"conditional-source-sampler-parity.json"
    if parity.exists():
        r=json.loads(parity.read_text())
        assert len(r)==2 and all(x["max_fractional_torus_error"]==0 and x["species_identical"] for x in r)
        text.append(r"On two test cells at each setting, equal initial noise gave identical fractional positions and species with the authors' complete sampler. The largest absolute lattice-entry difference was below $10^{-6}$~\AA.")
    gpucheck=HERE/"gpu-source-numerics.json"
    if gpucheck.exists():
        records=json.loads(gpucheck.read_text())
        strict=[r for r in records if r["deterministic"]]
        assert len(strict)==4 and all(r["source_vs_portable"]["fractional_max"]==0 for r in strict)
        text.append(r"With deterministic CUDA reductions, the two five-step implementations also returned identical species and fractional sites in both precision modes. Ordinary CUDA reductions can vary across repeated runs. The archives retain the draws used for scoring.")
    path=HERE/"csp-s5/summary.json"
    if path.exists():
        data=json.loads(path.read_text()); budgets=[k for k in (1,5,20) if f"9046@{k}" in data]
        if budgets:
            text.extend(["",r"\begin{center}\small\begin{tabular}{rrrrrrr}\hline",
                r"$k$ & $S$ & $kS$ & MR (\%) & Best RMS & Max-site (\%) & MR + volume (\%)\\\hline"])
            for k in budgets:
                row=data[f"9046@{k}"]
                text.append(f"{k} & 5 & {k*5} & {100*row['mr']:.2f} & {row['rmse']:.4f} & {100*row['maximum_site']:.2f} & {100*row['volume_qualified']:.2f}"+r"\\")
            text.extend([r"\hline\end{tabular}\end{center}",
                r"Each row uses the 9,046 public MP-20 test targets and the same primitive-cell, scale-normalized matcher as the compact models. The native-volume column applies the factor-of-1.10 rule to the same candidate as the RMS match. This single released checkpoint used one ordered noise schedule and FP32 tensors with TF32 matrix multiplication on an RTX 3060 Laptop GPU."])
            k=budgets[-1]
            diff=data[f"9046@{k}"]["minus_frozen_drift"]
            subject="one draw" if k==1 else f"{k} draws"
            text.append(("At %s the paired recovery difference from the original one-pass multielement CrytalDrift was $%+.2f$ percentage points (95\\%% target-bootstrap interval $[%.2f,%.2f]$).") % (subject,100*diff[0],100*diff[1],100*diff[2]))
    timing=HERE/"timing-s5/runtime.json"
    if timing.exists():
        t=json.loads(timing.read_text())
        text.append(("On the separate matched GPU workload, five-step CrystalFlow took %.2f ms per serial candidate (median) and %.2f minutes per 10,000 candidates at batch size 24. All timed GPU arms used FP32 with TF32 disabled.")%(t["median_b1_ms"],t["batch_total_seconds"]/60))
    sensitivity=HERE/"steps-5-vs-100.json"
    if sensitivity.exists():
        s=json.loads(sensitivity.read_text())
        text.append(("We repeated the first draw on the 128 hash-selected multielement targets from the same initial noise with the authors' 100-step, annealed sampler. The five-step, unannealed run recovered %d targets and the published recipe recovered %d. Mean normalized RMS among the %d targets recovered in both runs changed from %.4f to %.4f.")%(s["hits"][0],s["hits"][1],s["both"],*s["common_hit_mean_rms"]))
        factorial=HERE/"step-annealing-factorial.json"
        text.append(r"The comparison changes the integration resolution and the site-velocity schedule together.")
        if factorial.exists():
            fc=json.loads(factorial.read_text())["cells"]
            text.append(("Holding the site-velocity multiplier at one, raising the update count from five to 100 changed recovery from %d to %d targets. With annealing enabled at both resolutions, the corresponding counts were %d and %d. All four cells use the same 128 targets and starting noise.")%(fc["5/0"]["hits"],fc["100/0"]["hits"],fc["5/5"]["hits"],fc["100/5"]["hits"]))
        else:
            text.append(r"Their individual effects require separate controls.")
    text.extend(["",r"For joint generation, we score the authors' separate unconditional CrystalFlow model from its deposited sequence of raw structures \cite{crystalflow}. We checked the species-aligned fractional positions and native lattice against an independent pymatgen POSCAR reader on 100 hash-selected nonempty entries and found identical values. [\linkto{https://doi.org/10.6084/m9.figshare.29978608.v3}{Source structures}]"])
    joint=HERE.parent/"denovo/pools/crystalflow-joint-local.json"
    if joint.exists():
        r=json.loads(joint.read_text())
        text.append(("We also ran the released unconditional checkpoint locally on 10,000 fresh draws. It samples the site count from its published MP-20 prior and jointly evolves lattice, sites and a 28-channel periodic-table encoding for 100 evaluations. The complete batch-24 workload took %.2f minutes on the same GPU, including atom-type decoding and host coordinates. These timing and quality records refer to the newly generated draws.")%(r["elapsed_s"]/60))
    (PAPER/"exp-external.tex").write_text("\n".join(text)+"\n",encoding="utf8")
if __name__=="__main__":main()
