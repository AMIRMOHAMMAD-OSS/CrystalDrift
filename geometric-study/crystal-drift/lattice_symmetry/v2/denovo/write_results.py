"""Typeset the local fixed-chemistry comparison from its independently checked reductions."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
LABEL={"angular":"Physical angular","radial":"Physical radial","fixed-basis":"Fixed-basis release"}


def write(d):
    rows={r["arm"]:r for r in d["raw"]};phy={r["arm"]:r for r in d["physical"]}
    a,r=rows["angular"],rows["radial"]
    paired=d["angular_minus_radial"]
    t=[r"\section{De novo generation with matched sampled chemistry}\label{sec:denovo}",
       r"We tested the geometry learned from formula-conditioned prediction in a complete sampling pipeline. A train-only prior draws a cell multiplicity pattern and then its distinct atomic numbers in ascending order. Within each pattern the next-element counts shrink toward a shared element-rank distribution; the shrinkage strength of four was selected on validation in the earlier release. The prior receives no target formula. Its random stream produced the same 10,000 compositions in all three arms, including 6,478 draws at reduced formulas absent from MP-20 training. The physical angular and radial networks share the physical seed prior, fit seed and inference noise.",
       r"\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{3.8pt}",
       r"\caption{De novo geometry on 10,000 unfiltered draws per arm. Structural (SV), compositional (CV) and joint validity (JV) use all attempts. Coverage recall uses the 9,046 test structures; precision uses all 10,000 generated structures. Density, formation energy and distinct-element Wasserstein distances use 1,000 valid draws selected independently within each arm.}\label{tab:new-raw}",
       r"\begin{tabular}{lrrrrrrrr}\hline Method & SV (\%) & CV (\%) & JV (\%) & COV-R (\%) & COV-P (\%) & $W_\rho$ & $W_E$ & $W_{\rm el}$\\\hline"]
    for n in ("fixed-basis","radial","angular"):
        s=rows[n]
        t.append(f"{LABEL[n]} & {s['struct_valid']/100:.2f} & {s['comp_valid']/100:.2f} & {s['joint_valid']/100:.2f} & {100*s['coverage_recall']:.2f} & {100*s['coverage_precision']:.2f} & {s['density_w1']:.3f} & {s['energy_w1']:.3f} & {s['elements_w1']:.3f}"+r"\\")
    t.extend([r"\hline\end{tabular}\end{table}",
       r"Following the released DiffCSP evaluator, we require the minimum distinct-site separation to be at least 0.5~\AA\ and the cell volume to be at least 0.1~\AA$^3$ \cite{diffcsp}. We test charge/electronegativity compatibility with SMACT, with the released elemental and all-metal exceptions \cite{smact}. Joint validity also requires successful fingerprints. The structure fingerprint is the site-averaged 61-component CrystalNN vector. The 132-component Magpie composition vector is standardized with the released constants \cite{matminer}. Coverage requires a neighbor within both thresholds, 0.4 in structure space and 10 in chemistry space; the minimizing neighbors may differ. Density is in g\,cm$^{-3}$. The released MP-20 property network supplies the raw formation-energy values in eV/atom. Each distance uses the same predictor or observable on all 9,046 references.",
       r"\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{4pt}",
       r"\caption{Raw diversity and measured end-to-end CPU cost. U is the first representative in its generated bank, N has no training-set structural match, and V denotes joint validity. Each count is out of 10,000 attempts. Time includes composition and geometry draws, graph construction, and decoding to host coordinates. All three generators use one network evaluation per attempt.}\label{tab:new-cost}",
       r"\begin{tabular}{lrrrrrr}\hline Method & U & N & V$\cap$U$\cap$N & B1 (ms) & 10,000 (s) & V$\cap$U/s\\\hline"])
    for n in ("fixed-basis","radial","angular"):
        s=rows[n]
        t.append(f"{LABEL[n]} & {s['unique']:,} & {s['structural_novel']:,} & {s['valid_unique_novel']:,} & {s['median_cpu_ms']:.2f} & {s['cpu_seconds']:.2f} & {s['valid_unique_per_second']:.1f}"+r"\\")
    t.append(r"\hline\end{tabular}\end{table}")
    delta=paired["struct_valid"]; cov=paired["coverage_recall"]
    t.append(f"The angular-minus-radial difference in structural validity was {delta['pp']:+.2f} percentage points (paired 95\\% interval $[{delta['interval'][0]:.2f},{delta['interval'][1]:.2f}]$). Coverage recall changed by {cov['pp']:+.2f} points ($[{cov['interval'][0]:.2f},{cov['interval'][1]:.2f}]$). The first interval resamples generated draw pairs; the second resamples test-reference coverage indicators conditional on the two realized banks. Each result uses a single fitted seed. The raw novelty check applies a species-aware primitive-cell match against all 27,136 training structures with the same reduced formula as the draw.")
    assert all(rows[arm]["energy_w1"] is not None for arm in rows)
    t.append(r"\paragraph{Relaxation and a restricted competing-phase hull.} The first sixteen draws of each new arm entered the same CHGNet 0.3.0 zero-pressure cell relaxation using ASE FIRE for at most 100 steps and a 0.10~eV/\AA\ generalized force tolerance \cite{chgnet}. For each chemical system, the MP-20 near-hull library includes the lowest deposited formation-energy example per available reduced formula at or below 0.10~eV/atom on the deposited hull, plus elemental references. We use the same relaxed CHGNet energy convention for candidates and competing phases. Every phase and elemental terminal must be available for a complete label. The candidate-augmented hull takes the nonnegative part of the energy difference to the best competing mixture. A screened hit must converge, avoid contacts below 0.60 times the sum of covalent radii, lie within 0.10~eV/atom of that restricted hull, be unique within its relaxed cohort and have no MP-20 training match.")
    for n in ("radial","angular"):
        s=phy[n]
        t.append(f"The {n} model had {s['converged']}/16 converged relaxations, {s['short']} final short-contact cases, {s['complete']}/16 complete hulls and {s['hits']}/16 screened hits. Candidate relaxations took {s['candidate_worker_s']:.1f} CPU worker-seconds, with {s['ranking_worker_s']:.2f} seconds for final identity and hull checks.")
    t.append(r"The 195 distinct reference phases shared by the two arms all converged. Their charged cost was 238.5 CPU worker-seconds per arm. Charging generation, candidate and phase relaxations, and final scoring gives zero screened hits per summed worker-hour in each prospectively selected sixteen-draw cohort. A missing competing phase could alter a restricted-hull energy. The present physical-periodic checkpoints have no DFT-confirmed discovery yield. The eight earlier single-point calculations belong to the fixed-basis study.")
    t.extend([r"\begin{figure}[!htbp]\centering\includegraphics[width=.78\linewidth]{new-crystals.pdf}",
              r"\caption{Generated and unconstrained CHGNet-relaxed cells on the same physical scale. We traverse the angular draws and then the radial draws in their original orders, selecting the first cell with at least three species to converge with acceptable contacts and the first to fail either gate, wherever available. Intracell sites are drawn in the deposited basis. The accompanying CIFs contain the complete coordinates.}\label{fig:denovo-examples}\end{figure}"])
    (HERE.parent/"paper/denovo.tex").write_text("\n".join(t)+"\n",encoding="utf8")
