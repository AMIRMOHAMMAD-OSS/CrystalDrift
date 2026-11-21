"""Build the full-space manuscript subsection directly from the ordered score banks."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAMES = {
    "empirical-drift": "Empirical + CrytalDrift",
    "learned-drift": "Learned + CrytalDrift",
    "learned-flow16": "Learned + flow (16)",
    "learned-uflow": "Learned + uFlowCSP",
    "crystalflow-joint": "CrystalFlow (joint)",
}


def value(row, key, multiplier=1.0, precision=2):
    if key not in row:
        return r"\textemdash"
    return f"{row[key]*multiplier:.{precision}f}"


def main():
    fresh = "crystalflow-joint-local"
    fresh_report = HERE / f"panel/{fresh}.json"
    rows = {name: json.loads((HERE / f"panel/{name}.json").read_text()) for name in NAMES}
    paper = [
        r"""\subsection{Composition and geometry across multielement MP-20}
We draw an exact unit-cell composition $c$ and then draw a periodic structure $x$ from the conditional generator,
\begin{equation}
 Q(\mathrm dc,\mathrm dx)=\pi(\mathrm dc)\,Q_{\theta,c}(\mathrm dx).
 \label{eq:full-prior}
\end{equation}
At a composition absent from training, $Q_{\theta,c}$ is still defined by the network. The archive provides no empirical conditional distribution there. The empirical prior resamples a training cell's composition. Our learned prior first samples the multiplicities $m=(m_1,\ldots,m_d)$ of the elements in increasing atomic-number order from their training histogram. For species rank $j$, let $T(z)$ count occurrences of $z$ after the preceding element for that pattern and rank. Let $B(z)$ be the distribution of $z$ at rank $j$ in cells with $d$ species, smoothed by $1/2$ per element. We draw with weights $T(z)+\alpha B(z)$ over training elements above the preceding one that leave enough greater elements for the remaining ranks. We normalize $B$ on this eligible set. On 8,856 supported validation rows, mean negative log likelihood was 15.36, 14.51 and 14.73 nats for $\alpha=0.5,4,32$. The other 191 rows had an unseen pattern or element. We selected $\alpha=4$.

Each arm retains 10,000 ordered raw proposals. The geometry comparison uses the same sequence of compositions for learned-prior CrytalDrift, 16-step flow and the released one-step uFlowCSP. The compact models also share the noise and the seed-11, one-pass training budget. This flow budget differs from the validation-selected conditional comparison. The empirical arm changes only the composition prior. As an independent complete pipeline, we score the authors' MP-20 joint CrystalFlow raw POSCAR sequence from \linkto{https://doi.org/10.6084/m9.figshare.29978608.v3}{their released Table 3 data} \cite{crystalflow}. We retain the empty entry at zero-based position 4,731 as a failed proposal.
\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{3.3pt}
\caption{Raw multielement de novo generation on MP-20. Structural, compositional and joint validity are percentages of all 10,000 proposals. Coverage recall uses all 9,046 test references and precision uses all 10,000 proposals. Wasserstein distances compare 1,000 jointly valid proposals to all references. The units are g\,cm$^{-3}$ for density $\rho$, eV per atom for predicted formation energy $E_f$, and elements for $d$.}\label{tab:full-denovo}
\begin{tabular}{lrrrrrrrr}\hline
Prior and geometry & Struct. & Comp. & Joint & Cov-R & Cov-P & $W_1(\rho)$ & $W_1(E_f)$ & $W_1(d)$\\\hline"""
    ]
    for name, r in rows.items():
        values = [value(r, k, 0.01) for k in ("struct_valid", "comp_valid", "joint_valid")]
        values += [value(r, k, 100) for k in ("coverage_recall", "coverage_precision")]
        values += [
            value(r, k, precision=3) for k in ("w1_density", "w1_formation_energy", "w1_nelems")
        ]
        paper.append(" & ".join([NAMES[name], *values]) + r"\\")
    paper.append(
        r"""\hline\end{tabular}\end{table}
The DiffCSP evaluator uses a minimum distinct-site separation of 0.5\,\AA, cell volume of at least 0.1\,\AA$^3$, and a SMACT composition test that accepts unary elements and all-metal compositions. A joint-valid draw must also yield its fingerprints. We average the 61-component CrystalNN fingerprint across sites and standardize the 132-component Magpie fingerprint with the released composition scaler. Each fingerprint space supplies its own nearest neighbour, using thresholds of 0.4 and 10 respectively \cite{diffcsp,matminer,smact}. Coverage uses every successful fingerprint, including those from candidates that fail a validity predicate. The property subset is sampled without replacement with seed 7121. The formation-energy quantity is the released MP-20 supervised predictor's output on raw cells. It measures distributional agreement with the same predictor on held-out references.
\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{3.4pt}
\caption{Diversity and generation cost. Each of the first four columns is a percentage of the original 10,000 draws. UF refers to an unseen reduced formula, U to a first matching representative, N to no match in training and V to joint validity. Last three columns report median warm batch-one latency, minutes per 10,000 structures at batch 24, and the estimated number of valid unique structures per GPU minute. Timing includes composition sampling, host transfers and decoding. CrystalFlow's published samples carry no local generation timing.}\label{tab:full-denovo-time}
\begin{tabular}{lrrrrrrrr}\hline
Prior and geometry & UF & U & N & V$\cap$U$\cap$N & NFE & ms & min & V$\cap$U/min\\\hline"""
    )
    for name, r in rows.items():
        values = [
            value(r, k, 0.01)
            for k in ("formula_unseen_training", "unique", "structural_novel", "valid_unique_novel")
        ]
        s = 16 if name == "learned-flow16" else 1
        values += [
            str(s) if name != "crystalflow-joint" else r"100$^*$",
            value(r, "batch_one_median_ms", precision=1),
            value(r, "gpu_wall_s", 1 / 60, 3),
            value(r, "valid_unique_per_min", precision=0),
        ]
        paper.append(" & ".join([NAMES[name], *values]) + r"\\")
    paper.append(
        r"""\hline\end{tabular}\end{table}
$^*$CrystalFlow specifies 100 Euler steps as its default. Its published MP-20 results are 99.55\% structural validity, 81.96\% compositional validity, 98.21\% coverage recall and 99.84\% coverage precision \cite{crystalflow}. Table~\ref{tab:full-denovo} gives our independent evaluation of the deposited raw structures.
We test structural novelty against the 27,136 training cells and count the first geometrically distinct representative in proposal order. Both checks use species-preserving primitive-cell matching, volume scaling, lattice tolerance 0.3, site tolerance 0.5 and angular tolerance $10^\circ$, with no supercell search. Formula novelty compares reduced multiplicities and elements. It counts zero for the empirical prior, even when geometry at a training formula differs from every training structure.

Under the learned prior, 6,478 draws had a reduced formula absent from training, representing 6,368 distinct unseen reduced formulas. In CrytalDrift, 3,210/3,522 seen-formula draws and 5,860/6,478 unseen-formula draws pass the structural test. Median minimum separation falls from 1.10 to 0.98\,\AA. The corresponding composition-valid counts are 3,245 and 5,112. Across the same chemical inputs, the 16-step compact flow and CrytalDrift pass the structural test on 9,069 and 9,070 draws, while uFlowCSP passes on 9,493. With its own sampled chemistry, joint CrystalFlow passes on 9,963. This last comparison changes both the law of the input compositions and the geometry model.

We timed all local pipelines on one GeForce RTX 3060 Laptop GPU with FP32 networks and one Torch CPU thread. We collected 128 warm batch-one measurements and one fixed batch-24 workload per arm. The compact throughput estimates combine the unfiltered evaluation bank's valid unique fraction with an independent timed run at the same prior and model. The GPU wall times include composition sampling and lattice, fractional and Cartesian decoding to host arrays. They exclude checkpoint initialization, matching, property evaluation and relaxation. The measured throughputs for the four local pipelines, in table order, were """
    )
    speeds = [f"{10000/rows[n]['gpu_wall_s']:.1f}" for n in list(rows)[:4]]
    paper.append(
        ", ".join(speeds[:-1])
        + " and "
        + speeds[-1]
        + " structures per second under this hardware and host load."
    )
    if all("gpu_replay_matches" in rows[name] for name in list(rows)[:4]):
        paper[-2] = (
            paper[-2]
            .replace(
                "one fixed batch-24 workload", "an initial and a matched-noise batch-24 workload"
            )
            .replace(
                "The compact throughput estimates combine the unfiltered evaluation bank's valid unique fraction with an independent timed run at the same prior and model.",
                "The reported timed replay uses the same ordered chemical and geometric random draws as the quality evaluation. Every compact output agreed with the CPU evaluation bank to within $6\\times 10^{-7}$ in the lattice chart and periodic fractional coordinates. The uFlowCSP replay exactly reproduced all stored coordinates.",
            )
        )
        first = [f"{rows[name]['gpu_first_wall_s']:.1f}" for name in list(rows)[:4]]
        second = [f"{rows[name]['gpu_wall_s']:.2f}" for name in list(rows)[:4]]
        paper.append(
            "The initial workloads took "
            + ", ".join(first[:-1])
            + " and "
            + first[-1]
            + " seconds for 10,000. The matched-noise workloads took "
            + ", ".join(second[:-1])
            + " and "
            + second[-1]
            + " seconds. Host contention varied between runs, and the absolute latency measurements reflect that variation."
        )
    if all("coverage_recall" in r for r in rows.values()):
        paper.append(
            "With the common learned prior, coverage recall was "
            + ", ".join(
                [
                    f"{100*rows[n]['coverage_recall']:.2f}\\%"
                    for n in ("learned-drift", "learned-flow16")
                ]
            )
            + " and "
            + f"{100*rows['learned-uflow']['coverage_recall']:.2f}\\% for CrytalDrift, the compact 16-step flow and uFlowCSP, respectively. "
            + "Since their compositional fingerprints coincide draw for draw, this recall difference arises from the generated geometry."
        )
        if "reference_within_structure_cutoff" in rows["learned-uflow"]:
            shared = [rows[n] for n in ("learned-drift", "learned-flow16", "learned-uflow")]
            assert len({r["reference_within_chemistry_cutoff"] for r in shared}) == 1
            counts = [f"{r['reference_within_structure_cutoff']:,}" for r in shared]
            paper.append(
                f"The composition cutoff passes on {shared[0]['reference_within_chemistry_cutoff']:,}/9,046 references for each model. "
                + "The structure cutoff passes on "
                + ", ".join(counts[:2])
                + " and "
                + counts[2]
                + " references, respectively."
            )
    if (HERE / f"pools/{fresh}.json").exists():
        source = json.loads((HERE / f"pools/{fresh}.json").read_text())
        assert source.get("nfe", source.get("steps", 100)) == 100
        t = source["elapsed_s"]
        lat = sorted(source["b1_ms"])
        median = (lat[len(lat) // 2 - 1] + lat[len(lat) // 2]) / 2
        paper.append(
            "\\paragraph{Independent joint-sampler replay.} We ran the released joint CrystalFlow weights from random inputs in a separate timing experiment. The sampler drew site counts from the authors' MP-20 prior and jointly integrated lattice, sites and the 28-component periodic-table atom encoding for 100 Euler steps, without velocity annealing. The 10,000 independent draws took "
            + f"{t:.2f} seconds ({t/60:.2f} min per 10,000, {10000/t:.2f} structures/s, median batch-one latency {median:.1f} ms). The five primary arms retain the authors' deposited draw sequence."
        )
        if fresh_report.exists():
            local = json.loads(fresh_report.read_text())
            if "coverage_recall" in local and "w1_formation_energy" in local:
                paper.append(
                    "Secondary scoring of this independent bank found "
                    + f"{local['struct_valid']/100:.2f}\\% structural, {local['comp_valid']/100:.2f}\\% compositional and {local['joint_valid']/100:.2f}\\% joint validity. "
                    + f"Coverage recall and precision were {local['coverage_recall']*100:.2f}\\% and {local['coverage_precision']*100:.2f}\\%. "
                    + f"Density, formation energy and element-count Wasserstein distances were {local['w1_density']:.3f} g\\,cm$^{{-3}}$, {local['w1_formation_energy']:.3f} eV/atom and {local['w1_nelems']:.3f}, respectively. "
                    + f"There were {local['unique']:,} unique structures, {local['structural_novel']:,} with no training-set structural match and {local['formula_unseen_training']:,} draws at a training-unseen reduced formula. "
                    + f"The {local['valid_unique']:,} valid unique proposals correspond to {local['valid_unique_per_min']:.1f} per GPU minute."
                )
    paired_file = HERE / "panel/paired-validity.json"
    if paired_file.exists():
        paired = json.loads(paired_file.read_text())
        text = []
        for name, label in (("learned-flow16", "16-step flow"), ("learned-uflow", "uFlowCSP")):
            score = paired[name]["all"]["struct_valid"]
            lo, hi = score["paired_bootstrap95_pp"]
            text.append(
                f"{score['difference_pp']:+.2f} percentage points versus {label} ($[{lo:+.2f},{hi:+.2f}]$)"
            )
        paper.append(
            "At the same composition draws, the difference in structural validity for CrytalDrift was "
            + " and ".join(text)
            + ". Intervals are 95\\% paired bootstrap intervals over 10,000 draws, conditional on the fitted models."
        )
    if any("coverage_recall" not in r or "w1_formation_energy" not in r for r in rows.values()):
        missing = [
            NAMES[n]
            for n, r in rows.items()
            if "coverage_recall" not in r or "w1_formation_energy" not in r
        ]
        paper.append(
            "Dashes in Table~\\ref{tab:full-denovo} mark measurements still being evaluated for "
            + ", ".join(missing)
            + "."
        )
    (ROOT / "paper/exp-denovo.tex").write_text("\n".join(paper) + "\n", encoding="utf-8")
    strata = [
        r"""\subsection*{Raw geometry at seen and unseen reduced formulas}
\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{Geometry of the five primary de novo banks by formula overlap with the MP-20 training split. In each group $n$ counts draws, SV is the percentage passing structural validity, $r$ is the median closest distinct-site distance in \AA, and $W$ is the unfiltered density Wasserstein distance in g\,cm$^{-3}$ against held-out references in the same group. There are 1,896 formula-seen and 7,150 formula-unseen test references. The shared learned-prior arms have the same group membership.}\label{tab:dng-strata}
\begin{tabular}{lrrrrrrrr}\hline
&\multicolumn{4}{c}{Formula seen in training}&\multicolumn{4}{c}{Formula unseen in training}\\
Prior and geometry & $n$ & SV & $r$ & $W$ & $n$ & SV & $r$ & $W$\\\hline"""
    ]
    for name, row in rows.items():
        fields = []
        for group in ("seen", "unseen"):
            record = row["by_reduced_formula"].get(group)
            if record:
                n = record["proposals"]
                fields.extend(
                    [
                        str(n),
                        f"{100*record['struct_valid']/n:.1f}",
                        f"{record['median_shortest_A']:.2f}",
                        f"{record['raw_density_w1']:.3f}",
                    ]
                )
            else:
                fields.extend(["0", r"\textemdash", r"\textemdash", r"\textemdash"])
        strata.append(" & ".join([NAMES[name], *fields]) + r"\\")
    strata.append(
        r"""\hline\end{tabular}\end{table}
The evaluator caps the reported closest-site distance at 10.5\,\AA, including for cells with one site. A single empty published CrystalFlow record has no formula or geometry and enters neither group. The seen and unseen groups average over different chemistry. Within each group the three shared-prior arms allow a comparison of the conditional generators on identical compositions."""
    )
    (ROOT / "paper/exp-denovo-strata.tex").write_text("\n".join(strata) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
