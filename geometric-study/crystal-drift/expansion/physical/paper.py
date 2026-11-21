"""Render the physical screening tables from all 80 predeclared arm/draw records."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LABELS = {"empirical-drift": "Empirical + CrytalDrift", "learned-drift": "Learned + CrytalDrift",
          "learned-flow16": "Learned + flow (16)", "learned-uflow": "Learned + uFlowCSP",
          "crystalflow-joint": "CrystalFlow (joint)"}


def main():
    rows = list(csv.DictReader((HERE / "by-arm.csv").open()))
    summaries = {row["arm"]: row for row in rows if row["group"] == "all"}
    assert len(summaries) == 5
    material = [r"""\subsection{Relaxation and competing phases}
We screened the first sixteen draws in each ordered pool, including chemically implausible and unconverged structures. We used CHGNet weights 0.3.0, simultaneous cell and site relaxation, ASE FIRE, a maximum of 100 steps and a generalized force threshold of 0.10\,eV/\AA\ \cite{chgnet}. We recorded final atomic forces and contact lengths separately. A short contact has distance below 60\% of the corresponding covalent-radius sum.

We built a finite reference hull for each chemical system. We selected one MP-20 entry per reduced formula among phases within 0.10\,eV/atom of the archived MP hull, choosing the lowest deposited formation energy and breaking ties by material ID. We included the lowest-formation-energy elemental terminal even if it fell outside this window. We used all three MP-20 partitions to identify competing phases for physical evaluation. For each candidate we relaxed every selected subphase of its chemical system with the same CHGNet procedure. All 671 distinct references converged. In atom fractions, the signed gap is the candidate's CHGNet energy per atom minus the minimum energy of a convex combination of these phases at its composition. The energy above the hull augmented by the candidate is the positive part of this gap.

A screened hit requires candidate convergence, passing the contact test, energy within 0.10\,eV/atom of this restricted surrogate hull, uniqueness within the arm's sixteen relaxed cells and novelty against the 27,136 deposited, unrelaxed MP-20 training structures, using the maximum-site matcher. All eighty candidate hulls contained their required phases and elemental terminals. Omitting competing phases can underestimate energy above hull. These labels refer to this CHGNet library and this optimizer budget.
\begin{table}[!htbp]\centering\small
\caption{Multielement relaxation on sixteen consecutive draws per arm. Short counts the final short-contact cases, gap pass counts the energetic criterion alone, and hit requires every screening criterion. Last columns give hits over all draws at training-seen and training-unseen reduced formulas.}\label{t:screen}
\begin{tabular}{lrrrrrrr}\hline Prior and geometry & Conv. & Short & Gap pass & Novel & Hit & Seen & Unseen\\\hline"""]
    costs = []
    pools = HERE.parent / "denovo/pools"
    for arm, label in LABELS.items():
        row = summaries[arm]
        assert row["n"] == row["unique"] == row["complete_hulls"] == "16"
        subsets = {r["group"]:r for r in rows if r["arm"] == arm}
        cells = [row[k] for k in ("converged","short","thermally_screened","novel","sun_lower")]
        for group in ("seen","unseen"):
            val = subsets.get(group)
            cells.append(f"{val['sun_lower']}/{val['n']}" if val else r"\textemdash")
        material.append(" & ".join([label]+cells)+r"\\")
        candidate = float(row["candidate_core_seconds"])
        reference = float(row["reference_core_seconds"])
        post = float(row["postprocess_seconds"])
        timing = pools / (arm+"-gpu-matched.json")
        gpu = json.loads(timing.read_text())["elapsed_s"] if timing.exists() else None
        rate = 3600*int(row["sun_lower"])/(gpu+candidate+reference+post) if gpu else None
        costs.append(" & ".join([label, f"{gpu:.1f}" if gpu else r"\textemdash", f"{candidate:.1f}",
                                  f"{reference:.1f}",f"{post:.2f}",f"{rate:.2f}" if rate else r"\textemdash"])+r"\\")
    material.extend([r"""\hline\end{tabular}\end{table}
At the shared learned-prior compositions, five of twelve training-unseen drifting proposals converged, compared with six of twelve for the compact flow. None of the twelve drifting proposals met the full screen. The empirical prior supplied only seen formulas. Every relaxed draw was unique within its sixteen-structure arm.
\begin{table}[!htbp]\centering\small
\caption{Compute charged to each stand-alone pipeline. The generation column charges the entire timed 10,000-draw GPU workload. Other columns give single-thread CPU worker seconds for the sixteen candidates, their distinct reference phases and the final matcher and hull calculations. Hits per hour divides observed hits by the sum of these device-seconds.}\label{t:screen-time}
\begin{tabular}{lrrrrr}\hline Prior and geometry & Generate & Candidates & Phases & Score & Hits/h\\\hline""",
                     *costs,r"""\hline\end{tabular}\end{table}
References shared across arms are charged separately to each arm, and once per phase within an arm. These are yields of the specified generate-10,000, screen-16 workload with warm models. Totals exclude fitting, loading archives, the independent property benchmark and first-principles checks. They count GPU and CPU worker time at equal weight. The sample is small, and generation time for the deposited joint CrystalFlow pool is unknown.
\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{exp-structures.pdf}
\caption{Generated and relaxed multielement cells drawn from the released CIFs, at a common length scale. We show the first three-or-more-element cell satisfying convergence and contact criteria for each displayed generator and the first eligible failure in archived arm and draw order. The fourth structure reached the optimizer step limit. Element colors, draw IDs, minimum separations and convergence outcomes are retained for each display.}\label{fig:physical-multielement}\end{figure}"""])
    (ROOT/"paper/exp-screen.tex").write_text("\n".join(material)+"\n",encoding="utf8")


if __name__ == "__main__":
    main()
