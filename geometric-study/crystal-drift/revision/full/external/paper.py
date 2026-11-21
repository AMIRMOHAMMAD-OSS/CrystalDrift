"""Generate the released-checkpoint paragraph and table from completed full splits."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    runs = {k: json.loads(p.read_text()) for k in (1, 5, 20)
            if (p := HERE/"mp20"/f"full-k{k}.json").is_file()}
    assert {1,5}.issubset(runs) and all(r["targets"] == 9046 for r in runs.values())
    columns = ("rms_recovery", "strict_recovery", "rms_and_volume_1.10")
    lines = []
    for k, data in runs.items():
        stats = data["uFlowCSP"]
        rates = [f"{100*stats[c]['mean']:.2f}" for c in columns]
        rms = stats["mean_best_rms_among_recovered"]["mean"]
        lines.append(" & ".join([str(k), *rates, f"{rms:.3f}"]) + r"\\")
    def points(k, method, invert=False):
        s = runs[k]["compact_minus_uFlowCSP"][method]["rms_recovery"]
        a, b, c = ((-s["mean"], -s["hi"], -s["lo"]) if invert else
                   (s["mean"], s["lo"], s["hi"]))
        return f"{100*a:.2f} points ($[{100*b:.2f},{100*c:.2f}]$)"
    content = r"""\paragraph{A published one-step predictor.}
We replayed the released Cartesian uFlowCSP checkpoint \cite{uflow} on all 9,046 MP-20 test IDs. The 131.2-million-parameter network received the cell composition, null symmetry label and its original 100-token input. Each proposal used one FP32 network evaluation. The original 128-target, five-draw predictions were retained exactly. We applied the same matcher and native-volume test to every unrelaxed proposal.

\begin{table}[!htbp]\centering\small
\caption{Released uFlowCSP on the full MP-20 test split. MR, maximum-site recovery, and MR with native volume agreement are percentages. Best RMS is the mean of the best normalized RMS displacement over RMS-recovered targets.}\label{t:revision-external-full}
\begin{tabular}{rrrrr}\hline
Draws & MR & Max-site & MR + volume & Best RMS\\\hline
""" + "\n".join(lines) + r"""
\hline\end{tabular}\end{table}

"""
    content += ("Training-set retrieval led uFlowCSP by " + points(1, "retrieval")
                + " in one-draw recovery. At five draws uFlowCSP led retrieval by "
                + points(5, "retrieval", invert=True) + " and CrytalDrift by "
                + points(5, "drift", invert=True)
                + ". The intervals pair target IDs and condition on the three compact fits and one released checkpoint.\n")
    (HERE.parents[2]/"paper/revision-external-full.tex").write_text(content,encoding="utf-8")


if __name__ == "__main__":
    main()
