"""Regenerate the volume-correction table from the completed rematch."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
d = json.loads((HERE/"summary.json").read_text())
lines = [r"""\subsection{Correcting the native volume}
We fit a ridge regression to log volume per atom using elemental fractions and an unpenalized intercept on the 27,136 MP-20 training structures. We add the mean training residual at an observed reduced formula, shrunk by $n/(n+3)$ for $n$ training occurrences. For an unseen formula the residual is zero. The correction adds $(\log V_{\rm pred}-\log V_{\rm draw})I/3$ to the log-stretch tensor. This preserves the volume-normalized shape and every fractional position. We reconstructed all 542,760 cells from the three original fits and ran both matchers again.
\begin{table}[!htbp]\centering\small
\caption{Volume correction on 9,046 official MP-20 targets. Each pair gives original and corrected recovery in percent. Intervals resample target IDs, paired across arms and seeds, and refer to the difference in native-volume-qualified recovery, in percentage points.}\label{t:correct-volume}
\begin{tabular}{rrrrr}\hline $k$ & Standard MR & Maximum-site MR & MR with native volume & Difference [95\% interval]\\\hline"""]
for k in (1,5,20):
    cols = []
    for key in ("mr","maximum_site","volume_match"):
        row = d[f"{key}@{k}"]
        cols.append(f"{row['raw']*100:.2f} / {row['corrected']*100:.2f}")
    ci = [100*x for x in row["ci"]]
    cols.append(f"{row['difference']*100:+.2f} [{ci[0]:.2f},{ci[1]:.2f}]")
    lines.append(" & ".join([str(k)]+cols)+r"\\")
a,b = d["mean_absolute_log_volume_error"]
lines.extend([r"\hline\end{tabular}\end{table}",
              f"Mean absolute log-volume error per draw changed from {a:.3f} to {b:.3f}. "
              f"The number of individual acceptance decisions changed by the rematch was {d['flipped_draws']:,}. "
              "Finite absolute tolerances during primitive reduction can affect even a scale-normalized comparison."])
(HERE.parents[1]/"paper/exp-volume.tex").write_text("\n".join(lines)+"\n")
