"""Plot held-out recovery from the per-target audit tables."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parent


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    full = {
        row["stratum"]: row for row in read_csv(HERE / "paired_strata.csv")
        if row["draws"] == "20" and row["angular_seed"] == "11"
    }
    symbolic = json.loads((V2 / "diagnostics_template_prediction" / "confirmation" /
                           "extension" / "summary.json").read_text())["20"]["counts"]
    donor = {
        row["method"]: int(row["recovered"])
        for row in read_csv(V2 / "diagnostics_seed" / "test384" / "counts.csv")
        if row["stratum"] == "all" and row["k"] == "20" and row["metric"] == "scaled"
    }

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    blue, orange, gray = "#225b8f", "#ab511c", "#7c8b99"
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.3),
                                      gridspec_kw={"width_ratios": [1, 1.26]})
    groups = ["1-4", "5-8", "9-12", "13-20"]
    xs = np.arange(len(groups))
    for col, color, name, marker in (
        ("mean_angular_pct", blue, "Released angular CrytalDrift", "o"),
        ("uflow_pct", orange, "Released uFlowCSP", "s")
    ):
        vals = [float(full["sites" + g][col]) for g in groups]
        left.plot(xs, vals, marker=marker, color=color, linewidth=2.3, markersize=6, label=name)
    left.set(ylim=(0, 103), xlim=(-.15, 3.15), xticks=xs,
             xticklabels=[g.replace("-", "–") for g in groups],
             ylabel="Targets recovered in 20 draws (%)", xlabel="Sites in the target unit cell")
    left.legend(frameon=False, loc="lower left", bbox_to_anchor=(-.01, -.37), fontsize=9)
    left.set_title("A   Full MP-20 test set · 9,046 targets", loc="left", fontsize=11, weight="bold", pad=13)
    left.grid(axis="y", color="#dddddd", linewidth=.7)
    left.set_axisbelow(True)

    rows = [
        ("Original angular¹", symbolic["Amean"], blue),
        ("Wyckoff template only", symbolic["prior"], gray),
        ("Template + angular", symbolic["C"], blue),
        ("Mixed geometry seeds only", donor["hybrid_prior"], gray),
        ("Geometry seeds + angular²", donor["volume_anchor"], blue),
        ("Released uFlowCSP", donor["uflow"], orange),
    ]
    for y, (_, hits, color) in enumerate(rows):
        rate = 100 * hits / 384
        right.barh(y, rate, .57, color=color)
        right.text(rate + 1.4, y, f"{rate:.1f}%", va="center", fontsize=9, color="#202a33")
    right.set(yticks=range(len(rows)), yticklabels=[r[0] for r in rows], xlim=(0, 100),
              xlabel="Targets recovered in 20 draws (%)")
    right.invert_yaxis()
    right.spines["left"].set_visible(False)
    right.tick_params(axis="y", length=0)
    right.set_title("B   Same 384 targets for every method", loc="left", fontsize=11, weight="bold", pad=13)
    fig.subplots_adjust(left=.075, right=.985, bottom=.27, top=.88, wspace=.70)
    fig.text(.545, .041, "¹ Mean of three fits.   ² Donor volume preserved.", fontsize=8, color="#425465")
    fig.savefig(HERE / "one_call_recovery.svg", bbox_inches="tight")
    fig.savefig(HERE / "one_call_recovery.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
