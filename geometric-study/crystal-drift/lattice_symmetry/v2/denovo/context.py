"""Audit chemistry strata and the sibling benchmark under the same raw MP-20 evaluator."""
import csv
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def inspect():
    result={}
    for method in ("fixed-basis","radial","angular","empirical-drift","learned-uflow","crystalflow-joint-local"):
        own=method in ("fixed-basis","radial","angular")
        source=HERE if own else ROOT/"expansion/denovo"
        panel=load(source/"panel"/(method+".json"))
        with np.load(source/"panel"/(method+"-cases.npz")) as c:
            assert len(c["ids"])==10000
            strata={}
            for seen in (True,False):
                key="seen" if seen else "unseen"
                mask=c["seen_reduced_formula"]==seen
                strata[key]=dict(draws=int(mask.sum()),structural=int(c["struct_valid"][mask].sum()),
                                 joint=int(c["valid"][mask].sum()))
                if not mask.any():
                    assert not panel["by_reduced_formula"].get(key,{}).get("proposals",0)
                    continue
                record=panel["by_reduced_formula"][key]
                for a,b in (("draws","proposals"),("structural","struct_valid"),("joint","joint_valid")):
                    assert strata[key][a]==record[b],(method,key,record)
            assert int(c["valid"].sum())==panel["joint_valid"]
        result[method]=dict(strata=strata,structural=panel["struct_valid"],joint=panel["joint_valid"],
                            unseen=panel["formula_unseen_training"],recall=panel["coverage_recall"],
                            precision=panel["coverage_precision"])
    with (HERE/"screen/per-candidate.csv").open(newline="") as fp:records=list(csv.DictReader(fp))
    for name in ("radial","angular"):
        for key in ("seen","unseen"):
            rows=[r for r in records if r["arm"]==name and (r["seen"]=="True")== (key=="seen")]
            result[name]["strata"][key]["screen_draws"]=len(rows)
            result[name]["strata"][key]["converged"]=sum(r["converged"]=="True" for r in rows)
            result[name]["strata"][key]["screen_hits"]=sum(r["screened_sun"]=="True" for r in rows)
    (HERE/"formula-strata.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    return result


def write():
    r=inspect()
    t=[r"\paragraph{Which geometries did the prior reach?} The shared chemistry sample contains 3,522 draws at seen reduced formulas and 6,478 at unseen ones. The first set assesses geometry where training contains at least one member of that formula. The second tests extrapolation to previously unseen stoichiometries."]
    for n in ("radial","angular"):
        ss=r[n]["strata"]
        a,b=ss["seen"],ss["unseen"]
        t.append(f"The {n} generator passed structural validity on {a['structural']:,}/{a['draws']:,} seen draws and {b['structural']:,}/{b['draws']:,} unseen draws. In the first sixteen proposals its relaxations converged on {a['converged']}/{a['screen_draws']} and {b['converged']}/{b['screen_draws']}, respectively.")
    a,b,c=(r[n] for n in ("empirical-drift","learned-uflow","crystalflow-joint-local"))
    t.append(r"\paragraph{Earlier experiments with the same evaluator.} Using the empirical training-composition prior, the earlier fixed-basis checkpoint generated no unseen formulas and attained "+f"{100*a['recall']:.2f}"+r"\% coverage recall. Using one-step uFlowCSP on the identical learned-prior chemistry, our earlier replay attained "+f"{100*b['recall']:.2f}"+r"\% recall. Our separate replay of joint CrystalFlow sampled its own chemistry and geometry with 100 Euler steps, generating "+f"{c['unseen']:,}"+r" training-unseen formula draws among 10,000, "+f"{c['joint']/100:.2f}"+r"\% joint validity and "+f"{100*c['recall']:.2f}"+r"\% coverage recall. That bank took 29.33 minutes on the same GPU under its own batch-24 protocol. Six of its first sixteen proposals passed the restricted physical screen. Formula overlap, raw geometry, and screened physical yield measure different stages of the generator \cite{uflow,crystalflow}.")
    (HERE.parent/"paper/chemistry.tex").write_text("\n".join(t)+"\n",encoding="utf8")


if __name__=="__main__":
    write()
