"""Freeze named weights from development-only selections; never inspect test or challenge scores."""
import argparse
import hashlib
import json
import os
from pathlib import Path
V3=Path(__file__).resolve().parents[1]
HERE=V3/"evaluation"

def digest(p):
    p=Path(p)
    if not p.exists():
        pointer=json.loads(p.with_suffix(p.suffix+".pointer.json").read_text())
        return digest(Path(pointer["windows_path" if os.name=="nt" else "path"]))
    return hashlib.sha256(p.read_bytes()).hexdigest()

def model(tag,folder,**extra):
    base=V3/"models/fits"/folder;selection=json.loads((base/"selection.json").read_text())
    p=base/selection["chosen"]
    assert digest(p)==selection["sha256"]
    return dict(tag=tag,checkpoint=str(p.relative_to(V3)).replace(chr(92),"/"),sha=digest(p),**extra)

def freeze(key,methods):
    data=dict(frozen=True,rule="prospective development-only register",methods=methods)
    path=HERE/(key+".json")
    if path.exists():
        if json.loads(path.read_text())!=data:raise ValueError("refusing to change a frozen comparison")
        return path
    tmp=path.with_suffix(f".{os.getpid()}.pending")
    tmp.write_text(json.dumps(data,indent=2)+"\n")
    try:
        os.link(tmp,path)
    except FileExistsError:
        if json.loads(path.read_text())!=data:raise ValueError("conflicting frozen comparison")
    finally:tmp.unlink(missing_ok=True)
    return path

def main():
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["first","baseline","replicates","extended"]);a=ap.parse_args()
    if a.stage=="first":
        # All three were predeclared, retained even when their winners tie.
        m=[dict(tag="prior")]
        m.extend(model(f"{arm}-small-11",f"small-{arm}-laplace-11") for arm in ("angular","contacts","reciprocal"))
        m.append(model("angular-medium-11","medium-angular-laplace-11"))
        freeze("first-frozen",m)
    elif a.stage=="baseline":
        m=[dict(tag="geometry-donor",family="donor"),model("geometry-donor-angular","small-angular-laplace-11",family="donor"),
           dict(tag="learned-rank",selector="learned"),model("learned-rank-angular","small-angular-laplace-11",selector="learned"),
           dict(tag="empirical-rank",selector="frequency")]
        freeze("baseline-frozen",m)
    elif a.stage=="replicates":
        m=[model(f"{arm}-small-{s}",f"small-{arm}-laplace-{s}") for arm in ("angular","contacts") for s in (23,37)]
        m.extend([model("meanflow-small-11","small-meanflow-laplace-11"),model("meanflow-medium-11","medium-meanflow-laplace-11"),
                  model("contacts-medium-11","medium-contacts-laplace-11")])
        freeze("replicates-frozen",m)
    else:raise SystemExit("Use actual registered extended selection paths after validation; freeze() accepts them independently.")

if __name__=="__main__":main()
