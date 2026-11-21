"""Freeze matched-budget geometric comparisons at one full exposure without selecting a duration."""
import json
from register import V3,freeze,digest

def row(arm,seed=11):
    fit=V3/"models/fits"/f"small-{arm}-laplace-{seed}"
    check=fit/"full.pt"
    if not check.is_file():raise FileNotFoundError(check)
    return dict(tag=f"{arm}-small-full-{seed}",checkpoint=str(check.relative_to(V3)).replace(chr(92),"/"),sha=digest(check))
if __name__=="__main__":
    methods=[row(a) for a in ("angular","contacts","reciprocal","meanflow")]
    freeze("same-exposure-frozen",methods)
