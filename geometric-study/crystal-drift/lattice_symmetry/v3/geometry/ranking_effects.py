"""Target-paired changes in setting-equivalent occupation coverage for frozen rankings."""
import json
from pathlib import Path
import numpy as np
from ..evaluation.analysis import ci

HERE=Path(__file__).resolve().parent;OUT=HERE/"audits"
split=json.loads((HERE.parent/"protocol/splits.json").read_text())
secondary=json.loads((OUT/"arrangement-frozen.json").read_text())
res={}
for file in sorted(OUT.glob("ranking-coverage-*.npz")):
    panel=file.stem.removeprefix("ranking-coverage-")
    with np.load(file) as d:
        methods=d["methods"].tolist();fields=d["fields"].tolist()
        eq=d["flags"][:,fields.index("equivalent")];ids=d["ids"].tolist()
    groups=None
    if panel in ("family","shared"):
        groups=[split["family_by_challenge_id"][i] for i in ids]
    elif panel.startswith("arrangement"):
        groups=[secondary["key_by_id"][i] for i in ids]
    res[panel]={}
    for j,m in enumerate(methods[1:],start=1):
        comp={}
        for k in (1,5,20):
            sel,chem=eq[[j,0],:,:k].any(2)
            delta=sel.astype(float)-chem.astype(float)
            comp[k]=dict(ranking_only=int((sel&~chem).sum()),lookup_only=int((chem&~sel).sum()),
                points=100*float(delta.mean()),ci_95_points=(100*np.array(ci(delta,groups))).tolist())
        res[panel][m]=comp
(OUT/"ranking-paired.json").write_text(json.dumps(res,indent=2)+"\n",encoding="utf-8")
print(res["test"])
