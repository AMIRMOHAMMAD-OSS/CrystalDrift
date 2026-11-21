"""Compare CHGNet-relaxed binary proposals with composition-weighted elemental endpoints."""
import bz2
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import data
import relax_candidates as relaxation


HERE = Path(__file__).resolve().parent
ORIGINAL = HERE/"de-novo"/"relaxation"
OUT = HERE/"de-novo"/"elemental-screen"


def choose_elements():
    b=np.load(HERE/"de-novo"/"candidates.npz")
    needed=set(b["z"].flat)-{0}
    choices={}
    for split in ("test","val","train"):
        p=data.DATA/f"mp20_{split}.csv.bz2"
        if not p.exists():p=HERE.parent/"crystal"/"data"/p.name
        with bz2.open(p,"rt",encoding="utf-8",newline="") as f:
            for row in csv.DictReader(f):
                head,_,body=row["cif"].partition("_atom_site_occupancy\n")
                sites=[line.split() for line in body.strip().splitlines()]
                symbol=sites[0][0];el=data.Z[symbol]
                if el not in needed or any(v[0]!=symbol for v in sites):continue
                priority=(float(row["e_above_hull"]),float(row["formation_energy_per_atom"]),len(sites),row["material_id"])
                if el in choices and choices[el][0]<=priority:continue
                fields={s.split(maxsplit=1)[0]:s.split(maxsplit=1)[1] for s in head.splitlines() if s.startswith("_cell_")}
                xyz=[[float(t) for t in s[3:6]] for s in sites]
                choices[el]=(priority,dict(sample_id=row["material_id"],z=[el]*len(sites),role="elemental_endpoint",
                                            f=xyz,G=data.cell_gram(fields).tolist(),split=split,
                                            generator_formula=symbol,source_e_above_hull=priority[0],
                                            source_formation_energy_per_atom=priority[1]))
    return {el:item for el,(_,item) in choices.items()}


def initialize():
    relaxation.OUTPUT=OUT
    relaxation.load_model()


def main():
    for folder in (OUT/"records",OUT/"cifs"):folder.mkdir(parents=True,exist_ok=True)
    selected=choose_elements()
    (OUT/"selection.json").write_text(json.dumps(list(selected.values()),indent=1)+"\n")
    with ProcessPoolExecutor(max_workers=4,initializer=initialize) as pool:
        records=list(pool.map(relaxation.relax,selected.values()))
    endpoints={r["sample_id"]:r for r in records}
    archive=np.load(HERE/"de-novo"/"candidates.npz")
    rows=[]
    for mid,zz in zip(archive["ids"],archive["z"]):
        z,counts=np.unique(zz[zz>0],return_counts=True)
        c=json.loads((ORIGINAL/"records"/f"{mid}.json").read_text())
        required=[selected[i]["sample_id"] for i in z if i in selected]
        refs=[endpoints[i] for i in required]
        result=dict(id=mid,formula=c["species_formula"],candidate_converged=c["converged"],
                    elemental_MP_ids=" ".join(required),endpoints_converged=all(r["converged"] for r in refs),
                    element1_fraction=counts[0]/counts.sum())
        if len(required)==len(z) and "final_energy_eV_per_atom" in c and all("final_energy_eV_per_atom" in r for r in refs):
            base=float(np.average([r["final_energy_eV_per_atom"] for r in refs],weights=counts))
            result.update(final_energy_eV_per_atom=c["final_energy_eV_per_atom"],elemental_mixture_energy_eV_per_atom=base,
                          delta_vs_elements_eV_per_atom=c["final_energy_eV_per_atom"]-base)
        rows.append(result)
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (OUT/"candidate-gaps.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    print("Candidates compared",sum("delta_vs_elements_eV_per_atom" in r for r in rows),flush=True)


if __name__=="__main__":
    main()
