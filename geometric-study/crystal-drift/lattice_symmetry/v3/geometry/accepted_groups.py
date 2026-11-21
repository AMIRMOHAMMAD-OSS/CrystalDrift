"""Detect the raw space-group type of every accepted symbolic candidate on protected panels."""
import argparse
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[variable]="1"
import numpy as np
import spglib

from ..evaluation.storage import pool

HERE=Path(__file__).resolve().parent
OUT=HERE/"audits"


def audit(panel,tags):
    assert spglib.__version__=="2.5.0"
    with np.load(OUT/f"reachable-{panel}.npz") as f:
        ids=f["ids"].astype(str)
        reference=f["truth_group"].astype(int)
    dest=OUT/f"accepted-groups-{panel}"
    dest.mkdir(exist_ok=True)
    summary_path=dest/"summary.json"
    summary=json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    for tag in tags:
        scores=HERE.parent/f"evaluation/scores/{panel}/{tag}"
        with np.load(scores/"outcomes.npz") as f:
            assert np.array_equal(f["ids"].astype(str),ids)
            rms=f["rms"]
        assert rms.shape==(len(ids),20)
        detected=np.full(rms.shape,-1,np.int16)
        cursor=0
        for file in sorted(pool(panel,tag).glob("?????-?????.npz")):
            with np.load(file) as raw:
                count=len(raw["ids"])
                assert np.array_equal(raw["ids"].astype(str),ids[cursor:cursor+count])
                lattice,sites,species=(raw[key] for key in ("L","F","Z"))
                for i,j in np.argwhere(np.isfinite(rms[cursor:cursor+count])):
                    mask=species[i,j]>0
                    geometry=(lattice[i,j].astype(float),sites[i,j,mask].astype(float),species[i,j,mask])
                    answer=spglib.get_symmetry_dataset(geometry,symprec=.01)
                    detected[cursor+i,j]=0 if answer is None else int(answer.number)
                cursor+=count
        assert cursor==len(ids) and np.array_equal(detected>=0,np.isfinite(rms))
        same=(detected==reference[:,None])&(detected>0)
        rows={}
        for k in (1,5,20):
            hits=np.isfinite(rms[:,:k])
            best=np.where(hits,rms[:,:k],np.inf).argmin(axis=1)
            any_right=same[:,:k].any(axis=1)
            rows[k]=dict(targets=len(ids),matched_targets=int(hits.any(axis=1).sum()),
                matched_draws=int(hits.sum()),failed_detections=int((detected[:,:k]==0).sum()),
                correct_group_draws=int(same[:,:k].sum()),correct_group_any_draw=int(any_right.sum()),
                correct_group_best_draw=int(same[np.arange(len(ids)),best].sum()))
            if tag!="prior":
                with np.load(dest/"prior.npz") as p:
                    assert np.array_equal(p["ids"],ids)
                    prior_match=(p["detected"][:,:k]>=0).any(axis=1)
                new=hits.any(axis=1)&~prior_match
                rows[k].update(added_vs_prior=int(new.sum()),correct_group_on_added=int((new&any_right).sum()))
        np.savez_compressed(dest/f"{tag}.npz",ids=ids,rms=rms,reference=reference,detected=detected)
        summary[tag]=dict(symprec_A=.01,spglib=spglib.__version__,rows=rows)
        summary_path.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8")
        print(panel,tag,rows[20],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel",choices=("family","shared"))
    parser.add_argument("tags",nargs="+")
    args=parser.parse_args()
    audit(args.panel,args.tags)
