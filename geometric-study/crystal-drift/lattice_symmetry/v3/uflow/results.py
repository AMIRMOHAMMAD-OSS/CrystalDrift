"""Rebuild complete-cohort comparisons from per-target scores, clustered by target or held family."""
import csv
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def read(path):
    with np.load(path) as source:
        return {k:source[k] for k in ("ids","rms","strict_fit","log_volume_ratio")}

def reorder(data,ids):
    lookup={str(mid):i for i,mid in enumerate(data["ids"])}
    indices=[lookup[str(mid)] for mid in ids]
    return {k:v[indices] for k,v in data.items()}

def ci(values,groups):
    if not len(values):return [None,None]
    names,idx=np.unique(groups,return_inverse=True)
    numerator=np.bincount(idx,weights=values)
    denom=np.bincount(idx)
    rng=np.random.default_rng(7331)
    # Target bootstrap of ternary paired indicators has exactly the same sampling
    # distribution as drawing the empirical counts once per bootstrap replicate.
    if len(names)==len(values) and len(np.unique(values))<=9:
        categories,counts=np.unique(values,return_counts=True)
        cases=rng.multinomial(len(values),counts/len(values),size=5000)
        draws=cases@categories/len(values)
        return [round(float(x),5) for x in np.percentile(draws,[2.5,97.5])]
    draws=[]
    for first in range(0,5000,32):
        sampled=rng.integers(0,len(names),size=(min(32,5000-first),len(names)))
        draws.extend(numerator[sampled].sum(1)/denom[sampled].sum(1))
    return [round(float(x),5) for x in np.percentile(draws,[2.5,97.5])]

def by_draw(d,k):
    rms=d["rms"][:,:k]
    mask=np.isfinite(rms)
    native=mask&(np.abs(d["log_volume_ratio"][:,:k])<=np.log(1.1))
    return (mask.any(1),np.where(mask,rms,np.inf).min(1),
            native.any(1),np.where(native,rms,np.inf).min(1))

def dump(rows,path):
    if not rows:return
    with path.open("w",encoding="utf8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0])
        writer.writeheader();writer.writerows(rows)

def main():
    protocol=json.loads((HERE.parent/"protocol/splits.json").read_text())
    released=read(ROOT/"revision/full/external/mp20/complete-crosswalk-k20.npz")
    train_formulas=set()
    for cache in (HERE/"train-cache").glob("*.npz"):
        with np.load(cache) as f:
            train_formulas.update(tuple(sorted(z[z>0])) for z in f["z"])
    summary=[];paired=[]
    for panel,split in (("test","test"),("family","train"),("shared","train"),("confirm","val")):
        ids=protocol[panel+"_ids"]
        groups=np.array([protocol["family_by_challenge_id"][m] if panel in ("family","shared") else m for m in ids])
        with np.load(ROOT/f"scaled/mp20-data/{split}.npz") as ref:
            lookup={str(mid):i for i,mid in enumerate(ref["ids"])}
            atoms=ref["z"][[lookup[i] for i in ids]]
            sizes=(atoms>0).sum(1)
            seen=np.array([tuple(sorted(z[z>0])) in train_formulas for z in atoms])
        models={}
        for path in sorted((HERE/"scores").glob(f"*/frozen/{panel}/outcomes.npz")):
            data=read(path)
            assert set(data["ids"])==set(ids)
            models[path.parents[2].name]=reorder(data,ids)
        if panel=="test":
            models["released-separate-noise"]=reorder(released,ids)
        for model,data in models.items():
            for k in (1,5,20):
                if data["rms"].shape[1]<k:continue
                ok,best,native,native_best=by_draw(data,k)
                for label,sel in [("all",np.ones(len(ok),bool)),("1-4",sizes<=4),
                     ("5-8",(sizes>4)&(sizes<=8)),("9-12",(sizes>8)&(sizes<=12)),("13-20",sizes>12),("9+",sizes>8),
                     ("seen-formula",seen),("unseen-formula",~seen)]:
                    n=int(sel.sum())
                    if not n:continue
                    matched=sel&ok
                    summary.append(dict(panel=panel,model=model,k=k,sites=label,n=n,hits=int(matched.sum()),
                         mr_pct=float(100*ok[sel].mean()),native_hits=int((native&sel).sum()),
                         strict_hits=int(data["strict_fit"][sel,:k].any(1).sum()),
                         matched_rms=float(best[matched].mean()) if matched.any() else None))
                for comparator,other in models.items():
                    contrast=("prior-11","released-separate-noise",model.removesuffix("-long"))
                    if comparator==model or comparator not in contrast or other["rms"].shape[1]<k:
                        continue
                    ob,other_best,ovn,other_native_best=by_draw(other,k)
                    for endpoint,ours,baseline,ours_rms,baseline_rms in [("scale",ok,ob,best,other_best),
                            ("native",native,ovn,native_best,other_native_best)]:
                        for label,mask in (("all",np.ones(len(ids),bool)),("9+",sizes>=9)):
                            if not mask.any():continue
                            diff=ours[mask].astype(float)-baseline[mask]
                            lower,upper=ci(100*diff,groups[mask])
                            matched=ours&baseline&mask
                            rms_low,rms_high=ci(ours_rms[matched]-baseline_rms[matched],groups[matched])
                            paired.append(dict(panel=panel,model=model,baseline=comparator,k=k,endpoint=endpoint,sites=label,
                                targets=int(mask.sum()),clusters=len(set(groups[mask])),gained=int((ours&~baseline&mask).sum()),
                                lost=int((baseline&~ours&mask).sum()),delta_pp=float(100*diff.mean()),
                                low95=lower,high95=upper,both=int(matched.sum()),
                                common_rms_model=float(ours_rms[matched].mean()) if matched.any() else None,
                                common_rms_baseline=float(baseline_rms[matched].mean()) if matched.any() else None,
                                common_rms_diff_low95=rms_low,common_rms_diff_high95=rms_high))
    dump(summary,HERE/"recovery.csv");dump(paired,HERE/"paired.csv")
if __name__=="__main__":main()
