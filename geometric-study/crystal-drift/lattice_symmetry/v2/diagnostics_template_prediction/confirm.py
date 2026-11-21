"""One frozen formula-only test confirmation. All pilot choices remain fixed."""
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

from experiment import (OUT,ROOT,V2,SETTINGS,Structure,StructureMatcher,apply,donor_index,
                        model_and_hash,native,np,Prior,proposals,records,torch,unpack_gram)

N=384
K=5
SALT="crytaldrift-onecall-transfer-confirm:"
CONF=OUT/"confirmation"


def archived(path,ids):
    with np.load(path,allow_pickle=False) as d:
        index={str(mid):i for i,mid in enumerate(d["ids"])}
        if len(index)!=len(d["ids"]) or not set(ids)<=index.keys():
            raise ValueError("Repeated or missing reference ID")
        indices=[index[x] for x in ids]
        return np.asarray(d["rms"][indices,:K])


def main():
    torch.set_num_threads(1)
    CONF.mkdir(exist_ok=True)
    start=perf_counter()
    train=records()
    prior=Prior(train)
    with np.load(ROOT/"scaled/mp20-data/train.npz") as data:
        donors=donor_index(train,data["z"])
        train_ids=set(map(str,data["ids"]))
        train_formulas={tuple(z[z>0]) for z in data["z"]}
    model,checkpoint=model_and_hash("angular-C")
    with np.load(ROOT/"scaled/mp20-data/test.npz",allow_pickle=False) as raw:
        all_ids=raw["ids"].astype(str)
        order=sorted(range(len(all_ids)),key=lambda i:hashlib.sha256((SALT+all_ids[i]).encode()).digest())[:N]
        ids=all_ids[order];zs=raw["z"][order]
        assert len(set(ids))==N and not set(ids)&train_ids
        # No test geometry or reference symmetry labels are accessed during generation.
        maxn=zs.shape[1]
        L=np.full((N,K,2,3,3),np.nan)
        F=np.full((N,K,2,maxn,3),np.nan,dtype=np.float32)
        Z=np.zeros((N,K,2,maxn),dtype=np.int16)
        template_meta=[]
        for i,index in enumerate(order):
            options=proposals(zs[i],donors,K)
            rng=np.random.default_rng(np.random.SeedSequence([70693,int(index)]))
            seeds=[prior.draw(g,letters,rng) for g,letters,_ in options]
            updated=apply(model,seeds)
            for j,(g,letters,parent) in enumerate(options):
                rr=dict(id=ids[i],draw=j+1,group=g,donor=parent,pattern=letters)
                for a,item in enumerate((seeds[j],updated[j])):
                    try:
                        s=native(item,g,zs[i])
                        L[i,j,a]=s.lattice.matrix;F[i,j,a,:len(s)]=s.frac_coords
                        Z[i,j,a,:len(s)]=s.atomic_numbers
                    except ValueError as e:
                        rr[f"error_{a}"]=str(e)
                rr["pattern"]=[(int(z),letter) for z,letter in letters]
                template_meta.append(rr)
        np.savez_compressed(CONF/"predictions.npz",ids=ids,L=L,F=F,Z=Z)
        (CONF/"symbolic-choices.json").write_text(json.dumps(template_meta,separators=(",",":"))+"\n")
        print(f"{N} × {K} candidate pairs frozen before scoring, {perf_counter()-start:.1f}s",flush=True)
        # Open selected reference geometries only after all candidates are frozen on disk.
        target_cells=np.linalg.cholesky(unpack_gram(raw["k"][order]))
        target_f=raw["f"][order]
    rms=np.full((N,K,2),np.nan)
    vol=np.full_like(rms,np.nan)
    matcher=StructureMatcher(**SETTINGS)
    for i in range(N):
        target=Structure(target_cells[i],zs[i,zs[i]>0],target_f[i,zs[i]>0])
        for j in range(K):
            for a in range(2):
                mask=Z[i,j,a]>0
                if not mask.any():continue
                crystal=Structure(L[i,j,a],Z[i,j,a,mask],F[i,j,a,mask])
                result=matcher.get_rms_dist(crystal,target)
                vol[i,j,a]=np.log(crystal.volume/target.volume)
                if result is not None:rms[i,j,a]=result[0]
    src=ROOT/"revision/full/external/mp20/complete-crosswalk-k20.npz"
    methods={"symbolic_prior":rms[:,:,0],"symbolic_angular":rms[:,:,1],"uflow":archived(src,ids)}
    for seed in (11,23,37):
        methods[f"A{seed}"]=archived(V2/f"csp/test/angular-seed{seed}/outcomes.npz",ids)
    seen=np.array([tuple(z[z>0]) in train_formulas for z in zs])
    np.savez_compressed(CONF/"scores.npz",ids=ids,log_volume_ratio=vol,seen=seen,**methods)
    summary={"n":N,"k":K,"salt":SALT,"train_overlap":0,"missing_archived_ids":0,
             "generation_errors":int(sum("error_0" in q or "error_1" in q for q in template_meta)),
             "checkpoint":checkpoint,"results":{}}
    for k in (1,K):
        hits={m:np.isfinite(mat[:,:k]).any(1).astype(int) for m,mat in methods.items()}
        hits["Amean"]=np.stack([hits["A11"],hits["A23"],hits["A37"]]).mean(0)
        details={}
        for mask_name,mask in (("all",np.ones(N,bool)),("unseen_formula",~seen),("seen_formula",seen),("9plus",(zs>0).sum(1)>=9)):
            details[mask_name]=dict(n=int(mask.sum()),hits={m:float(v[mask].sum()) for m,v in hits.items()})
        details["native_vol_hits"]={m:int((np.isfinite(rms[:,:k,a])&(np.abs(vol[:,:k,a])<=np.log(1.1))).any(1).sum())
                                   for a,m in enumerate(("symbolic_prior","symbolic_angular"))}
        for other in ("symbolic_prior","Amean","uflow"):
            delta=hits["symbolic_angular"]-hits[other]
            rng=np.random.default_rng(111589)
            interval=np.quantile(delta[rng.integers(0,N,(5000,N))].mean(1),[.025,.975])
            details["versus_"+other]={"delta":float(delta.mean()),"ci95":interval.tolist()}
        summary["results"][k]=details
    (CONF/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary["results"][K],indent=2),flush=True)


if __name__=="__main__":
    main()
