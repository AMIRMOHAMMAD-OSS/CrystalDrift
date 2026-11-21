"""Exploratory @20 extension of the fixed 384-target panel; retains every frozen prefix."""
import json
from time import perf_counter
from confirm import CONF,N,K,ROOT,V2,records,Prior,donor_index,model_and_hash,proposals,native,apply,Structure,StructureMatcher,SETTINGS,unpack_gram,np,torch


def main():
    torch.set_num_threads(1)
    start=perf_counter()
    out=CONF/"extension";out.mkdir(exist_ok=True)
    with np.load(CONF/"predictions.npz") as prefix:
        ids=prefix["ids"];l5,f5,z5=[prefix[k] for k in ("L","F","Z")]
    with np.load(CONF/"scores.npz") as old:
        oldr=np.stack((old["symbolic_prior"],old["symbolic_angular"]),axis=-1)
        oldv=old["log_volume_ratio"]
    rows=records();prior=Prior(rows)
    with np.load(ROOT/"scaled/mp20-data/train.npz") as train:
        donors=donor_index(rows,train["z"])
    net,meta=model_and_hash("angular-C")
    M=20
    L=np.full((N,M,2,3,3),np.nan);F=np.full((N,M,2,*f5.shape[-2:]),np.nan,dtype=np.float32)
    Z=np.zeros((N,M,2,z5.shape[-1]),dtype=np.int16)
    L[:,:K]=l5; F[:,:K]=f5; Z[:,:K]=z5
    with np.load(ROOT/"scaled/mp20-data/test.npz") as raw:
        ix={str(v):i for i,v in enumerate(raw["ids"])}
        order=[ix[q] for q in ids];zs=raw["z"][order];choices=[]
        for i,index in enumerate(order):
            opts=proposals(zs[i],donors,M)[K:]
            rng=np.random.default_rng(np.random.SeedSequence([70693,int(index),6]))
            seeds=[prior.draw(g,pairs,rng) for g,pairs,_ in opts]
            updated=apply(net,seeds)
            for j,(g,pairs,donor) in enumerate(opts,K):
                rec=dict(id=str(ids[i]),draw=j+1,group=g,donor=donor,pattern=pairs)
                for a,item in enumerate((seeds[j-K],updated[j-K])):
                    try:
                        s=native(item,g,zs[i]);L[i,j,a]=s.lattice.matrix
                        F[i,j,a,:len(s)]=s.frac_coords;Z[i,j,a,:len(s)]=s.atomic_numbers
                    except ValueError as exc:rec[f"error_{a}"]=str(exc)
                choices.append(rec)
        np.savez_compressed(out/"predictions.npz",ids=ids,L=L,F=F,Z=Z)
        (out/"suffix-choices.json").write_text(json.dumps(choices,separators=(",",":"))+"\n")
        target_cells=np.linalg.cholesky(unpack_gram(raw["k"][order]))
        targets=raw["f"][order]
    print(f"Prefix frozen, new 5,760 proposal pairs saved at {perf_counter()-start:.1f}s",flush=True)
    matcher=StructureMatcher(**SETTINGS)
    vol=np.full((N,M,2),np.nan);rms=np.full_like(vol,np.nan)
    rms[:,:K]=oldr;vol[:,:K]=oldv
    parity=[]
    for i in range(N):
        active=zs[i]>0
        target=Structure(target_cells[i],zs[i,active],targets[i,active])
        for j in range(M):
            for a in range(2):
                mask=Z[i,j,a]>0
                if not mask.any():continue
                if j>=K:
                    crystal=Structure(L[i,j,a],Z[i,j,a,mask],F[i,j,a,mask])
                    vol[i,j,a]=np.log(crystal.volume/target.volume)
                    match=matcher.get_rms_dist(crystal,target)
                    if match is not None:rms[i,j,a]=match[0]
            # Check public matcher invariance under isotropic correction on the full @5 prefix
            # and ~1/13 of further draws. No target is used to compute the stretch.
            if (j<K or (i*71+j)%13==0) and np.isfinite(L[i,j]).all():
                v0,v1=np.abs(np.linalg.det(L[i,j]))
                preserved=L[i,j,1]*(v0/v1)**(1/3)
                m=Z[i,j,1]>0
                match=matcher.get_rms_dist(Structure(preserved,Z[i,j,1,m],F[i,j,1,m]),target)
                parity.append([i,j,np.isfinite(rms[i,j,1]),match is not None])
    np.testing.assert_equal(rms[:,:K],oldr)
    np.testing.assert_equal(vol[:,:K],oldv)
    def archive(path):
        with np.load(path) as data:
            lookup={str(v):i for i,v in enumerate(data["ids"])}
            return data["rms"][[lookup[q] for q in ids],:M]
    archiveA=[archive(V2/f"csp/test/angular-seed{s}/outcomes.npz") for s in (11,23,37)]
    u=archive(ROOT/"revision/full/external/mp20/complete-crosswalk-k20.npz")
    np.savez_compressed(out/"scores.npz",ids=ids,rms=rms,vol=vol,uflow=u,angular=archiveA,parity=parity)
    summarize(out)


def summarize(out):
    with np.load(out/"scores.npz") as data:
        rms,vol,ids,u,archiveA,parity=[data[x] for x in ("rms","vol","ids","uflow","angular","parity")]
    M=rms.shape[1]
    summary={"targets":N,"draws":M,"confirmed_prefix_unchanged":True,"errors":int((~np.isfinite(vol).all(axis=2)).sum()),
             "volume_matchparity":{"checked":len(parity),"difference_count":int((parity[:,2]!=parity[:,3]).sum())}}
    for k in (1,5,20):
        matches=np.isfinite(rms[:,:k]);native_ok=np.abs(vol[:,:k])<=np.log(1.1)
        hp,hc=matches.any(1).T.astype(float);hu=np.isfinite(u[:,:k]).any(1).astype(float)
        ha=np.stack([np.isfinite(s[:,:k]).any(1) for s in archiveA]).mean(0)
        np0,nc=(matches&native_ok).any(1).T.astype(float)
        both=np.isfinite(vol[:,:k]).all(axis=2)
        keep=(matches[:,:,1]&native_ok[:,:,0]&both).any(1).astype(float)
        d={"counts":{"prior":int(hp.sum()),"C":int(hc.sum()),"Amean":float(ha.sum()),"uflow":int(hu.sum()),
                     "native_prior":int(np0.sum()),"native_C":int(nc.sum()),"native_C_keep_prior_volume":int(keep.sum())}}
        brng=np.random.default_rng(111589);straps=brng.integers(len(ids),size=(5000,len(ids)))
        for n,change in (("C_minus_prior",hc-hp),("C_minus_uflow",hc-hu),("C_minus_Amean",hc-ha),("native_keep_minus_C",keep-nc)):
            d[n]={"delta":float(change.mean()),"ci95":np.quantile(change[straps].mean(1),[.025,.975]).tolist()}
        summary[k]=d
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__": main()
