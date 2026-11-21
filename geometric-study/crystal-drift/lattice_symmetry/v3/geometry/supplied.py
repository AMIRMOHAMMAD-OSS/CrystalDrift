"""Compare the frozen orbit models with identical priors under inferred and supplied occupations."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import time

for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[k]="1"
import numpy as np

HERE=Path(__file__).resolve().parent
V3=HERE.parent
ROOT=V3.parents[1]
OUT=HERE/"audits/supplied-confirm"
ARMS=("prior","angular-medium-long","contacts-medium-long")
REGIMES=("inferred","supplied")
K=5


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def export_inputs():
    from .reachability import records
    from ..evaluation.predict import setup_split,draw_seed
    cohort,split=setup_split("confirm")
    assert split=="val" and len(cohort)==96
    labels={r["id"]:r for r in records(split)}
    clean=[]
    for mid,ix,z in cohort:
        r=labels[mid]
        rec=dict(id=mid,species=sorted(map(int,z[z>0])),seed=draw_seed("confirm",ix))
        if "error" in r:rec["error"]="unavailable reference occupation"
        else:
            rec.update(group=int(r["group"]),hall=int(r["hall"]),
                       occupation=sorted((int(e),str(l)) for e,l in r["template"]))
        clean.append(rec)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"inputs.json").write_text(json.dumps(clean,indent=1)+"\n",encoding="utf8")


def generate():
    import torch
    from ..models.sampler import Sampler
    from ..evaluation.predict import atomic_npz,array,h
    records=json.loads((OUT/"inputs.json").read_text())
    freeze=V3/"evaluation/clock-frozen.json"
    register=json.loads(freeze.read_text())
    assert register["frozen"]
    specs={a["tag"]:a for a in register["methods"]}
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    service=Sampler("cpu")
    retained=set(service.ids)
    selected={}
    for arm in ARMS:
        item=specs[arm]
        if "checkpoint" in item:
            path=V3/item["checkpoint"]
            assert h(path)==item["sha"]
            selected[arm]=service.load(path)
        else:selected[arm]=None
    with np.load(HERE/"native-operations.npz") as ops:
        halls=ops["halls"]
    run=dict(register_sha256=digest(freeze),protocol_sha256=digest(HERE/"SUPPLIED-OCCUPATION.md"),
             sanitized_inputs_sha256=digest(OUT/"inputs.json"),device="cpu",threads=torch.get_num_threads(),
             torch=torch.__version__,python=platform.python_version(),model_sha={a:specs[a].get("sha") for a in ARMS},
             targets=len(records),draws=K,output_seconds={})
    for regime in REGIMES:
        start=time.perf_counter()
        shape=(len(records),K)
        groups=np.zeros(shape,np.int16);donors=np.full(shape,"",dtype="U32");cells=donors.copy()
        occupied=np.full(shape,"",dtype="U300");errors=np.full(shape,"",dtype="U150")
        drawsets=[]
        for i,r in enumerate(records):
            z=np.array(r["species"])
            rng=np.random.default_rng(r["seed"])
            if regime=="supplied":
                if "error" in r:
                    errors[i,:]=r["error"];continue
                g=int(r["group"])
                assert int(halls[g-1])==int(r["hall"]),(r["id"],g,r["hall"],halls[g-1])
                options=[(g,tuple((e,l) for e,l in r["occupation"]),"")]
            else:options=service.shortlist(tuple(z),False)
            for j in range(K):
                option=options[j%len(options)]
                groups[i,j]=option[0];donors[i,j]=option[2]
                occupied[i,j]=json.dumps(option[1],separators=(",",":"))
                try:
                    seed=service.draw(z,option,rng)
                    assert seed.cell_donor in retained and (not seed.donor or seed.donor in retained)
                    assert np.array_equal(np.sort(seed.Z),z)
                    cells[i,j]=seed.cell_donor
                    drawsets.append((i,j,seed))
                except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                    errors[i,j]=f"{type(exc).__name__}: {exc}"
        for arm,net in selected.items():
            L,F,Z=array(len(records),K)
            for start_at in range(0,len(drawsets),16):
                batch=drawsets[start_at:start_at+16]
                outs=service.apply(net,[q[2] for q in batch])
                for (i,j,seed),(l,f,z) in zip(batch,outs):
                    L[i,j]=l;F[i,j,:len(z)]=f;Z[i,j,:len(z)]=z
            dest=OUT/f"{regime}-{arm}.npz"
            atomic_npz(dest,ids=[r["id"] for r in records],L=L,F=F,Z=Z,group=groups,occupation=occupied,
                       errors=errors,cell_donor=cells,donor=donors)
        run["output_seconds"][regime]=time.perf_counter()-start
    (OUT/"settings.json").write_text(json.dumps(run,indent=2)+"\n",encoding="utf8")


def score():
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from standard_match import make_structure,SETTINGS
    from data import pack_gram
    assert version("pymatgen")=="2024.4.13"
    ref=ROOT/"scaled/mp20-data/val.npz"
    with np.load(ref) as d:
        index={str(v):i for i,v in enumerate(d["ids"])}
        targets={r["id"]:make_structure(d["k"][i],d["f"][i],d["z"][i])
                 for r in json.loads((OUT/"inputs.json").read_text()) for i in [index[r["id"]]]}
    match=StructureMatcher(**SETTINGS)
    for regime in REGIMES:
        for arm in ARMS:
            source=OUT/f"{regime}-{arm}.npz"
            with np.load(source) as d:
                ids=d["ids"];Z=d["Z"];L=d["L"];F=d["F"];errors=d["errors"].copy()
            rms=np.full(Z.shape[:2],np.inf);direct=np.full_like(rms,np.inf);vol=np.full_like(rms,np.nan)
            for i,mid in enumerate(ids):
                target=targets[str(mid)]
                for j in range(K):
                    if errors[i,j]:continue
                    active=Z[i,j]>0
                    try:
                        native=Structure(L[i,j],Z[i,j,active],F[i,j,active])
                        assert native.volume>.100001
                        gram=L[i,j]@L[i,j].T
                        proposed=make_structure(pack_gram(gram),F[i,j],Z[i,j])
                        assert sorted(proposed.atomic_numbers)==sorted(target.atomic_numbers)
                        vol[i,j]=np.log(proposed.volume/target.volume)
                        result=match.get_rms_dist(proposed,target)
                        if result is not None:rms[i,j]=result[0]
                        other=match.get_rms_dist(native,target)
                        if other is not None:direct[i,j]=other[0]
                    except (FloatingPointError,ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                        errors[i,j]=f"{type(exc).__name__}: {exc}"
            np.savez_compressed(OUT/f"{regime}-{arm}-scores.npz",ids=ids,rms=rms,direct_native_frame_rms=direct,log_volume_ratio=vol,
                                errors=errors,source_sha256=digest(source))
    (OUT/"matcher.json").write_text(json.dumps(dict(settings=SETTINGS,source_sha256=digest(ref),
        pymatgen=version("pymatgen"),spglib=version("spglib"),numpy=version("numpy"),threads=1,
        frame="Main endpoint uses the same log-metric encode, decode and Cholesky as evaluation/score.py. Direct native-frame public scores are saved as a numerical sensitivity."),indent=2)+"\n")


def report():
    from ..evaluation.analysis import ci
    endpoints={};hits={};frames={};archived={};failures={}
    for regime in REGIMES:
        for arm in ARMS:
            key=regime+"/"+arm
            with np.load(OUT/f"{regime}-{arm}-scores.npz") as d:
                r=d["rms"];vol=d["log_volume_ratio"];e=d["errors"];ids=d["ids"]
                assert ids.tolist()==[x["id"] for x in json.loads((OUT/"inputs.json").read_text())]
                assert str(d["source_sha256"])==digest(OUT/f"{regime}-{arm}.npz")
                direct=d["direct_native_frame_rms"];common=np.isfinite(r)&np.isfinite(direct)
                frames[key]=dict(hit_disagreements=int((np.isfinite(r)!=np.isfinite(direct)).sum()),
                                 max_rms_difference=float(abs(r[common]-direct[common]).max()))
                failures[key]={str(ids[i]):[str(v) for v in e[i]] for i in range(len(ids)) if np.any(e[i]!="")}
            if regime=="inferred":
                stem="prior/confirm" if arm=="prior" else f"medium-{arm.split('-')[0]}-laplace-11-8pass/frozen-confirm"
                earlier=V3/("models/fits/"+stem)
                with np.load(earlier.with_suffix(".npz")) as old,np.load(OUT/f"{regime}-{arm}.npz") as new:
                    assert np.array_equal(old["ids"],new["ids"])
                    oldz=old["Z"]
                    if oldz.ndim==2:oldz=np.broadcast_to(oldz[:,None,:],new["Z"].shape)
                    assert np.array_equal(oldz,new["Z"])
                    delta=(new["F"]-old["F"])[oldz>0]
                    size=float(np.max(np.abs(new["L"]-old["L"])))
                    sites=float(np.max(np.abs(delta-np.rint(delta))))
                with np.load(earlier.with_name(earlier.name+"-scores.npz")) as old:
                    assert np.array_equal(old["ids"],ids)
                    changed=np.isfinite(old["rms"])!=np.isfinite(r)
                    archived[arm]=dict(archived_hits=int(np.isfinite(old["rms"]).any(1).sum()),
                        hit_disagreements=int(changed.sum()),ids=ids[changed.any(1)].tolist(),
                        max_cell_entry_A=size,max_wrapped_fractional_site_entry=sites)
            endpoints[key]={};hits[key]={}
            for k in (1,5):
                best=r[:,:k].min(1);hit=np.isfinite(best)
                native=(np.isfinite(r[:,:k])&(np.abs(vol[:,:k])<=np.log(1.1))).any(1)
                hits[key][k]=hit
                endpoints[key][k]=dict(hits=int(hit.sum()),n=len(hit),native=int(native.sum()),
                      best_rms=float(best[hit].mean()) if hit.any() else None,errors=int((e[:,:k]!="").sum()))
    pairs={}
    for arm in ARMS:
        contrasts=[("supplied/"+arm,"inferred/"+arm)]
        if arm!="prior":contrasts.extend([(g+"/"+arm,g+"/prior") for g in REGIMES])
        for a,b in contrasts:
            pairs[a+" minus "+b]={}
            for k in (1,5):
                h,h0=hits[a][k],hits[b][k];diff=h.astype(float)-h0
                pairs[a+" minus "+b][k]=dict(learned_or_supplied_only=int((h&~h0).sum()),
                    reference_only=int((h0&~h).sum()),pp=100*diff.mean(),interval=(100*np.array(ci(diff))).tolist())
    result=dict(endpoints=endpoints,paired=pairs,direct_frame_check=frames,earlier_confirmation=archived,
                failures_by_target=failures)
    (OUT/"result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print({key:v[5] for key,v in endpoints.items()},flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=("inputs","generate","score","report"))
    {"inputs":export_inputs,"generate":generate,"score":score,"report":report}[ap.parse_args().stage]()
