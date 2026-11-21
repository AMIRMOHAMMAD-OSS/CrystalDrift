"""Resolve upstream code to commits and inspect spatial kernels, updates and evaluator entry points."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import requests

HERE=Path(__file__).resolve().parent
# Restrict source reads to files implementing the primary method named in the study.
SPECS=[
 ("deng","lambertae/drifting",r"(drift|loss)",["softmax","stop_gradient","temperature"]),
 ("symdrift","samirdarouich/SymDrift",r"(loss|field|embed|drift)",["kabsch","sort","softmax"]),
 ("lego","MaterSim/LEGO-xtal",r"(relax|local|SO3)",["SO3","optimize_xtal"]),
 ("gems","aklipf/GemsNet",r"(gemsnet|deformation|vector)",["lattice","deformation","cell"]),
 ("vector","aklipf/gemsdiff",r"(field|model|denois)",["vector","lattice","basis"]),
 ("diffcsp","jiaor17/DiffCSP",r"(compute_metrics|diffusion|cspnet)",["StructureMatcher","stol","cos"]),
 ("diffcspplus","jiaor17/DiffCSP-PP",r"(diffusion|wyckoff|sample)",["anchor","wyckoff","lattice"]),
 ("orbitol","stases/orbitol",r"(sample|wyckoff|orbit)",["wyckoff","orbit","expand"]),
 ("sgfm","omri1348/SGFM",r"(sgfm_model|flow_model|data_utils|crystal_family)",["wyckoff","symmetry"]),
 ("uflow","usccolumbia/uFlowCSP",r"(test|meanflow|sampler|model)",["meanflow","sample","crystal_system"])]

def read(url):
    r=requests.get(url,timeout=(6,14),headers={"User-Agent":"CrytalDrift source audit"})
    r.raise_for_status();return r

def inspect(spec):
    key,repo,pattern,terms=spec
    base="https://api.github.com/repos/"+repo
    record={"repo":repo,"examples":{}}
    try:
        branch=read(base).json()["default_branch"]
        tree=read(base+"/git/trees/"+branch+"?recursive=1").json()
        commit=tree["sha"];record["commit"]=commit
        files=[a["path"] for a in tree["tree"] if a["path"].endswith((".py",".ipynb")) and re.search(pattern,a["path"],re.I)]
        files=sorted(files,key=lambda s: (0 if any(x in Path(s).stem.lower() for x in ("loss","diffusion","compute_metrics","field","drift","wyckoff","embed","relax")) else 1,len(s)))
        for path in files[:10]:
            if len(record["examples"])==len(terms):break
            url="https://raw.githubusercontent.com/"+repo+"/"+commit+"/"+path
            r=read(url);lines=r.text.splitlines()
            for term in terms:
                if term in record["examples"]:continue
                ix=next((j for j,s in enumerate(lines,1) if term.lower() in s.lower()),None)
                if ix:
                    record["examples"][term]=dict(file=path,line=ix,
                        link=f"https://github.com/{repo}/blob/{commit}/{path}#L{ix}",
                        sha256=hashlib.sha256(r.content).hexdigest())
    except Exception as err:record["retrieval_error"]=str(err)
    print(key,len(record["examples"]),record.get("retrieval_error",""),flush=True)
    return key,record


if __name__=="__main__":
    store=HERE/"code-locations.json"
    existing=json.loads(store.read_text()) if store.exists() else {}
    todo=[s for s in SPECS if existing.get(s[0],{}).get("repo")!=s[1]
          or existing.get(s[0],{}).get("retrieval_error") or not existing.get(s[0],{}).get("examples")]
    with ThreadPoolExecutor(3) as exe: rows={**existing,**dict(exe.map(inspect,todo))}
    (HERE/"code-locations.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf8")
