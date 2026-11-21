"""Track the pinned DiffCSP property subset for every completed raw-geometry panel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import sleep
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matcher-python",type=Path,required=True)
    p.add_argument("--metrics-site",type=Path,required=True)
    p.add_argument("--pyg-site",type=Path,required=True)
    p.add_argument("--arms",nargs="+",default=["fixed-basis","radial","angular"])
    p.add_argument("--wait",action="store_true")
    a=p.parse_args()
    env=os.environ.copy()
    env["CRYSTAL_METRICS_SITE"]=str(a.metrics_site.resolve())
    env["CRYSTAL_PYG_SITE"]=str(a.pyg_site.resolve())
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):env[key]="1"
    for arm in a.arms:
        panel=HERE/"panel"/(arm+".json")
        while not (panel.exists() and json.loads(panel.read_text()).get("fingerprints_complete",False) and "structural_novel" in json.loads(panel.read_text())):
            if not a.wait:raise SystemExit("The complete panel is unavailable for "+arm)
            sleep(15)
        for phase in ("graphs","energy"):
            cmd=[str(a.matcher_python),str(HERE/"property.py"),phase,arm,"--workers","1"]
            subprocess.run(cmd,env=env,cwd=ROOT,check=True)
        subprocess.run([str(a.matcher_python),str(HERE/"evaluate.py"),"report","--arms",arm],env=env,cwd=ROOT,check=True)
        if "w1_formation_energy" not in json.loads(panel.read_text()):
            raise RuntimeError("Property output failed to join to the 1,000 sampled cases of "+arm)
        energies=HERE/"energies"/(arm+".npz")
        with np.load(energies) as e,np.load(HERE/"panel"/(arm+"-cases.npz")) as c:
            assert len(e["index"])==1000 and np.array_equal(e["index"],np.sort(c["w1_sample"]))
            np.testing.assert_array_equal(e["ids"],c["ids"][e["index"]])
            assert np.isfinite(e["energy_eV_per_atom"]).all() and c["valid"][e["index"]].all()
        weight=ROOT/"expansion/denovo/property/diffcsp/prop_models/mp20/epoch=839-step=89039.ckpt"
        reference=HERE/"energies/reference.npz"
        record=dict(arm=arm,checkpoint_relative=weight.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(weight),
                    energy_file_sha256=sha(energies),reference_count=9046,reference_sha256=sha(reference),
                    subset="1,000 seeded jointly valid IDs after the raw+fingerprint panel",property_subset_seed=7121,
                    W1_eV_per_atom=json.loads(panel.read_text())["w1_formation_energy"])
        (HERE/"energies"/(arm+"-provenance.json")).write_text(json.dumps(record,indent=2)+"\n")
        print(arm,record["W1_eV_per_atom"],flush=True)


if __name__=="__main__":main()
