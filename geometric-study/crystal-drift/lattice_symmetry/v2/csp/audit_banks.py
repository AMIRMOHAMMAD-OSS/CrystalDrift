"""Independently seal every raw candidate before retrospective structure matching."""
import hashlib
import json
import numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]

def hashfile(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def main():
    with np.load(ROOT/"revision/full/mp20/cohort.npz") as d:ids=d["ids"];z=d["z"]
    total=0;records={};selection=json.loads((HERE/"selection.json").read_text())
    for folder in sorted((HERE/"test").iterdir()):
        ready=json.loads((folder/"ready.json").read_text())
        m=ready["arm"];seed=ready["seed"]
        assert ready["targets"]==len(ids)==9046 and ready["draws"]==20 and ready["nfe"]==({"prior":0,"flow16":16}.get(m,1))
        state=HERE/"prior.npz" if m=="prior" else HERE/f"fits/{m}-{seed}/update-{selection[m]['updates']:05}.pt"
        assert hashfile(state)==ready["checkpoint_sha256"]
        if m!="prior":
            lock=HERE/"locks"/(ready["selection_sha256"]+".json")
            assert hashfile(lock)==ready["selection_sha256"] and json.loads(lock.read_text())[m]==selection[m]
        with np.load(folder/"crosswalk.npz") as a:np.testing.assert_equal(a["ids"],ids);np.testing.assert_equal(a["z"],z)
        kk=np.load(folder/"k.npy",mmap_mode="r");ff=np.load(folder/"f.npy",mmap_mode="r")
        assert kk.shape==(9046,20,6) and ff.shape==(9046,20,20,3)
        assert hashfile(folder/"k.npy")==ready["cell_sha256"] and hashfile(folder/"f.npy")==ready["sites_sha256"]
        for i in range(0,len(z),128):
            sel=slice(i,i+128);f=ff[sel];k=kk[sel];mask=(z[sel]>0)[:,None,:,None]
            assert np.isfinite(k).all() and np.isfinite(f).all() and (f>=0).all() and (f<=1).all()
            assert np.where(mask,0.,f).max()==0.
        records[folder.name]=dict(source=state.relative_to(ROOT).as_posix(),state_sha256=hashfile(state),manifest_sha256=hashfile(folder/"ready.json"))
        total+=kk.shape[0]*kk.shape[1]
    assert len(records)==18 and total==3256560
    (HERE/"raw-seal.json").write_text(json.dumps(dict(proposals=total,arms=records),indent=2)+"\n")
    print("Sealed",total,"finite, ordered outputs in",len(records),"banks.")

if __name__=="__main__":main()
