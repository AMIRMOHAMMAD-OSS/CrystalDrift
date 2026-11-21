"""Validate archive indexing, exact count-prior replay and the physical cell chart."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def main(source):
    path=HERE.parent/"denovo/pools/crystalflow-joint-local.npz"
    with np.load(path) as row:
        ids,z,k,f,L,valid,counts=[row[key] for key in ("ids","z","k","f","lattice","valid_output","num_sites_drawn")]
    assert ids.tolist()==[f"crystalflow-replay-{i:05d}" for i in range(10000)]
    assert z.shape==(10000,20) and f.shape==(10000,20,3) and L.shape==(10000,3,3)
    assert valid.shape==(10000,) and ((z>=0)&(z<=118)).all()
    assert np.array_equal(counts,(z>0).sum(1))
    assert ((np.arange(20)[None,:]<counts[:,None])==(z>0)).all()
    assert (np.diff(np.where(z>0,z,999),axis=1)>=0).all()
    assert (f[z==0]==0).all()
    assert (f[valid]>=0).all() and (f[valid]<=1).all()
    tree=ast.parse((source/"scripts/generation.py").read_text())
    dist,=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="train_dist" for t in n.targets)]
    prior,=[ast.literal_eval(v) for key,v in zip(dist.value.keys,dist.value.values) if ast.literal_eval(key)=="mp_20"]
    np.testing.assert_equal(counts,np.random.RandomState(241127).choice(len(prior),10000,p=prior))
    q=k[valid];s=np.zeros((len(q),3,3))
    s[:,np.arange(3),np.arange(3)]=q[:,:3]
    i,j=np.triu_indices(3,1);s[:,i,j]=s[:,j,i]=q[:,3:]/np.sqrt(2)
    lam,v=np.linalg.eigh(s);gram=(v*np.exp(2*lam)[:,None,:])@v.transpose(0,2,1)
    native=L[valid]@L[valid].transpose(0,2,1)
    np.testing.assert_allclose(gram,native,rtol=1e-9,atol=1e-8)
    report=dict(draws=len(z),native_output_finite=int(valid.sum()),failed_draws=np.flatnonzero(~valid).tolist(),
        count_prior="Exact replay of seeded MP-20 size distribution",
        min_sites=int(counts.min()),max_sites=int(counts.max()),unique_ids=len(set(ids)),
        gram_max_discrepancy_angstrom_squared=float(np.max(abs(gram-native))),
        source_checkpoint_sha256="abd359240252e6b2fe1740f006b6d80ee6df2f4b11a60f7030f4c9a75f617780",
        raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (HERE/"joint-local-audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",required=True,type=Path)
    main(p.parse_args().source)
