"""Validation follow-up for angular CrytalDrift. Run fit, then validate with the pinned matcher."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CSP = HERE.parent / "csp"
SEEDS = (11,23,37)
MILESTONES = (256,1024,3392)
SHRINKAGE = 0.25
TOL = np.log(1.1)

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def design(z):
    """The original elemental ridge plus a sparse dictionary of within-formula element pairs."""
    n = (z > 0).sum(1)
    x = np.zeros((len(z),139))
    row, col = np.nonzero(z)
    np.add.at(x, (row,z[row,col]), 1/n[row])
    x[np.arange(len(z)),118+n] = 1
    rows, cols, values = [], [], []
    for i,row in enumerate(x[:,:119]):
        species = np.flatnonzero(row)
        a,b = np.triu_indices(len(species),1)
        first,second = species[a],species[b]
        rows.extend([i]*len(a))
        cols.extend(second*(second-1)//2 + first)
        values.extend(2*np.sqrt(row[first]*row[second]))
    pairs = sparse.csr_matrix((values,(rows,cols)),shape=(len(z),119*118//2))
    return x,pairs,n

def fit():
    train_file = ROOT/"scaled/mp20-data/train.npz"
    with np.load(train_file) as records:
        x,pairs,n = design(records["z"])
        truth = records["k"][:,:3].sum(1,dtype=float)
    with np.load(CSP/"prior.npz") as prior:
        base = prior['coefficients']
    err = truth - np.log(n) - x@base
    beta = lsqr(pairs,err,damp=1.,atol=1e-10,btol=1e-10,iter_lim=600)[0]
    np.savez_compressed(HERE/'volume_head.npz',base=base,pair=beta,shrinkage=SHRINKAGE,
                        train_sha256=digest(train_file),prior_sha256=digest(CSP/'prior.npz'))
    return {'rows':len(n),'mean_absolute_train_log_volume_error':float(abs(err-pairs@beta).mean())}

class VolumeHead:
    """Isotropic, composition-conditioned readout. Fractions and shape are copied verbatim."""
    def __init__(self):
        with np.load(HERE/'volume_head.npz') as f:
            self.base=f['base']; self.pair=f['pair']; self.shrinkage=float(f['shrinkage'])
        a,b=np.triu_indices(119,k=1)
        self.interaction=np.zeros((119,119))
        self.interaction[a,b]=self.interaction[b,a]=self.pair[b*(b-1)//2+a]
    def expected(self,z):
        unique,index = np.unique(z,axis=0,return_inverse=True)
        # Independent of site order and linear in the number of site slots.
        n=(unique>0).sum(1)
        counts=np.zeros((len(unique),119))
        rows,cols=np.nonzero(unique)
        np.add.at(counts,(rows,unique[rows,cols]),1)
        fractions=counts/n[:,None]
        roots=np.sqrt(fractions)
        residual=(roots*(roots@self.interaction)).sum(1)
        return (np.log(n)+self.base[118+n]+fractions@self.base[:119]+residual)[index]
    def apply(self,packed,mean):
        calibrated = np.asarray(packed,dtype=float).copy()
        lv=calibrated[...,:3].sum(axis=-1)
        shift=(1-self.shrinkage)*(mean[:,None]-lv)/3
        calibrated[...,:3]+=shift[...,None]
        return calibrated

def load_calibrated_service(seed=11,device='cpu'):
    """Same (z, rng) -> (k, f) interface as the release, with one generator evaluation."""
    sys.path.insert(0,str(ROOT))
    from lattice_symmetry.v2.csp.model import load_service
    angular = load_service('angular',seed,device)
    head = VolumeHead()
    def propose(z,rng):
        k,f = angular(z,rng)
        return head.apply(k[:,None,:],head.expected(z))[:,0], f
    return propose

def interval(rows):
    # Paired bootstrap targets, fitted seeds retained together.
    rng=np.random.default_rng(52824); vals=[]
    for _ in range(50):
        ix=rng.integers(len(rows),size=(100,len(rows)))
        vals.extend(rows[ix].mean(1))
    return list(map(float,np.quantile(vals,[.025,.975])))

def summary(flags,logs):
    rec={}
    for d in sorted(set([1,min(5,flags.shape[-1]),flags.shape[-1]])):
        for tag,arr in [('scaled',flags),('native',flags & (abs(logs)<=TOL))]:
            per=arr[:,:,:d].any(2)
            rec[f'{tag}_{d}']={'matches_by_seed':per.sum(1).tolist(),'mean':float(per.mean())}
    rec['mean_absolute_log_volume']=float(abs(logs).mean())
    rec['signed_mean_log_volume_difference']=float(logs.mean())
    return rec

def contrasts(cases):
    out={}
    base_flags,base_logs=cases['angular-256']
    for tag,(flags,logs) in cases.items():
        if tag=='angular-256':continue
        for d in sorted(set([1,min(5,flags.shape[-1]),flags.shape[-1]])):
            for ep in ['scaled','native']:
                a = base_flags if ep=='scaled' else base_flags & (abs(base_logs)<=TOL)
                b = flags if ep=='scaled' else flags & (abs(logs)<=TOL)
                paired=b[:,:,:d].any(2).astype(float)-a[:,:,:d].any(2)
                per=paired.mean(0)
                out[f'{tag}-{ep}-{d}']={'delta':float(per.mean()),'ci95':interval(per)}
    return out

def validate():
    import importlib.metadata
    assert importlib.metadata.version('pymatgen') == '2024.4.13'
    sys.path.insert(0,str(ROOT/'revision/full'))
    from match_full import target_context, candidate
    from data import unpack_gram
    with np.load(CSP/'fits/angular-11/val-00256.npz') as pred:
        ids=pred['ids']; z=pred['z']
    with np.load(ROOT/'scaled/mp20-data/val.npz') as db:
        lookup={v:i for i,v in enumerate(db['ids'])}
        indices=[lookup[v] for v in ids]
        np.testing.assert_array_equal(db['z'][indices],z)
        kt,ft=db['k'][indices],db['f'][indices]
    head=VolumeHead()
    means=head.expected(z)
    key=lambda u,s: CSP/f'fits/angular-{s}/val-{u:05}.npz'
    ctx=[target_context(k,f,zz) for k,f,zz in zip(kt,ft,z)]
    tags=[(f'angular-{u}',u,False) for u in MILESTONES]+[('calibrated-256',256,True)]
    cases={}
    provenance={}
    for label,u,cal in tags:
        flags=np.full((len(SEEDS),len(z),5),False)
        logs=np.zeros(flags.shape)
        sources=[]
        for s,seed in enumerate(SEEDS):
            with np.load(key(u,seed)) as raw:
                np.testing.assert_array_equal(ids,raw['ids'])
                np.testing.assert_array_equal(z,raw['z'])
                k=raw['k'].reshape(len(z),5,6).astype(float)
                f=raw['f'].reshape(len(z),5,20,3)
                sources.append(str(raw['checkpoint_sha256']))
            if cal:k=head.apply(k,means)
            logs[s]=k[:,:,:3].sum(2)-kt[:,:3].sum(1,dtype=float)[:,None]
            g=unpack_gram(k)
            for i,zz in enumerate(z):
                atomic=zz[zz>0].astype(np.int64)
                for j in range(5):
                    hit,_,_=candidate(ctx[i],k[i,j],f[i,j],atomic,g[i,j],False)
                    flags[s,i,j]=np.isfinite(hit)
        cases[label]=(flags,logs)
        provenance[label]=sources
        print(label,summary(flags,logs),flush=True)
    report={'validation_targets':len(z),'seed_ids':list(SEEDS),'draws':5,
            'design':'Fixed first 128 validation IDs and saved common five draws per seed. Full matcher rerun per candidate for every arm.',
            'checkpoint_sha256':provenance,'calibration_sha256':digest(HERE/'volume_head.npz'),
            'arms':{k:summary(*v) for k,v in cases.items()},'paired_target_differences':contrasts(cases),
            'scaled_per_draw_changes_after_calibration':int(np.sum(cases['calibrated-256'][0]!=cases['angular-256'][0]))}
    (HERE/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(HERE/'validation_outcomes.npz',ids=ids,**{k+'_'+name:vals[j] for k,vals in cases.items() for j,name in enumerate(['hit','log_volume_ratio'])})

def archived_scale_sensitivity():
    """Reuse archived scaled matches; read out native-volume agreement for fixed coordinates and shape."""
    with np.load(ROOT/'revision/full/mp20/cohort.npz') as rec:
        ids=rec['ids']; z=rec['z']; truth=rec['k'][:,:3].sum(1,dtype=float)
    head=VolumeHead()
    means=head.expected(z)
    flags=[]; old=[]; new=[]
    for seed in SEEDS:
        base=CSP/f'test/angular-seed{seed}'
        with np.load(base/'outcomes.npz') as scores:
            np.testing.assert_array_equal(ids,scores['ids'])
            flags.append(np.isfinite(scores['rms']) & ~scores['error'].astype(bool))
            old.append(scores['log_volume_ratio'].astype(float))
        k=np.load(base/'k.npy',mmap_mode='r')
        predicted=k[:,:,:3].sum(2,dtype=float)
        np.testing.assert_allclose(predicted-truth[:,None],old[-1],atol=1e-6)
        new.append((head.shrinkage*predicted+(1-head.shrinkage)*means[:,None])-truth[:,None])
    flags=np.asarray(flags)
    cases={'angular-256':(flags,np.asarray(old)),'calibrated-256':(flags,np.asarray(new))}
    report={'N':len(ids),'training_seeds':list(SEEDS),
            'protocol':'Fixed archived candidate fractional sites, shape and released scaled match flags. Recalculate native-volume agreement (factor 1.1); no full test matcher rerun, see validation for paired direct re-matching.',
            'head_sha256':digest(HERE/'volume_head.npz'),'arms':{k:summary(*v) for k,v in cases.items()},
            'paired_native_at_20':interval((((flags & (abs(np.asarray(new))<=TOL)).any(2)).astype(float)-(flags & (abs(np.asarray(old))<=TOL)).any(2)).mean(0))}
    (HERE/'archived_volume_sensitivity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

def independent_rematch():
    """Rerun the matcher after volume changes on 256 deterministic test IDs disjoint from validation."""
    import importlib.metadata
    assert importlib.metadata.version('pymatgen') == '2024.4.13'
    sys.path.insert(0,str(ROOT/'revision/full'))
    from match_full import target_context,candidate
    from data import unpack_gram
    with np.load(ROOT/'revision/full/mp20/cohort.npz') as records:
        all_ids=records['ids']
        prefix='volume-head-forward-check-v1:'
        ix=np.asarray(sorted(range(len(all_ids)),key=lambda i:hashlib.sha256((prefix+str(all_ids[i])).encode()).digest())[:256])
        ids=all_ids[ix]
        z,kt,ft=(records[key][ix] for key in ('z','k','f'))
    ctx=[target_context(k,f,s) for k,f,s in zip(kt,ft,z)]
    truth=kt[:,:3].sum(1,dtype=float)
    head=VolumeHead()
    means=head.expected(z)
    old_flags=[];old_logs=[];new_flags=[];new_logs=[]
    for seed in SEEDS:
        base=CSP/f'test/angular-seed{seed}'
        k=np.asarray(np.load(base/'k.npy',mmap_mode='r')[ix],dtype=float)
        f=np.load(base/'f.npy',mmap_mode='r')[ix]
        with np.load(base/'outcomes.npz') as old:
            np.testing.assert_array_equal(ids,old['ids'][ix])
            old_flags.append(np.isfinite(old['rms'][ix]) & ~old['error'][ix].astype(bool))
            old_logs.append(old['log_volume_ratio'][ix])
        k=head.apply(k,means);g=unpack_gram(k)
        flags=np.zeros(k.shape[:2],dtype=bool)
        for i,row in enumerate(z):
            atoms=row[row>0].astype(np.int64)
            for j in range(k.shape[1]):
                rms,_,_=candidate(ctx[i],k[i,j],f[i,j],atoms,g[i,j],False)
                flags[i,j]=np.isfinite(rms)
        new_flags.append(flags);new_logs.append(k[:,:,:3].sum(2)-truth[:,None])
        print('direct test rematch, seed',seed,'done',flush=True)
    cases={'angular-256':(np.array(old_flags),np.array(old_logs)),'calibrated-256':(np.array(new_flags),np.array(new_logs))}
    report={'N':len(ids),'id_salt':prefix,'seeds':list(SEEDS),'draws':20,
            'scaled_draw_switches':int(np.sum(cases['angular-256'][0]!=cases['calibrated-256'][0])),
            'arms':{tag:summary(*values) for tag,values in cases.items()},'differences':contrasts(cases)}
    np.savez_compressed(HERE/'direct_test_check_outcomes.npz',ids=ids,**{tag+'_'+name:values[i] for tag,values in cases.items() for i,name in enumerate(('hit','log_volume_ratio'))})
    (HERE/'direct_test_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['scaled_draw_switches'],report['differences']['calibrated-256-native-20'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('stage',choices=['fit','validation','archive','rematch'])
    args=parser.parse_args()
    if args.stage=='fit':print(fit())
    elif args.stage=='validation':validate()
    elif args.stage=='archive':archived_scale_sensitivity()
    else:independent_rematch()
