"""Read the WyckoffDiff MP-20 release and assess binary primitive crystal samples."""
import argparse
import bz2
import csv
import hashlib
import io
import json
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist, pdist


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BASE = "https://public.openmaterialsdb.se/wyckoffdiff/data/mp20/raw"
ELEMENTS = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()
Z = {s: i for i, s in enumerate(ELEMENTS)}
SQRT_TWO = np.sqrt(2.)
SHELL_CUTOFF = 0.5  # Å, conventional MP-20 validity threshold


def pack_gram(G):
    """Return six Euclidean coordinates of half the matrix logarithm of G."""
    eig, q = np.linalg.eigh(G)
    if np.any(eig <= 0):
        raise ValueError("Lattice Gram matrix must be positive definite")
    s = (q * (np.log(eig) / 2)[..., None, :]) @ q.swapaxes(-1, -2)
    return np.stack((s[..., 0, 0], s[..., 1, 1], s[..., 2, 2], SQRT_TWO*s[..., 0, 1],
                     SQRT_TWO*s[..., 0, 2], SQRT_TWO*s[..., 1, 2]), axis=-1)


def unpack_gram(k):
    """Decode (...,6) log stretch coordinates to (...,3,3) Gram matrices in Å²."""
    k = np.asarray(k, dtype=float)
    s = np.zeros((*k.shape[:-1], 3, 3), dtype=k.dtype)
    for j in range(3):
        s[..., j, j] = k[..., j]
    for j, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
        s[..., a, b] = s[..., b, a] = k[..., j+3] / SQRT_TWO
    eig, q = np.linalg.eigh(s)
    return (q * np.exp(2*eig)[..., None, :]) @ q.swapaxes(-1, -2)


def cell_gram(values):
    lengths = np.array([float(values[f"_cell_length_{a}"]) for a in "abc"])
    ca, cb, cg = np.cos(np.deg2rad([float(values[f"_cell_angle_{a}"]) for a in ("alpha", "beta", "gamma")]))
    return np.outer(lengths, lengths) * np.array([[1., cg, cb], [cg, 1., ca], [cb, ca, 1.]])


def extract(split):
    path = DATA / f"mp20_{split}.csv.bz2"
    if split == "train" and not path.is_file():
        path = HERE.parent / "crystal" / "data" / path.name
    raw = path.read_bytes()
    decoded = bz2.decompress(raw)  # verifies compressed end-of-stream, including checksums
    rows = csv.DictReader(io.StringIO(decoded.decode("utf-8"), newline=""))
    all_ids = set()
    eligible = []
    for row_number, row in enumerate(rows, 1):
        mid = row["material_id"]
        if mid in all_ids:
            raise ValueError(f"{split} contains duplicate material ID {mid}")
        all_ids.add(mid)
        cif = row["cif"]
        head, marker, atoms = cif.partition("_atom_site_occupancy\n")
        if not marker:
            raise ValueError(f"Missing CIF atom loop at {mid}")
        operations = head.partition("_symmetry_equiv_pos_as_xyz\n")[2].partition("loop_\n")[0].strip()
        if "_symmetry_Int_Tables_number   1\n" not in head or operations != "1  'x, y, z'":
            raise ValueError(f"Expected explicit P1 coordinate list at {mid}")
        positions = [line.split() for line in atoms.strip().splitlines()]
        if not 2 <= len(positions) <= 8 or len({site[0] for site in positions}) != 2:
            continue
        if any(len(site) != 7 or float(site[2]) != 1 or float(site[6]) != 1 for site in positions):
            raise ValueError(f"Partial occupancy or repeated site in the P1 atom loop at {mid}")
        positions.sort(key=lambda site: Z[site[0]])  # stable within each species
        z = np.zeros(8, dtype=np.int16)
        z[:len(positions)] = [Z[site[0]] for site in positions]
        x = np.zeros((8, 3))
        x[:len(positions)] = np.array([[float(v) for v in site[3:6]] for site in positions]) % 1.
        fields = {part[0]: part[1] for line in head.splitlines() if line.startswith("_cell_")
                  for part in [line.split(maxsplit=1)] if len(part) == 2}
        G = cell_gram(fields)
        if not np.isclose(np.linalg.det(G)**0.5, float(fields["_cell_volume"]), rtol=2.e-6, atol=1.e-5):
            raise ValueError(f"CIF volume does not agree with Gram matrix at {mid}")
        eligible.append((mid, row_number, z, x, G))
    ids, ordinal, z, xyz, G = map(np.array, zip(*eligible))
    k = pack_gram(G)
    if not np.allclose(unpack_gram(k), G, atol=5.e-10):
        raise ArithmeticError("Lattice round-trip failed")
    n_atoms = (z != 0).sum(1)
    np.savez_compressed(DATA / f"multi-{split}.npz", ids=ids, source_record=ordinal, z=z, f=xyz,
                        k=k, G=G, n_atoms=n_atoms)
    two = n_atoms == 2
    pair = z[two, :2]
    relative = (xyz[two, 1] - xyz[two, 0]) % 1.
    np.savez_compressed(DATA / f"{split}.npz", ids=ids[two], source_record=ordinal[two], z=pair,
                        f=relative, G=G[two], k=k[two])
    summary = {"release_url": f"{BASE}/mp20_{split}.csv.bz2", "bz2_sha256": hashlib.sha256(raw).hexdigest(),
               "release_rows": len(all_ids), "P1_identity_checked": len(all_ids), "two_atom_binary": int(two.sum()),
               "distinct_two_atom_pairs": len(set(map(tuple, pair))), "binary_2_to_8": int(len(n_atoms)),
               "binary_by_atom_count": dict(sorted(Counter(map(int, n_atoms)).items()))}
    return summary, all_ids


def _direct_neighbors(G, x, na, radius):
    """Fast exhaustive symmetric image box for a batch with identical stoichiometry."""
    nbatch, n, _ = x.shape
    y = np.array(list(product(range(-radius,radius+1), repeat=3)), dtype=float)
    yy = np.einsum("si,bij,sj->bs", y, G, y)
    yy[:,np.all(y == 0,axis=1)] = np.inf
    same_atom = yy.min(1)
    answer = np.stack([same_atom, np.full(nbatch,np.inf), same_atom], axis=1)
    i,j = np.triu_indices(n,1)
    delta = x[:,j,:] - x[:,i,:]
    delta -= np.floor(delta+0.5)
    dG = delta @ G
    dd = (dG * delta).sum(-1)
    yy[:,np.all(y == 0,axis=1)] = 0.
    d2 = np.maximum(dd[:,:,None] + 2.*(dG @ y.T) + yy[:,None,:],0.).min(-1)
    tags = np.where(j<na,0,np.where(i>=na,2,1))
    for s in range(3):
        if np.any(tags == s):
            answer[:,s] = np.minimum(answer[:,s],d2[:,tags==s].min(1))
    return answer


def _exact_neighbors_cell(G, x, na, bound):
    """Sphere enumeration inside the supplied valid upper bounds (dimension three)."""
    r = np.linalg.cholesky(G).T
    costs = bound.copy()
    def improve(d, maximum, exclude_zero):
        value = np.empty(3)
        best = float(maximum)
        def search(axis, partial):
            nonlocal best
            if axis < 0:
                if not exclude_zero or np.any(value != 0.):
                    best = min(best, partial)
                return
            offset = r[axis,axis+1:] @ value[axis+1:]
            center = -offset/r[axis,axis]-d[axis]
            width = np.sqrt(max(0.,best-partial))/r[axis,axis] + 2e-13
            for integer in range(int(np.ceil(center-width)), int(np.floor(center+width))+1):
                value[axis] = integer + d[axis]
                candidate = partial + (r[axis,axis]*value[axis]+offset)**2
                if candidate <= best * (1.+2e-14):
                    search(axis-1, candidate)
        search(2,0.)
        return best
    # The same-site lattice vector is admissible for both same-element classes.
    norm2 = improve(np.zeros(3), max(costs[0],costs[2]), True)
    costs[[0,2]] = np.minimum(costs[[0,2]],norm2)
    for i,j in zip(*np.triu_indices(len(x),1)):
        s = 0 if j<na else 2 if i>=na else 1
        delta = x[j] - x[i]
        costs[s] = improve(delta-np.floor(delta+.5),costs[s],False)
    return costs


def geometry_features(k, f, z):
    """Return volume/atom and shortest AA, AB, BB distances for each binary cell.

    CC coordinates are row vectors with norm² x G xᵀ. An image beyond [-r,r]^3
    has a component >=r+0.5 after wrapping delta into [-0.5,0.5). Since
    |x_i| <= sqrt((G^-1)_ii) ||x||_G, a box minimum below this bound is exact.
    Uncertified entries use an exact three-dimensional Cholesky sphere search.
    """
    k,f,z = np.asarray(k,float),np.asarray(f,float),np.asarray(z)
    if k.ndim != 2 or z.ndim != 2 or k.shape != (len(z),6):
        raise ValueError("Specify one lattice and composition for each sample")
    if f.shape == (len(z),3) and z.shape[1] == 2:
        f = np.stack([np.zeros_like(f),f],axis=1)
    if f.shape != (*z.shape,3):
        raise ValueError("Coordinates must have shape (n,atoms,3), or (n,3) for AB")
    feature = np.full((len(k),4),np.nan)
    valid = np.isfinite(k).all(-1) & np.isfinite(f).all(axis=(1,2))
    # Beyond this envelope the exponential metric is unreliable in float64.
    # Count these samples as unevaluable rather than silently clipping predictions.
    valid &= np.max(np.abs(k),axis=1) < 12
    n = (z>0).sum(1)
    na = (z==z[:,:1]).sum(1)
    for atoms,first_species in np.unique(np.column_stack([n,na]),axis=0):
        if not 0 < first_species < atoms:
            raise ValueError("Expected two sorted occupied species and zero padding")
        group = np.flatnonzero(valid & (n==atoms) & (na==first_species))
        for start in range(0,len(group),192):
            idx = group[start:start+192]
            G = unpack_gram(k[idx]); xc = f[idx,:atoms]
            eigen = np.linalg.eigvalsh(G)
            usable = (eigen[:,0] > eigen[:,-1]*1.e-10) & (eigen[:,0]>1.e-12)
            idx,G,xc = idx[usable],G[usable],xc[usable]
            if not len(idx):continue
            diag_inv = np.diagonal(np.linalg.inv(G),axis1=1,axis2=2).max(1)
            distance2 = _direct_neighbors(G,xc,first_species,1)
            unresolved = np.flatnonzero(distance2.max(1)*diag_inv >= 1.5**2*(1.-1e-11))
            if len(unresolved):
                distance2[unresolved] = _direct_neighbors(G[unresolved],xc[unresolved],first_species,2)
                for t in unresolved[distance2[unresolved].max(1)*diag_inv[unresolved] >= 2.5**2*(1.-1e-11)]:
                    distance2[t] = _exact_neighbors_cell(G[t],xc[t],first_species,distance2[t])
            feature[idx,0] = np.sqrt(np.linalg.det(G))/atoms
            feature[idx,1:] = np.sqrt(np.maximum(distance2,0.))
    return feature


def _observed_features(split):
    if "observed_features" in split:
        return np.asarray(split["observed_features"], float)
    return geometry_features(split["k"],split["f"],split["z"])


def retrieval_baseline(test, train, draws, seed):
    """Draw a compatible recorded geometry without using held-out coordinates.

    The multisite fallback retains the lower/higher Z multiplicities whenever
    available. For a novel stoichiometry it selects a count-compatible cell by
    total site count, whose sites are then decorated with the requested elements.
    """
    rng = np.random.default_rng(seed)
    tz,sz = train["z"],test["z"]
    tn = (tz>0).sum(1); ta = (tz==tz[:,:1]).sum(1)
    tnuniq = np.column_stack([tz[:,0],tz.max(1)])
    indices = np.empty((len(sz),draws),int)
    for i,row in enumerate(sz):
        n,a = int((row>0).sum()),int((row==row[0]).sum())
        exact = np.flatnonzero(np.all(tz==row,axis=1))
        if not len(exact):
            pool = np.flatnonzero((tn==n)&(ta==a))
            if not len(pool):pool=np.flatnonzero(tn==n)
            if not len(pool):raise ValueError(f"No train geometry with {n} occupied sites")
            distance = np.abs(tnuniq[pool]-[row[0],row.max()]).sum(1)
            exact = pool[distance==distance.min()]
        indices[i] = rng.choice(exact,size=draws,replace=True)
    return train["k"][indices],train["f"][indices],indices


def _corr(x):
    finite = np.isfinite(x).all(1)
    if np.sum(finite) < 2 or np.any(np.std(x[finite],axis=0) < 1.e-12):return None
    return float(np.corrcoef(x[finite].T)[0,1])


def _lattice_length_correlations(k):
    v = np.asarray(k,float).reshape(-1,6)
    finite = np.isfinite(v).all(1) & (np.max(np.abs(v),axis=1)<12)
    if not np.any(finite):return None
    log_axes = .5*np.log(np.diagonal(unpack_gram(v[finite]),axis1=1,axis2=2))
    return {name:_corr(log_axes[:,indices]) for name,indices in
            (("a_b",[0,1]),("a_c",[0,2]),("b_c",[1,2]))}


def _evaluate(k, f, test, train, multi):
    """Evaluate ordered condition-specific ensembles against the same release split."""
    z = test["z"]
    k,f = np.asarray(k),np.asarray(f)
    if k.ndim == 2:k,f=k[:,None],f[:,None]
    if k.ndim != 3 or k.shape[0] != len(z) or k.shape[-1]!=6 or k.shape[:2] != f.shape[:2]:
        raise ValueError("Predictions must be ordered (held-out item, draw, ...)")
    budget = 5 if multi else 20
    k,f = k[:,:budget],f[:,:budget]
    n,d = k.shape[:2]
    sample = geometry_features(k.reshape(-1,6),f.reshape((n*d,)+f.shape[2:]),np.repeat(z,d,axis=0)).reshape(n,d,4)
    hold = _observed_features(test)
    fit = _observed_features(train)
    if not (np.isfinite(fit).all() and np.isfinite(hold).all()):
        raise ArithmeticError("Reference structures must be evaluable")
    # In an AB two-site cell AA and BB yield the same distance. Count it once.
    columns = [0,1,2,3] if multi else [0,1,2]
    transform = lambda a: np.log(np.maximum(a[...,columns],1.e-8))
    t,h,s = transform(fit),transform(hold),transform(sample)
    scale = t.std(0); mean = t.mean(0)
    scale[scale<1e-8]=1.
    t,h,s = (t-mean)/scale,(h-mean)/scale,(s-mean)/scale
    selected = np.random.default_rng(7301).choice(len(t),min(512,len(t)),replace=False)
    bandwidth = np.sqrt(np.median(pdist(t[selected],"sqeuclidean")))
    gamma = 0.5/bandwidth**2
    # Exact, biased RBF MMD² across held-out conditions for each draw index.
    # Each conditional energy score compares draws for one fixed formula.
    mmd = []
    finite = np.isfinite(s).all(-1)
    for j in range(d):
        ok = finite[:,j]
        if not np.any(ok):continue
        w,y = s[ok,j],h[ok]
        kernel = lambda a,b: np.exp(-gamma*cdist(a,b,"sqeuclidean")).mean()
        mmd.append(kernel(w,w)+kernel(y,y)-2.*kernel(w,y))
    energy, error, nearest_error, spread = [],[],[],[]
    for x,y,okay in zip(s,h,finite):
        accepted=x[okay]
        if len(accepted):
            dist=np.linalg.norm(accepted-y,axis=1)
            error.append(float(dist.mean())); nearest_error.append(float(dist.min()))
            cross = pdist(accepted)
            energy.append(float(dist.mean()-cross.sum()/len(accepted)**2))
            spread.append(float(cross.mean()) if len(cross) else 0.)
        else:
            energy.append(None);error.append(None);nearest_error.append(None);spread.append(None)
    close = np.nan_to_num(sample[:,:,1:],nan=-np.inf).min(-1)<SHELL_CUTOFF
    within = np.max(np.abs(np.log(np.maximum(sample,1.e-8)/hold[:,None,:])),axis=-1) <= np.log(1.15)
    truth_pairs = np.stack((z[:,0], z.max(1)),axis=1)
    fractions=(z==z[:,:1]).sum(1)/(z>0).sum(1)
    log_volume_error = np.abs(np.log(sample[:,:,0]/hold[:,None,0]))
    per_item_volume = [float(row[mask].mean()) if mask.any() else None for row,mask in zip(log_volume_error,finite)]
    def mean_or_none(values):
        values=[v for v in values if v is not None]
        return float(np.mean(values)) if values else None
    return {"conditions":n,"draws":d,"evaluable_fraction":float(finite.mean()),
            "numeric_failure_or_short_contact_fraction":float(close.mean()),
            "numeric_failure_or_short_contact_fraction_by_case":close.mean(1).tolist(),
            "evaluable_fraction_by_case":finite.mean(1).tolist(),
            "geometry_valid_fraction":float((finite & ~close).mean()),
            "conditional_energy":mean_or_none(energy),"conditional_energy_by_case":energy,
            "mean_standardized_environment_error":mean_or_none(error),
            "mean_best_of_draws_environment_error":mean_or_none(nearest_error),
            "within_15pct_environment_fraction":float((within & finite).mean()),
            "has_draw_within_15pct_environment_fraction":float((within & finite).any(1).mean()),
            "has_draw_within_15pct_environment_fraction_by_case":(within & finite).any(1).astype(float).tolist(),
            "mean_abs_log_volume_ratio_by_case":per_item_volume,
            "standardized_ensemble_spread_by_case":spread,
            "mean_draw_pooled_rbf_mmd2_v":mean_or_none(mmd),"train_rbf_bandwidth":float(bandwidth),
            "reference_log_volume_nearest_AB_correlation":_corr(np.log(hold[:,[0,2]])),
            "generated_log_volume_nearest_AB_correlation":_corr(np.log(np.maximum(sample[:,:,[0,2]].reshape(-1,2),1.e-8))),
            "reference_release_basis_log_length_correlations":_lattice_length_correlations(test["k"]),
            "generated_release_basis_log_length_correlations":_lattice_length_correlations(k),
            "reference_feature_medians":np.median(hold,axis=0).tolist(),
            "generated_feature_medians":np.nanmedian(sample,axis=(0,1)).tolist(),
            "species_conditioning":"lower-Z species first; exact supplied species fractions used for all samples",
            "mean_lower_Z_fraction":float(fractions.mean()),"distinct_element_pairs":len(np.unique(truth_pairs,axis=0))}


def evaluate_predictions(k, f, test, train):
    return _evaluate(k,f,test,train,False)


def evaluate_multi(k, f, test, train):
    return _evaluate(k,f,test,train,True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("splits", nargs="*", default=["train", "val", "test"])
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    summaries, ids = {}, {}
    for split in args.splits:
        summaries[split], ids[split] = extract(split)
        print(split, summaries[split], flush=True)
    pairs = [(a,b) for a in ids for b in ids if a < b]
    overlaps = {a+"_"+b:len(ids[a] & ids[b]) for a, b in pairs}
    if any(overlaps.values()):
        raise ValueError(f"Material IDs overlap between source splits: {overlaps}")
    if set(ids) == {"train","val","test"}:
        train = np.load(DATA / "train.npz")
        known = set(map(tuple, train["z"]))
        for split in ("val","test"):
            heldout = np.load(DATA / f"{split}.npz")
            summaries[split]["two_atom_rows_with_pair_in_train"] = sum(tuple(p) in known for p in heldout["z"])
        (DATA / "manifest.json").write_text(json.dumps({"splits":summaries, "full_split_id_overlap":overlaps},
                                                       indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
