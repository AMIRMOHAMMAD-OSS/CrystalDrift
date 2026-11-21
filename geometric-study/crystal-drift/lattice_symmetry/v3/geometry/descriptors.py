"""Finite-bank crystal features. All reciprocal lengths are cycles per angstrom."""
from functools import lru_cache
from math import pi
import numpy as np
import torch

from lattice_symmetry.v2.neighbors import edges
from lattice_symmetry.v2.physical import blocks, features, harmonics


ARMS = ("pair", "center0", "angular", "variance", "contacts", "reciprocal", "radial4")
PAIR_CUTOFF = 4.0
RECIPROCAL_CUTOFF = 0.35
SLOTS = 7
SHELLS = 3


def smooth_shells(t, shells=SHELLS):
    """C-infinity taper of squared normalized radius times Gaussian radial bases."""
    centers = torch.linspace(0, 1, shells, dtype=t.dtype, device=t.device)
    taper = torch.exp(1 - 1 / (1 - t).clamp_min(torch.finfo(t.dtype).eps))
    return taper[:, None] * (-0.5 * ((t[:, None] - centers) * max(1, shells - 1)).square()).exp()


def dense_pair_shells(cell, frac, z, cutoff=PAIR_CUTOFF, n_radial=4, graph=None):
    """Additional radial resolution with the released distinct-site normalization."""
    s,d,b,v = graph if graph is not None else edges(cell,frac,z,cutoff)
    active=(z>0).to(cell.dtype)
    counts=cell.new_zeros(len(z),119).scatter_add(1,z,active)
    rank=(counts>0).long().cumsum(1).sub(1).gather(1,z).clamp_min(0).flatten()
    keep=s!=d
    hi=torch.maximum(rank[s[keep]],rank[d[keep]])
    lo=torch.minimum(rank[s[keep]],rank[d[keep]])
    pairs=SLOTS*(SLOTS+1)//2
    index=b[keep]*pairs+hi*(hi+1)//2+lo
    weights=smooth_shells((v[keep]/cutoff).square().sum(-1),n_radial)
    val=cell.new_zeros(len(z)*pairs,n_radial).index_add(0,index,weights)
    return val.reshape(len(z),-1)/(2*active.sum(1,keepdim=True))


def orbit_layout(z, orbits, device):
    """Dense orbit slots and species/multiplicity pair classes, independent of input labels."""
    zs = z.detach().cpu().numpy()
    labels = orbits.detach().cpu().numpy() if torch.is_tensor(orbits) else orbits
    dense = np.zeros(zs.shape, dtype=np.int64)
    layouts = []
    for b, (species, index) in enumerate(zip(zs, labels)):
        mask = species > 0
        ids = np.asarray(index)
        active_ids = ids[mask] if len(ids) == len(mask) else ids
        if len(active_ids) != mask.sum():
            raise ValueError("Each occupied site needs an orbit identifier")
        _, first, d, mult = np.unique(active_ids, return_index=True, return_inverse=True,
                                      return_counts=True)
        active_z = species[mask]
        elements = active_z[first]
        if not np.all(elements[d] == active_z):
            raise ValueError("An occupied orbit cannot mix elements")
        dense[b, mask] = d
        ranks = np.unique(elements, return_inverse=True)[1]
        if len(np.unique(ranks)) > SLOTS:
            raise ValueError("Increase species slots for this chemistry")
        layouts.append((ranks, mult))
    md = max(len(m) for _, m in layouts)
    table = np.full((len(zs), md, md), -1, np.int64)
    classes, denominators = [], []
    offset = 0
    for b, (ranks, mult) in enumerate(layouts):
        i, j = np.triu_indices(len(mult), 1)
        hi, lo = np.maximum(ranks[i], ranks[j]), np.minimum(ranks[i], ranks[j])
        # Compatible orbit pairs share unordered elements and two multiplicity attributes.
        cls = ((hi * (hi + 1) // 2 + lo) * 2 + (mult[i] == mult[j])) * 2
        cls += np.minimum(mult[i], mult[j]) > 1
        classes.append(cls + b * (SLOTS * (SLOTS + 1) // 2) * 4)
        denominators.append(mult[i] * mult[j])
        ix = np.arange(offset, offset + len(i))
        table[b, i, j] = table[b, j, i] = ix
        offset += len(i)
    arrays = (dense, table, np.concatenate(classes), np.concatenate(denominators))
    return tuple(torch.as_tensor(x, device=device) for x in arrays)


def inter_orbit_contacts(cell, frac, z, orbits, cutoff=PAIR_CUTOFF, graph=None):
    """Per-unordered-pair shell means and raw second moments within typed classes."""
    dense, table, cls, denom = orbit_layout(z, orbits, z.device)
    s, d, b, v = graph if graph is not None else edges(cell, frac, z, cutoff)
    pair = table[b, dense.flatten()[s], dense.flatten()[d]]
    keep = pair >= 0
    radial = smooth_shells((v[keep] / cutoff).square().sum(-1))
    rows = cell.new_zeros((len(cls), SHELLS)).index_add(0, pair[keep], radial)
    rows = rows / (2 * denom[:, None])  # physical.edges returns both directed orientations
    size = len(z) * SLOTS * (SLOTS + 1) // 2 * 4
    count = torch.bincount(cls, minlength=size).clamp_min(1)
    moments = torch.cat([rows, rows.square()], dim=-1)
    return (cell.new_zeros((size, 2*SHELLS)).index_add(0, cls, moments)
            / count[:, None]).reshape(len(z), -1)


def orbit_variance(cell, frac, z, orbits, graph=None):
    """Released variance, also permitting non-contiguous padding and arbitrary orbit IDs."""
    s, d, b, v = graph if graph is not None else edges(cell, frac, z, PAIR_CUTOFF)
    q = v / PAIR_CUTOFF
    taper = (1 - q.square().sum(1)).clamp_min(0).pow(3)
    local = []
    for harmonic in harmonics(q):
        sums = cell.new_zeros(z.numel(), harmonic.shape[1]).index_add(0,s,taper[:,None]*harmonic)
        local.append(sums.square().sum(1))
    power = torch.stack(local, -1).reshape(*z.shape, 3)
    out = []
    for i, (species, index) in enumerate(zip(z, orbits)):
        mask = species > 0
        index = torch.as_tensor(index, dtype=torch.long, device=z.device)
        index = index[mask] if len(index) == len(mask) else index
        _, dense, mult = torch.unique(index, return_inverse=True, return_counts=True)
        atom = species[mask]
        elements = torch.zeros_like(mult).scatter(0, dense, atom)
        _, cls, count = torch.unique(torch.stack([elements,mult],1), dim=0,
                                     return_inverse=True, return_counts=True)
        orbit = cell.new_zeros(len(mult),3).index_add(0,dense,power[i,mask]) / mult[:,None]
        means = cell.new_zeros(len(count),3).index_add(0,cls,orbit) / count[:,None]
        out.append((orbit - means[cls]).square().mean(0))
    return torch.stack(out)


@lru_cache(maxsize=128)
def _integer_box(a, b, c):
    return np.stack(np.meshgrid(np.arange(-a,a+1), np.arange(-b,b+1), np.arange(-c,c+1),
                                indexing="ij"), axis=-1).reshape(-1,3)


def reciprocal_indices(cell, cutoff):
    cells = cell.detach().to(device="cpu", dtype=torch.float64).numpy()
    batches, indices = [], []
    for b, lattice in enumerate(cells):
        bounds = np.floor(np.linalg.norm(lattice, axis=1)*cutoff + 1e-10).astype(int)
        candidates = _integer_box(*bounds)
        physical = candidates @ np.linalg.inv(lattice).T
        keep = ((physical * physical).sum(1) < cutoff**2) & (candidates != 0).any(1)
        indices.append(candidates[keep])
        batches.append(np.full(keep.sum(), b, dtype=np.int64))
    return (torch.as_tensor(np.concatenate(batches), device=cell.device),
            torch.as_tensor(np.concatenate(indices), device=cell.device, dtype=cell.dtype))


def reciprocal_shells(cell, frac, z, cutoff=RECIPROCAL_CUTOFF):
    """Include all nonzero +/-h in the physical sphere. Recompute differentiable lengths."""
    b, h = reciprocal_indices(cell, cutoff)
    directions = torch.bmm(torch.linalg.inv(cell)[b], h[:,:,None]).squeeze(-1)
    weights = smooth_shells((directions / cutoff).square().sum(-1))
    active = (z > 0).to(cell.dtype)
    counts = cell.new_zeros(len(z),119).scatter_add(1,z,active)
    ranks = (counts > 0).long().cumsum(1).sub(1).gather(1,z).clamp_min(0)
    onehot = cell.new_zeros(*z.shape,SLOTS).scatter(2,ranks[:,:,None],active[:,:,None])
    fractions = onehot / onehot.sum(1,keepdim=True).clamp_min(1)
    lo, hi = torch.triu_indices(SLOTS,SLOTS,device=z.device)
    total = cell.new_zeros(len(z),len(lo),SHELLS)
    for first in range(0,len(b),1024):
        ix = slice(first,first+1024)
        bi = b[ix]
        phases = 2 * pi * (frac[bi] @ h[ix,:,None]).squeeze(-1)
        re = (phases.cos()[:,None,:] @ fractions[bi]).squeeze(1)
        im = (phases.sin()[:,None,:] @ fractions[bi]).squeeze(1)
        intensity = re[:,lo]*re[:,hi] + im[:,lo]*im[:,hi]
        total = total.index_add(0,bi,intensity[:,:,None]*weights[ix,None,:])
    density = torch.linalg.det(cell).abs() * (4*pi/3 * cutoff**3)
    return (total / density[:,None,None]).flatten(1)


def feature_blocks(cell, frac, z, orbits=None, arm="angular"):
    if arm not in ARMS:
        raise ValueError(f"Unknown descriptor arm {arm}")
    base = features(cell,frac,z,angular=arm!="pair")
    if arm == "center0":
        base = base[:, :blocks()["power0"].stop]
    out = {"base": base}
    if arm == "contacts":
        out[arm] = inter_orbit_contacts(cell,frac,z,orbits)
    elif arm == "variance":
        out[arm] = orbit_variance(cell,frac,z,orbits)
    elif arm == "reciprocal":
        out[arm] = reciprocal_shells(cell,frac,z)
    elif arm == "radial4":
        out[arm] = dense_pair_shells(cell,frac,z)
    return out


def combine(raw, stats):
    """Stats entries are (mean,scale); short base arms use a prefix of angular stats."""
    vectors = []
    for name, x in raw.items():
        mean, scale = (torch.as_tensor(v,device=x.device,dtype=x.dtype) for v in stats[name])
        vectors.append((x-mean[..., :x.shape[-1]]) / (scale[:x.shape[-1]] if scale.ndim else scale))
    return torch.cat(vectors, dim=-1)


def fit_stats(raw):
    """Fit on aligned, eligible training rows supplied by the caller; return NumPy arrays."""
    result = {}
    for name, array in raw.items():
        x = (array.detach().cpu().numpy() if torch.is_tensor(array) else np.asarray(array))
        if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError(f"Empty or nonfinite calibration data for {name}")
        x = x.astype(np.float64)
        mean = x.mean(0)
        scale = np.ones(x.shape[1], np.float64)
        slices = list(blocks().values()) if name == "base" else [slice(0,x.shape[1])]
        for sl in slices:
            if sl.start < x.shape[1]:
                scale[sl] = max(0.01, np.sqrt(np.square(x[:,sl] - mean[sl]).sum(1).mean()))
        result[name] = mean.astype(np.float32), scale.astype(np.float32)
    return result


@torch.no_grad()
def bank_direction(q, p, n, affinity="laplace", temperatures=(0.3, 0.8), return_info=False):
    if p.shape != n.shape or not temperatures or min(temperatures) <= 0:
        raise ValueError("Require equal reference-bank shapes and positive temperatures")
    distance = torch.cdist(q,torch.cat([p,n],1))
    result = torch.zeros_like(q)
    info = []
    for tau in temperatures:
        if affinity == "laplace":
            logits = -distance/tau
        elif affinity == "gaussian":
            logits = -0.5 * (distance/tau).square()
        else:
            raise ValueError(f"Unknown scalar affinity {affinity}")
        weight = (0.5 * (logits.log_softmax(1) + logits.log_softmax(2))).exp()
        wp, wn = weight.chunk(2, dim=-1)
        mp, mn = wp.sum(-1,keepdim=True), wn.sum(-1,keepdim=True)
        drift = (wp @ p)*mn - (wn @ n)*mp
        rms = (drift.norm(dim=-1).square().mean()+1e-8).sqrt()
        result += drift/rms
        if return_info:
            def effective(w):
                mass = w.sum(-1)
                return torch.where(mass>0,mass.square()/w.square().sum(-1).clamp_min(1e-30),0).mean().item()
            info.append(dict(tau=float(tau),raw_field_rms=float(rms),effective_positive=effective(wp),
                             effective_negative=effective(wn),positive_mass=float(mp.mean()),
                             negative_mass=float(mn.mean())))
    result = result / len(temperatures)
    return (result,info) if return_info else result


def drift_loss(q,p,n,affinity="laplace",temperatures=(0.3,0.8),return_info=False):
    result = bank_direction(q.detach(),p.detach(),n.detach(),affinity,temperatures,return_info)
    field = result[0] if return_info else result
    loss = 0.5 * (q-(q.detach()+field)).square().sum(-1).mean()
    return (loss,result[1]) if return_info else loss
