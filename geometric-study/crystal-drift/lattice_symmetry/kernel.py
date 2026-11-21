"""Finite-range crystal features and drifting affinities in a row lattice basis."""

import math

import torch


def periodic_edges(cell, frac, cutoff):
    """Return source indices, target indices and Cartesian vectors to all images within cutoff.

    A nonzero periodic image of the source site is a neighbor. The zero image of
    that same site is excluded. Different sites can occupy coincident positions.
    Cell rows are lattice vectors and fractional coordinates may be unwrapped.
    """
    if cell.shape != (3, 3) or frac.ndim != 2 or frac.shape[1] != 3 or not len(frac):
        raise ValueError("Expected a 3x3 cell and at least one fractional site of dimension 3.")
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("The cutoff must be finite and positive.")

    delta = frac[None, :, :] - frac[:, None, :]
    delta = delta - delta.detach().round()
    # If v=(delta+n)@cell and |v|<cutoff, then |delta_k+n_k| is bounded by
    # cutoff times the norm of column k of cell^{-1}. This includes oblique cells.
    inverse = torch.linalg.inv(cell.detach().double())
    bounds = (cutoff * inverse.norm(dim=0) + 0.5).ceil().to(torch.long)
    sizes = (2 * bounds + 1).tolist()
    n_images = math.prod(sizes)
    block = max(1, 131072 // len(frac) ** 2)
    same_site = torch.eye(len(frac), dtype=torch.bool, device=frac.device)
    sources, targets, vectors = [], [], []
    for start in range(0, n_images, block):
        indices = torch.arange(start, min(start + block, n_images), device=frac.device)
        shifts = torch.stack((indices // (sizes[1] * sizes[2]),
                              (indices // sizes[2]) % sizes[1], indices % sizes[2]), dim=-1) - bounds
        cart = (delta[None] + shifts[:, None, None, :]) @ cell
        within = cart.detach().square().sum(dim=-1) < cutoff ** 2
        zero_image = (shifts == 0).all(dim=-1)
        image, source, target = (within & ~(zero_image[:, None, None] & same_site)).nonzero(as_tuple=True)
        sources.append(source)
        targets.append(target)
        vectors.append(cart[image, source, target])
    return torch.cat(sources), torch.cat(targets), torch.cat(vectors)


def periodic_features(cell, frac, z, species, cutoff=5.0, n_radial=8):
    """Embed log cell volume, stoichiometry, lattice shells, and species-resolved pair shells.

    Use the same ordered species vocabulary, cutoff (in cell length units) and radial
    basis for query and reference samples. Inputs represent a single unpadded cell.
    Finite radial shells may coincide for distinct structures.
    """
    vocabulary = tuple(int(s) for s in species)
    if not vocabulary or len(set(vocabulary)) != len(vocabulary) or n_radial < 1:
        raise ValueError("Expected a nonempty distinct species vocabulary and a positive radial basis size.")
    if z.ndim != 1 or len(z) != len(frac):
        raise ValueError("Each fractional site needs one chemical species.")
    matches = z[:, None] == z.new_tensor(vocabulary)
    if not matches.any(dim=1).all():
        raise ValueError("All site species must occur in the shared vocabulary.")
    labels = matches.to(torch.long).argmax(dim=1)
    source, target, vectors = periodic_edges(cell, frac, cutoff)

    radial_coordinate = (vectors / cutoff).square().sum(dim=-1, keepdim=True)
    centers = torch.linspace(0, 1, n_radial, dtype=cell.dtype, device=cell.device)
    basis = torch.exp(-0.5 * ((radial_coordinate - centers) * max(1, n_radial - 1)).square())
    # A compact C-infinity window in squared distance. Its value and all its
    # derivatives match the zero extension at the cutoff, even as image lists change.
    remaining = (1 - radial_coordinate).clamp_min(torch.finfo(cell.dtype).eps)
    basis = basis * torch.exp(1 - 1 / remaining)

    lattice_shell = basis[(source == 0) & (target == 0)].sum(dim=0)
    distinct = source != target
    s, t = labels[source[distinct]], labels[target[distinct]]
    high, low = torch.maximum(s, t), torch.minimum(s, t)
    pair_index = high * (high + 1) // 2 + low
    pair_count = len(vocabulary) * (len(vocabulary) + 1) // 2
    pair_shells = cell.new_zeros(pair_count, n_radial).index_add(0, pair_index, basis[distinct])
    pair_shells = pair_shells / (2 * len(frac))  # directed edges count both orientations
    counts = matches.sum(dim=0).to(cell.dtype)
    volume = torch.linalg.slogdet(cell).logabsdet.reshape(1)
    return torch.cat((volume, counts, lattice_shell, pair_shells.flatten()))


def crystal_kernel(x, y, tau=0.8):
    """Unnormalized Laplace affinity between two periodic feature vectors."""
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("The kernel temperature must be finite and positive.")
    return torch.exp(-torch.linalg.vector_norm(x - y, dim=-1) / tau)


def feature_drift(query, positive, negative):
    """Existing normalized two-temperature CrystalDrift field for one unpadded reference bank.

    Rows are periodic feature vectors. Positive and negative banks have the same
    size, as in the original experimental implementation. Targets are detached.
    """
    from scaled.family import within_direction

    if (query.ndim != 2 or positive.ndim != 2 or positive.shape != negative.shape
            or query.shape[1] != positive.shape[1] or not len(query) or not len(positive)):
        raise ValueError("Expected nonempty feature matrices and equally sized reference banks.")
    return within_direction(query[None], positive[None], negative[None])[0]
