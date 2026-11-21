"""Random crystal seeds conditioned on a supplied space group in a fixed setting."""

import math

import numpy as np
import torch


def _periodic_match(delta, tolerance):
    return np.all(np.abs(delta - np.rint(delta)) <= tolerance, axis=-1)


def _group_operations(rotations, translations, tolerance):
    r = torch.as_tensor(rotations, dtype=torch.float64).detach().cpu().numpy()
    t = torch.as_tensor(translations, dtype=torch.float64).detach().cpu().numpy()
    if r.ndim != 3 or r.shape[1:] != (3, 3) or not len(r) or t.shape != (len(r), 3):
        raise ValueError("Supply one 3x3 fractional rotation and one translation per operation.")
    if not (np.isfinite(r).all() and np.isfinite(t).all()):
        raise ValueError("Space-group operation entries must be finite.")
    if np.any(np.abs(r - np.rint(r)) > tolerance):
        raise ValueError("Fractional rotations must have integer entries.")
    r = np.rint(r).astype(np.int64)
    if np.any(np.abs(np.rint(np.linalg.det(r))) != 1):
        raise ValueError("Fractional rotations must be unimodular.")
    t = np.remainder(t, 1)

    # Centered settings can have several translations for the same rotation.
    representatives, labels = np.unique(r, axis=0, return_inverse=True)
    by_rotation = {tuple(s.flat): t[labels == i] for i, s in enumerate(representatives)}
    identity = by_rotation.get(tuple(np.eye(3, dtype=np.int64).flat))
    if identity is None or not _periodic_match(identity, tolerance).any():
        raise ValueError("The operation list must contain the identity.")
    for shifts in by_rotation.values():
        equal = _periodic_match(shifts[:, None] - shifts[None, :], tolerance)
        if equal.sum() != len(shifts):
            raise ValueError("Duplicate space-group operations modulo lattice translations.")
    for left in representatives:
        shifts_left = by_rotation[tuple(left.flat)]
        for right in representatives:
            target = by_rotation.get(tuple((left @ right).flat))
            if target is None:
                raise ValueError("Operation list is not closed under composition.")
            composed = (shifts_left @ right)[:, None] + by_rotation[tuple(right.flat)][None, :]
            if not _periodic_match(composed[:, :, None] - target, tolerance).any(axis=-1).all():
                raise ValueError("Operation list is not closed modulo lattice translations.")
    return r, t


def spacegroup_seed(site_species, rotations, translations, *, volume, generator=None,
                    dtype=torch.float64, device=None, tolerance=1e-6):
    """Return (cell, fractional sites, atomic numbers) for general-position orbits.

    Fractional positions are rows. Supply every operation modulo the integer
    translations of the chosen cell, acting as f -> f @ R + t. Transpose matrices
    provided by crystallographic tools that use the column-vector convention.
    The setting, origin, cell volume and one atomic number per independent orbit
    are supplied by the caller. Two orbits can have the same species. A uniform
    torus representative induces the uniform measure on the general-position
    asymmetric unit modulo this finite group. Sites carry no special Wyckoff
    restrictions. The generated symmetry may include additional operations.
    """
    if not math.isfinite(volume) or volume <= 0 or not 0 < tolerance < 0.01:
        raise ValueError("Expected positive finite volume and a tolerance between 0 and 0.01.")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Use float32 or float64 crystal coordinates.")
    device = generator.device if device is None and generator is not None else device
    atoms = torch.as_tensor(site_species, device=device)
    if (atoms.ndim != 1 or not len(atoms) or atoms.is_floating_point() or atoms.is_complex()
            or atoms.dtype == torch.bool or ((atoms < 1) | (atoms > 118)).any()):
        raise ValueError("Specify one integer atomic number (1 to 118) per general orbit.")
    r, t = _group_operations(rotations, translations, tolerance)
    r = torch.as_tensor(r, device=atoms.device, dtype=dtype)
    t = torch.as_tensor(t, device=atoms.device, dtype=dtype)

    # R G R^T = G holds after averaging because left-multiplication permutes the group.
    settings = dict(dtype=dtype, device=atoms.device)
    noise = torch.randn((3, 3), generator=generator, **settings)
    base = torch.eye(3, **settings) + noise @ noise.T / 3
    metric = (r @ base @ r.transpose(-1, -2)).mean(dim=0)
    cell = torch.linalg.cholesky((metric + metric.T) / 2)
    cell = cell * (volume / cell.diag().prod()).pow(1 / 3)

    orbits = []
    for _ in range(len(atoms)):
        for _attempt in range(64):
            point = torch.rand(3, generator=generator, **settings)
            orbit = (point @ r + t).remainder(1)
            previous = torch.cat([orbit, *orbits])
            delta = orbit[:, None] - previous[None, :]
            duplicate = ((delta - delta.round()).abs() <= tolerance).all(dim=-1)
            if duplicate.sum() == len(orbit):  # only each site's match with itself
                orbits.append(orbit)
                break
        else:
            raise RuntimeError("Could not draw distinct general-position orbits at this tolerance.")
    return cell, torch.cat(orbits), atoms.long().repeat_interleave(len(r))
