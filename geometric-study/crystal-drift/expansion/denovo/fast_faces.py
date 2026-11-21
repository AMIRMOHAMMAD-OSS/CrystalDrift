"""Evaluate a Voronoi face with the same triangulation and angle convention as pymatgen."""

import numpy as np
from numba import njit


@njit(cache=True)
def measure(center, vertices, indices):
    v = vertices[indices] - center
    a = v[0]
    ra = np.sqrt(np.dot(a, a))
    angle = volume = 0.0
    for j in range(1, len(v) - 1):
        b = v[j]
        c = v[j + 1]
        rb = np.sqrt(np.dot(b, b))
        rc = np.sqrt(np.dot(c, c))
        t = abs(
            a[0] * (b[1] * c[2] - b[2] * c[1])
            + a[1] * (b[2] * c[0] - b[0] * c[2])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
        den = ra * rb * rc + rc * np.dot(a, b) + rb * np.dot(a, c) + ra * np.dot(b, c)
        theta = (np.pi / 2 if t > 0 else -np.pi / 2) if den == 0 else np.arctan(t / den)
        angle += 2 * (theta if theta > 0 else theta + np.pi)
        volume += t / 6
    return angle, volume


def install_faces():
    from pymatgen.analysis.local_env import VoronoiNN

    original = VoronoiNN._extract_cell_info

    def extract(self, site_idx, sites, targets, voro, compute_adj_neighbors=False):
        result = {}
        center = sites[site_idx].coords
        for indices, verts in voro.ridge_dict.items():
            if site_idx not in indices:
                continue
            other = indices[1] if indices[0] == site_idx else indices[0]
            if -1 in verts:
                return original(self, site_idx, sites, targets, voro, compute_adj_neighbors)
            site = sites[other]
            if not site.is_ordered or site.specie not in targets:
                continue
            delta = site.coords - center
            dist = np.linalg.norm(delta)
            angle, volume = measure(center, voro.vertices, np.asarray(verts, np.int64))
            result[other] = dict(
                site=site,
                normal=delta / dist,
                solid_angle=angle,
                volume=volume,
                face_dist=dist / 2,
                area=6 * volume / dist,
                n_verts=len(verts),
            )
        if not result:
            return original(self, site_idx, sites, targets, voro, compute_adj_neighbors)
        # CrystalNN uses only solid angle and face area from this result.
        return result

    VoronoiNN._extract_cell_info = extract
    return original
