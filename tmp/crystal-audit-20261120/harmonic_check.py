"""Exact collinear bispectrum for the two eight-site motifs in supplement S3.

Run with Python, NumPy and SymPy. Outputs harmonic_check.json beside this file.
The reciprocal-vector convention omits 2π from its norm and includes 2π in
the Fourier phase. Both coefficients are normalized by the species count.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import sympy as sp


MOTIFS = ((0, 1, 4, 6), (0, 1, 3, 7))
Z = np.array([8] * 4 + [14] * 4)
H = np.array([1, 0, 0])


def exact_alpha(subset, k):
    phases = [sp.pi * ((k * s) % 12) / 6 for s in subset]
    return sp.simplify(sum(sp.cos(a) + sp.I * sp.sin(a) for a in phases) / len(subset))


def exact_parts(value):
    return {'real': str(sp.simplify(sp.re(value))), 'imag': str(sp.simplify(sp.im(value)))}


def sites(subset):
    f = np.zeros((8, 3))
    f[:, 0] = np.tile(np.array(subset) / 12, 2)
    f[4:, 1] = 1 / 3
    return f


def bispectrum(f, z, h):
    a = np.exp(2j * np.pi * (f[z == 8] @ h)).mean()
    b2 = np.exp(2j * np.pi * (f[z == 14] @ (2 * h))).mean()
    return a**2 * b2.conjugate()


def reciprocal_shell(cell, squared_norm):
    # h = L q, hence |h_i| <= ||L_i|| ||q||. This bounds every index in the shell.
    bounds = np.ceil(np.linalg.norm(cell, axis=1) * np.sqrt(squared_norm) + 1e-12).astype(int)
    grid = np.array(list(itertools.product(*(range(-x, x + 1) for x in bounds))))
    physical = np.linalg.solve(cell, grid.T).T
    return grid[np.isclose((physical**2).sum(1), squared_norm, atol=1e-12, rtol=0)]


def main():
    gram = 100 * np.diag([1., np.sqrt(2.), np.sqrt(3.)]); cell = np.linalg.cholesky(gram)
    squared_norm = .01
    shell = reciprocal_shell(cell, squared_norm)
    assert set(map(tuple, shell)) == {(-1, 0, 0), (1, 0, 0)}
    rng = np.random.default_rng(7157)
    u = np.array([[1, 0, 0], [2, 1, 0], [0, 1, 1]])
    changed_shell = reciprocal_shell(u @ cell, squared_norm)
    assert set(map(tuple, changed_shell)) == {tuple(u @ H), tuple(-u @ H)}
    records = []; exact = []; residuals = {}

    def check(label, x, y):
        err = float(abs(x - y))
        residuals[label] = max(residuals.get(label, 0.), err)
        assert err < 1e-12, (label, x, y, err)

    for idx, subset in enumerate(MOTIFS):
        a, a2 = [exact_alpha(subset, k) for k in (1, 2)]
        b = sp.simplify(sp.expand_complex(a**2 * sp.conjugate(a2)))
        expected = ((2 * idx - 1) * sp.sqrt(3) + sp.I) / 32
        assert sp.simplify(b - expected) == 0
        exact.append(b)
        f = sites(subset); observed = bispectrum(f, Z, H)
        check('complex_value_vs_symbolic', observed, complex(b))
        # All ordered A,A,B triples, with repeated A indices included.
        t = (f[Z == 8][:, None, None] + f[Z == 8][None, :, None] - 2 * f[Z == 14][None, None, :]) @ H
        check('direct_cosine_triplet_mean', np.cos(2 * np.pi * t).mean(), observed.real)
        for _ in range(12):
            shift = rng.random(3); perm = rng.permutation(len(Z))
            check('origin_and_simultaneous_site_label_permutation', bispectrum((f[perm] + shift) % 1, Z[perm], H), observed)
        check('origin_fixed_inversion_conjugation', bispectrum((-f) % 1, Z, H), observed.conjugate())
        for signs in itertools.product((-1, 1), repeat=3):
            check('rectangular_lattice_axis_reflections_real', bispectrum((f * signs) % 1, Z, H).real, observed.real)
        check('basis_change_covariant_index', bispectrum((f @ np.linalg.inv(u)) % 1, Z, u @ H), observed)
        changed_f = (f @ np.linalg.inv(u)) % 1
        check('basis_independent_reciprocal_shell', np.mean([bispectrum(changed_f, Z, h) for h in changed_shell]), observed.real)
        for determinant in (-1, 1):
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            rotation[:, 0] *= determinant * np.sign(np.linalg.det(rotation))
            new_cell = cell @ rotation
            cart = f @ cell @ rotation + rng.normal(size=3)
            new_f = (cart @ np.linalg.inv(new_cell)) % 1
            rotated_shell = reciprocal_shell(new_cell, squared_norm)
            check('cartesian_orthogonal_transform_and_shift', np.mean([bispectrum(new_f, Z, h) for h in rotated_shell]), observed.real)
        records.append({'x_numerators_mod_12': list(subset), 'alpha_1': exact_parts(a), 'alpha_2': exact_parts(a2),
                        'B_AAB_e1_e1': exact_parts(b), 'absolute_value': str(sp.simplify(sp.Abs(b))),
                        'shell_average_numeric': float(observed.real)})
    gap = sp.simplify(sp.re(exact[1] - exact[0]))
    assert gap == sp.sqrt(3) / 16 and sp.im(exact[1] - exact[0]) == 0
    modulus_agrees = all(sp.simplify(sp.Abs(exact_alpha(MOTIFS[0], k))**2 - sp.Abs(exact_alpha(MOTIFS[1], k))**2) == 0 for k in range(12))
    assert modulus_agrees
    report = {'frequency': 'B = a_A(e1)^2 conjugate(a_B(2e1)), with exp(+2 pi i f h)',
              'source': 'Two eight-site O4Si4 motifs, binary mathematical supplement S3 Proposition 3.',
              'underlying_cyclic_sets': 'Mandereau et al., Z-relation and homometry in musical distributions, Fig. 1. https://doi.org/10.1080/17459737.2011.608819',
              'gram_diagonal_A2': ['100', '100*sqrt(2)', '100*sqrt(3)'], 'y_positions_A_B': ['0', '1/3'],
              'motifs': records, 'real_difference_S2_minus_S1': str(gap), 'absolute_difference_imaginary': '0',
              'absolute_difference_magnitudes': '0', 'all_12_single_species_power_coefficients_identical_symbolically': modulus_agrees,
              'reciprocal_shell_squared_norm_A^-2': squared_norm, 'shell_h_before_reindex': shell.tolist(),
              'unimodular_reindex_U': u.tolist(), 'shell_h_after_reindex': changed_shell.tolist(),
              'observation': 'The shell average equals the real part because its only indices are a conjugate pair. '
                             'For any frequency pair, the modulus of a triple product is fixed by its three constituent moduli.',
              'max_absolute_numeric_residuals': residuals, 'numpy': np.__version__, 'sympy': sp.__version__}
    out = Path(__file__).with_suffix('.json')
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(f'{out.name}: separation {gap}, largest numeric residual {max(residuals.values()):.3g}')


if __name__ == '__main__':
    main()
