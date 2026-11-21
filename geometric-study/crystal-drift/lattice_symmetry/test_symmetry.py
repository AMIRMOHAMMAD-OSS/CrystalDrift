"""CPU regressions. From crystal-drift, run python -m lattice_symmetry.test_symmetry."""
import itertools
import unittest

import torch

from .generator import PeriodicGenerator
from .kernel import crystal_kernel, feature_drift, periodic_edges, periodic_features


DTYPE = torch.float64
SPECIES = (6, 8, 14)
CUTOFF = 3.2
ROTATION = torch.tensor([[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]], dtype=DTYPE)
REFLECTION = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=DTYPE))
SHEAR = torch.tensor([[1, 3, -1], [0, 1, 1], [0, 0, 1]], dtype=DTYPE)
IDENTITY = torch.eye(3, dtype=DTYPE)
ORDER = torch.tensor([3, 0, 2, 1])
SHIFTS = torch.tensor([[3, -4, 2], [0, 1, -2], [-3, 2, 1], [1, -1, -2]], dtype=DTYPE)


def seed(index=0):
    cell = torch.tensor([[3.3, 0.2, 0.1], [0.1, 3.7, -0.2], [-0.1, 0.3, 4.1]], dtype=DTYPE)
    frac = torch.tensor([[0.98, 0.12, 0.45], [0.02, 0.37, 0.23],
                         [0.31, 0.74, 0.84], [0.67, 0.15, 0.11]], dtype=DTYPE)
    frac[1, 0] += 0.061 * index
    cell[1, 2] -= 0.037 * index
    return cell, frac, torch.tensor([6, 8, 6, 14])


def reexpress(crystal, basis, frame, origin, order):
    cell, frac, species = crystal
    new_frac = frac[order] @ torch.linalg.inv(basis) + origin + SHIFTS
    return basis @ cell @ frame, new_frac, species[order]


def independently_reexpress(crystal, index):
    basis = (SHEAR, REFLECTION, SHEAR.T)[index % 3]
    frame = (ROTATION, REFLECTION, ROTATION @ REFLECTION)[index % 3]
    origin = torch.tensor([0.32, -0.74, 0.19], dtype=DTYPE) * (index + 1)
    return reexpress(crystal, basis, frame, origin, ORDER.roll(index))


def fingerprint(crystal):
    return periodic_features(*crystal, SPECIES, cutoff=CUTOFF, n_radial=5)


def bank(crystals):
    return torch.stack([fingerprint(crystal) for crystal in crystals])


def torus_error(actual, expected, cell):
    difference = actual - expected
    return ((difference - difference.round()) @ cell).norm(dim=1).max().item()


class SymmetryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.model = PeriodicGenerator(cutoff=CUTOFF, width=16, n_radial=5).double()

    def test_images_in_skew_basis_against_independent_enumeration(self):
        cell, frac, _ = seed()
        offsets = torch.tensor(list(itertools.product(range(-3, 4), repeat=3)), dtype=DTYPE)
        # Offsets outside this cube cannot reach the cutoff in this reference basis.
        self.assertGreater(3 * torch.linalg.svdvals(cell).min().item(), CUTOFF)
        basis = torch.tensor([[1, 5, 0], [0, 1, 0], [0, 0, 1]], dtype=DTYPE)
        src, dst, vectors = periodic_edges(basis @ cell, frac @ torch.linalg.inv(basis) + SHIFTS, CUTOFF)
        for i, j in itertools.product(range(len(frac)), repeat=2):
            expected = (frac[j] - frac[i] + offsets) @ cell
            radii = expected.norm(dim=1)
            expected = expected[(radii < CUTOFF) & (radii > 0)]
            actual = vectors[(src == i) & (dst == j)]
            self.assertEqual(len(actual), len(expected))
            if len(actual):
                nearest = torch.cdist(actual, expected)
                self.assertLess(nearest.amin(dim=1).amax().item(), 1e-10)
                self.assertLess(nearest.amin(dim=0).amax().item(), 1e-10)

    def test_feature_and_generator_transformations(self):
        base = seed()
        reference = seed(2)
        phi = fingerprint(base)
        reference_kernel = crystal_kernel(phi, fingerprint(reference))
        self.assertGreater((phi - fingerprint(reference)).norm().item(), 1e-6)
        self.assertGreater((crystal_kernel(phi, phi) - reference_kernel).item(), 1e-8)
        with torch.no_grad():
            out_cell, out_frac = self.model(*base)
            for basis, frame in ((IDENTITY, ROTATION), (IDENTITY, REFLECTION), (SHEAR, IDENTITY),
                                 (REFLECTION, IDENTITY), (SHEAR.T, ROTATION @ REFLECTION)):
                origin = torch.tensor([0.21, -0.71, 0.39], dtype=DTYPE)
                transformed = reexpress(base, basis, frame, origin, ORDER)
                phi_transformed = fingerprint(transformed)
                torch.testing.assert_close(phi_transformed, phi, rtol=1e-10, atol=1e-11)
                kernel = crystal_kernel(phi_transformed, fingerprint(independently_reexpress(reference, 2)))
                torch.testing.assert_close(kernel, reference_kernel, rtol=1e-10, atol=1e-11)
                actual_cell, actual_frac = self.model(*transformed)
                torch.testing.assert_close(actual_cell, basis @ out_cell @ frame, rtol=1e-10, atol=1e-10)
                expected_frac = out_frac[ORDER] @ torch.linalg.inv(basis) + origin
                self.assertLess(torus_error(actual_frac, expected_frac, actual_cell), 1e-9)

    def test_lattice_shell_detects_shape_at_fixed_volume(self):
        cubic = 2.5 * IDENTITY
        distorted = torch.diag(torch.tensor([2.0, 2.5, 3.125], dtype=DTYPE))
        frac = torch.tensor([[0.16, 0.42, 0.78]], dtype=DTYPE)
        z = torch.tensor([8])
        torch.testing.assert_close(torch.linalg.det(cubic), torch.linalg.det(distorted))
        first = fingerprint((cubic, frac, z))
        second = fingerprint((distorted, frac, z))
        self.assertGreater((first - second).norm().item(), 1e-4)
        equivalent = fingerprint((SHEAR @ distorted @ ROTATION, frac @ torch.linalg.inv(SHEAR), z))
        torch.testing.assert_close(second, equivalent, rtol=1e-10, atol=1e-11)

    def test_banked_drift_and_one_training_update(self):
        crystals = [seed(index) for index in range(6)]
        other = [independently_reexpress(crystal, i) for i, crystal in enumerate(crystals)]

        def objective(records):
            generated = []
            for crystal in records[:2]:
                cell, frac = self.model(*crystal)
                generated.append((cell, frac, crystal[2]))
            query = bank(generated)
            drift = feature_drift(query.detach(), bank(records[2:4]), bank(records[4:]))
            loss = 0.5 * (query - (query.detach() + drift.detach())).square().sum(-1).mean()
            return drift, loss

        drift, loss = objective(crystals)
        other_drift, other_loss = objective(other)
        self.assertGreater(drift.norm().item(), 1e-5)
        torch.testing.assert_close(other_drift, drift, rtol=2e-8, atol=2e-9)
        params = tuple(self.model.parameters())
        gradient = torch.autograd.grad(loss, params)
        other_gradient = torch.autograd.grad(other_loss, params)
        grad = torch.cat([part.flatten() for part in gradient])
        other_grad = torch.cat([part.flatten() for part in other_gradient])
        self.assertTrue(torch.isfinite(grad).all().item())
        self.assertGreater(grad.norm().item(), 1e-6)
        torch.testing.assert_close(other_grad, grad, rtol=2e-8, atol=2e-9)
        before = torch.cat([p.detach().flatten() for p in params])
        opt = torch.optim.SGD(params, lr=1e-3)
        for param, deriv in zip(params, gradient):
            param.grad = deriv
        opt.step()
        after = torch.cat([p.detach().flatten() for p in params])
        self.assertGreater((after - before).norm().item(), 0.0)

    def test_twofold_screw_of_seed_is_preserved(self):
        cell = torch.diag(torch.tensor([3.4, 3.7, 4.0], dtype=DTYPE))
        rotation = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=DTYPE))
        translation = torch.tensor([0.0, 0.0, 0.5], dtype=DTYPE)
        reps = torch.tensor([[0.23, 0.14, 0.43], [0.31, 0.65, 0.13]], dtype=DTYPE)
        frac = torch.cat([reps, (reps @ rotation + translation).remainder(1)])
        with torch.no_grad():
            out_cell, out_frac = self.model(cell, frac, torch.tensor([6, 8, 6, 8]))
        gram = out_cell @ out_cell.T
        torch.testing.assert_close(rotation @ gram @ rotation.T, gram, rtol=1e-10, atol=1e-10)
        self.assertLess(torus_error(out_frac[2:], out_frac[:2] @ rotation + translation, out_cell), 1e-9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
