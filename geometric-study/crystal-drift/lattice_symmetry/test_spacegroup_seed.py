"""Test the unique-b P2_1 screw, including an oblique change of lattice basis."""

import unittest

import torch

from .generator import PeriodicGenerator
from .spacegroup_seed import spacegroup_seed


class SpaceGroupSeedTests(unittest.TestCase):
    def test_screw_in_seed_and_generator_output(self):
        identity = torch.eye(3, dtype=torch.float64)
        screw = torch.diag(torch.tensor([-1., 1., -1.], dtype=torch.float64))
        shift = torch.tensor([0., .5, 0.], dtype=torch.float64)
        shear = torch.tensor([[1., 1., 0.], [0., 1., 1.], [0., 0., 1.]], dtype=torch.float64)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(93)
            network = PeriodicGenerator(cutoff=3.5, width=12, n_radial=4).double()
        for basis in (identity, shear):
            with self.subTest(basis=basis.tolist()):
                rotation = basis @ screw @ torch.linalg.inv(basis)
                translation = shift @ torch.linalg.inv(basis)
                cell, frac, atoms = spacegroup_seed(
                    [6, 8, 8], torch.stack([identity, rotation]),
                    torch.stack([torch.zeros_like(shift), translation]), volume=90.0,
                    generator=torch.Generator().manual_seed(33),
                )
                torch.testing.assert_close(torch.linalg.det(cell), cell.new_tensor(90.))
                orbit_mates = torch.tensor([1, 0, 3, 2, 5, 4])
                self.assertEqual(atoms.tolist(), [6, 6, 8, 8, 8, 8])
                out_cell, out_frac = network(cell, frac, atoms)
                self.assertGreater((out_cell - cell).abs().max().item(), 1.e-7)
                for lattice, sites in ((cell, frac), (out_cell, out_frac)):
                    metric = lattice @ lattice.T
                    torch.testing.assert_close(rotation @ metric @ rotation.T, metric,
                                               atol=3e-11, rtol=1e-12)
                    residual = sites @ rotation + translation - sites[orbit_mates]
                    self.assertLess((residual - residual.round()).abs().max().item(), 3e-12)

    def test_reject_incomplete_translation_group(self):
        rotations = torch.eye(3).repeat(2, 1, 1)
        with self.assertRaisesRegex(ValueError, "closed"):
            spacegroup_seed([8], rotations, [[0., 0., 0.], [0., 1./3., 0.]], volume=30.)

    def test_centering_translations(self):
        rotations = torch.eye(3).repeat(2, 1, 1)
        _, sites, atoms = spacegroup_seed([14], rotations, [[0., 0., 0.], [.5, .5, .5]],
                                         volume=30., generator=torch.Generator().manual_seed(45))
        diff = sites[1] - sites[0] - .5
        torch.testing.assert_close(diff, diff.round())
        self.assertEqual(atoms.tolist(), [14, 14])


if __name__ == "__main__":
    unittest.main()
