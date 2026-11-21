"""Experimental one-call crystal generator for composition-labelled periodic seeds."""

import math

import torch
from torch import nn

from .kernel import periodic_edges


class PeriodicGenerator(nn.Module):
    """Update a row-basis cell and its fractional sites from periodic Cartesian edges.

    The input is one nonsingular cell, an unpadded set of sites and their atomic
    numbers (1 to 118). Sampling calls this module once on a chosen seed. A
    transformed run uses the same physical seed expressed in the new frame.
    """

    def __init__(self, cutoff=4.0, width=32, n_radial=8):
        super().__init__()
        if not math.isfinite(cutoff) or cutoff <= 0 or width < 1 or n_radial < 1:
            raise ValueError("cutoff, width and n_radial must be positive")
        self.cutoff = float(cutoff)
        self.register_buffer("centers", torch.linspace(0, 1, n_radial))
        self.radial_precision = float(n_radial**2)
        self.element = nn.Embedding(119, width)
        self.message = nn.Sequential(nn.Linear(2 * width + n_radial, width), nn.SiLU())
        self.site = nn.Sequential(nn.Linear(2 * width + 1, width), nn.SiLU())
        self.edge = nn.Sequential(
            nn.Linear(2 * width + n_radial, width), nn.SiLU(), nn.Linear(width, 2)
        )
        self.dilation = nn.Linear(width, 1)

    def forward(self, cell, frac, z):
        """Return (cell_out, frac_out), with fractional coordinates wrapped to [0, 1)."""
        if z.ndim != 1 or z.shape[0] == 0 or frac.shape != (z.shape[0], 3):
            raise ValueError("a nonempty atomic number vector must label every site")
        if z.dtype != torch.long or torch.any((z < 1) | (z > 118)):
            raise ValueError("atomic numbers must be long integers between 1 and 118")
        source, target, vectors = periodic_edges(cell, frac, self.cutoff)
        scaled_vectors = vectors / self.cutoff
        squared_radius = scaled_vectors.square().sum(dim=-1, keepdim=True)
        radial = torch.exp(-self.radial_precision * (squared_radius - self.centers).square())
        envelope = (1 - squared_radius).clamp_min(0).pow(3)
        neighbor_mass = envelope.new_zeros((len(z), 1)).index_add(0, source, envelope)
        weights = envelope / (1 + neighbor_mass[source])

        elements = self.element(z)
        messages = self.message(torch.cat((elements[source], elements[target], radial), dim=-1))
        aggregates = torch.zeros_like(elements).index_add(0, source, weights * messages)
        log_volume = torch.linalg.slogdet(cell).logabsdet
        log_specific_volume = log_volume - math.log(len(z) * self.cutoff**3)
        site_inputs = torch.cat((elements, aggregates, log_specific_volume.expand(len(z), 1)), dim=-1)
        hidden = self.site(site_inputs)
        edge_outputs = self.edge(torch.cat((hidden[source], hidden[target], radial), dim=-1))
        shift_weights, strain_weights = (weights * edge_outputs.tanh()).unbind(dim=-1)

        # Cartesian vectors transform with the frame. Remove the uniform shift,
        # then convert to fractions in the seed cell before the affine cell update.
        weighted_vectors = shift_weights[:, None] * vectors
        displacement = frac.new_zeros(frac.shape).index_add(0, source, weighted_vectors)
        displacement = displacement - displacement.mean(dim=0)
        fractional_shift = torch.linalg.solve(cell.T, displacement.T).T
        frac_out = (frac + fractional_shift).remainder(1)

        # The two terms are equivariant symmetric tensors. Their exponential is
        # positive definite and preserves the cell's nonsingularity and handedness.
        identity = torch.eye(3, device=cell.device, dtype=cell.dtype)
        isotropic = self.dilation(hidden.mean(dim=0)).tanh() * identity
        anisotropic = scaled_vectors.T @ (strain_weights[:, None] * scaled_vectors) / len(z)
        strain = 0.25 * (isotropic + anisotropic)
        cell_out = cell @ torch.linalg.matrix_exp(strain)
        return cell_out, frac_out
