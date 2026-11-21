"""One-evaluation generator and crystal-level drifting objective for binary unit cells."""
import math

import torch
from torch import nn


# One member of each +/- pair, including the unit vectors and the sums used below.
WAVES = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),
         (1,-1,0),(1,0,-1),(0,1,-1),(2,0,0),(0,2,0),(0,0,2),(1,1,1)]


def species_masks(z):
    occupied = z > 0
    low = z.masked_fill(~occupied, 119).amin(1, keepdim=True)
    first = (z == low) & occupied
    return occupied, first, occupied & ~first


def masked_mean(values, mask):
    weights = mask.to(values.dtype)
    weights = weights / weights.sum(1, keepdim=True).clamp_min(1)
    return (values * weights.unsqueeze(-1)).sum(1)


def relative_phases(cosine, sine, mask):
    real = masked_mean(cosine, mask).unsqueeze(1)
    imag = masked_mean(sine, mask).unsqueeze(1)
    return torch.cat((cosine * real + sine * imag, sine * real - cosine * imag), dim=-1)


class CrystalGenerator(nn.Module):
    """A permutation-equivariant set map from atom and cell noise to a periodic cell."""

    def __init__(self, width=60):
        super().__init__()
        self.elements = nn.Embedding(119, 16, padding_idx=0)
        self.site = nn.Sequential(nn.Linear(36, width), nn.SiLU(), nn.Linear(width,width), nn.SiLU())
        self.context = nn.Sequential(nn.Linear(width + 8, width), nn.SiLU())
        self.shift = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, 3))
        self.cell = nn.Linear(width, 6)

    def forward(self, z, cell_noise, location_noise):
        occupied, first, second = species_masks(z)
        n = occupied.sum(1, keepdim=True)
        amounts = torch.cat((n / 8, first.sum(1, keepdim=True) / n), dim=1)
        angle = location_noise * (2 * math.pi)
        c, s = angle.cos(), angle.sin()
        geometry = torch.cat((relative_phases(c, s, first), relative_phases(c, s, second)), -1)
        context_noise = torch.cat((cell_noise, amounts), dim=1)
        features = torch.cat((self.elements(z), geometry,
                              context_noise.unsqueeze(1).expand(-1, z.shape[1], -1)),dim=-1)
        site = self.site(features)
        context = self.context(torch.cat((masked_mean(site, occupied), context_noise), dim=1))
        shifts = self.shift(torch.cat((site, context.unsqueeze(1).expand_as(site)), dim=-1))
        positions = torch.remainder(location_noise + 0.5 * shifts.tanh(), 1.0)
        cell = self.cell(context) + 0.15 * cell_noise
        return cell, positions * occupied.unsqueeze(-1)


def chemistry(z, dtype):
    """A fixed neighborhood for partial pooling over element pairs and cell stoichiometry."""
    occupied, first, _ = species_masks(z)
    species = torch.stack((z.masked_fill(~occupied, 119).amin(1), z.amax(1)), dim=1)
    count = occupied.sum(1, keepdim=True)
    # Scales define only affinity conditioning. Element numbers also enter the learned generator.
    return torch.cat((species.to(dtype) / 20, first.sum(1,keepdim=True)/count,
                      count.to(dtype) / 4), dim=1)


def crystal_features(k, f, z, representation):
    occupied, first, second = species_masks(z)
    cell = k / math.sqrt(6)
    if representation == "raw":
        if z.shape[1] != 2:
            raise ValueError("The chart ablation has one atom of each species.")
        delta = torch.remainder(f[:,1]-f[:,0], 1)
        return torch.cat((cell, (delta - 0.5) * 2), dim=-1)
    waves = f.new_tensor(WAVES)
    phase = (f @ waves.T) * (2 * math.pi)
    cos, sin = phase.cos(), phase.sin()
    a = torch.complex(masked_mean(cos, first), masked_mean(sin, first))
    b = torch.complex(masked_mean(cos, second), masked_mean(sin, second))
    ab = a.conj() * b
    size = math.sqrt(len(WAVES))
    parts = [cell, ab.real / size, ab.imag / size]
    if representation == "multi":
        # Bispectra at h1+h2 cancel the common phase of the third coefficient.
        i, j, summed = [0,0,1], [1,2,2], [3,4,5]
        tri = torch.cat((a[:,i] * a[:,j] * b[:,summed].conj(),
                         b[:,i] * b[:,j] * a[:,summed].conj()), dim=1)
        parts.extend((a.abs().square() / size, b.abs().square() / size,
                      tri.real / math.sqrt(6), tri.imag / math.sqrt(6)))
    elif representation != "ab":
        raise ValueError(representation)
    return torch.cat(parts, dim=1)


@torch.no_grad()
def drifting_field(query, positive, negative, cq, cr, temperatures=(0.3, 0.8)):
    """Independent equal-size reference pools share their list of chemical conditions."""
    if positive.shape != negative.shape or not len(positive):
        raise ValueError("Real and generated reference pools must have the same nonzero size.")
    ref = torch.cat((positive, negative))
    chemical_distance = torch.cdist(cq, cr).square()
    distance = torch.cdist(query, ref)
    fields = []
    for temperature in temperatures:
        log_affinity = -distance / temperature - chemical_distance.repeat(1, 2) / 2
        affinity = (0.5 * (log_affinity.log_softmax(0) + log_affinity.log_softmax(1))).exp()
        wp, wn = affinity.chunk(2, dim=1)
        force = (wp @ positive) * wn.sum(1, keepdim=True) - (wn @ negative) * wp.sum(1, keepdim=True)
        fields.append(force / (force.square().sum(1).mean() + 1e-8).sqrt())
    return torch.stack(fields).mean(0)


@torch.no_grad()
def within_formula_field(query, positive, negative, temperatures=(0.3, 0.8)):
    """First axis indexes formulas. Affinities never mix reference structures across it."""
    references = torch.cat((positive, negative), dim=1)
    distances = torch.cdist(query, references)
    fields = []
    for temperature in temperatures:
        logits = -distances / temperature
        affinity = (0.5 * (logits.log_softmax(1) + logits.log_softmax(2))).exp()
        wp, wn = affinity.chunk(2, dim=2)
        force = (wp @ positive) * wn.sum(2, keepdim=True) - (wn @ negative) * wp.sum(2, keepdim=True)
        fields.append(force / (force.square().sum(-1).mean() + 1e-8).sqrt())
    return torch.stack(fields).mean(0)
