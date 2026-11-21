"""Train the experimental periodic generator with independent crystal reference banks."""

import argparse
import json
from pathlib import Path

import torch

from .generator import PeriodicGenerator
from .kernel import feature_drift, periodic_features


def drifting_loss(model, positives, query_seeds, negative_seeds, species, cutoff, n_radial=8):
    """Compute one exact-composition bank loss. Each crystal is (cell, frac, z)."""
    if not query_seeds or not positives or len(positives) != len(negative_seeds):
        raise ValueError("Use nonempty query seeds and equally sized positive and negative banks.")
    composition = positives[0][2].sort().values
    for bank in (positives, query_seeds, negative_seeds):
        if any(not torch.equal(z.sort().values, composition) for _, _, z in bank):
            raise ValueError("All crystals in this training bank must have the same composition.")

    def encode(crystal, generate=False):
        cell, frac, z = crystal
        if generate:
            cell, frac = model(cell, frac, z)
        return periodic_features(cell, frac, z, species, cutoff, n_radial)

    queries = torch.stack([encode(seed, generate=True) for seed in query_seeds])
    with torch.no_grad():
        pos = torch.stack([encode(crystal) for crystal in positives])
        neg = torch.stack([encode(seed, generate=True) for seed in negative_seeds])
        field = feature_drift(queries.detach(), pos, neg)
        target = queries.detach() + field
    return 0.5 * (queries - target).square().sum(dim=-1).mean()


def physical_seed(z, rng, volume_per_site=18.0):
    """Draw one seed in a chosen basis and a random Cartesian frame, for the CPU demo."""
    if volume_per_site <= 0:
        raise ValueError("Seed volume per site must be positive.")
    noise = torch.randn((3, 3), generator=rng, dtype=torch.float64)
    stretch = torch.linalg.matrix_exp(0.1 * (noise + noise.T))
    frame, triangular = torch.linalg.qr(torch.randn((3, 3), generator=rng, dtype=noise.dtype))
    frame = frame * torch.diag(triangular).sign()
    cell = (volume_per_site * len(z)) ** (1 / 3) * (stretch @ frame)
    frac = torch.rand((len(z), 3), generator=rng, dtype=noise.dtype)
    return cell, frac, z


def run_mp20_smoke(data, updates, seed):
    """Exercise optimizer and sampler on training records only. Returns diagnostics."""
    import numpy as np

    from data import unpack_gram

    if updates < 1:
        raise ValueError("At least one optimizer update is required.")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed + 1)
    cutoff = 4.0
    model = PeriodicGenerator(cutoff=cutoff, width=24).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with np.load(data, allow_pickle=False) as records:
        z_rows, f_rows, k_rows = (records[k] for k in ("z", "f", "k"))
        patterns, labels, counts = np.unique(z_rows, axis=0, return_inverse=True, return_counts=True)
        eligible = [i for i, row in enumerate(patterns) if counts[i] >= 4
                    and 3 <= np.count_nonzero(row) <= 8 and len(set(row[row > 0])) >= 2]
        if not eligible:
            raise ValueError("Expected training groups with at least four records of the same composition.")
        gradients, record_ids = [], set()
        for _ in range(updates):
            group = eligible[torch.randint(len(eligible), (), generator=rng).item()]
            members = np.flatnonzero(labels == group)
            indices = members[torch.randperm(len(members), generator=rng)[:4].numpy()]
            z = torch.tensor(patterns[group][patterns[group] > 0], dtype=torch.long)
            vocabulary = tuple(z.unique().tolist())
            positives = []
            for idx in indices:
                cell = torch.from_numpy(np.linalg.cholesky(unpack_gram(k_rows[idx])))
                frac = torch.tensor(f_rows[idx, :len(z)], dtype=cell.dtype)
                positives.append((cell, frac, z))
                record_ids.add(str(records["ids"][idx]))
            queries = [physical_seed(z, rng) for _ in indices]
            negatives = [physical_seed(z, rng) for _ in indices]
            optimizer.zero_grad(set_to_none=True)
            loss = drifting_loss(model, positives, queries, negatives, vocabulary, cutoff)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
            gradients.append(float(norm))
            optimizer.step()
    model.eval()
    cell, frac, z = physical_seed(z, rng)
    with torch.no_grad():
        cell, frac = model(cell, frac, z)
    if not (torch.isfinite(cell).all() and torch.all((frac >= 0) & (frac < 1))):
        raise RuntimeError("Sampler produced an invalid numerical output.")
    return dict(optimizer_updates=updates, distinct_training_records=len(record_ids),
                gradient_norms=gradients, output_sites=len(frac),
                output_volume=float(torch.linalg.det(cell).abs()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = Path(__file__).resolve().parents[1] / "scaled" / "mp20-data" / "train.npz"
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(run_mp20_smoke(args.data, args.updates, args.seed), indent=2))
